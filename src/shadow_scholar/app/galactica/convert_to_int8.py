"""
https://gist.github.com/ibeltagy/23c1d71f2494837e91dcecd613c8b5cb

Original by @ibeltagy
Modified by @soldni
"""

from typing import List, Optional

from shadow_scholar.cli import safe_import

with safe_import():
    import torch

    # from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM, AutoTokenizer


# def replace_8bit_linear(
#     model, threshold=6.0, module_to_not_convert="lm_head"
# ):
#     for name, module in model.named_children():
#         if len(list(module.children())) > 0:
#             replace_8bit_linear(module, threshold, module_to_not_convert)

#         if (
#             isinstance(module, torch.nn.Linear)
#             and name != module_to_not_convert
#         ):
#             with init_empty_weights():
#                 model._modules[name] = bnb.nn.Linear8bitLt(
#                     module.in_features,
#                     module.out_features,
#                     module.bias is not None,
#                     has_fp16_weights=False,
#                     threshold=threshold,
#                 )
#     return model


def assert_all_approx_close(a, b, rtol, atol, count):

    idx = torch.isclose(a.float(), b.float(), rtol, atol)
    sumval = (idx == 0).sum().item()
    if sumval > count:
        print(f"Too many values not close: assert {sumval} < {count}")
        try:
            torch.testing.assert_allclose(a, b, rtol, atol)
        except Exception as e:
            print(e)


def get_memory_footprint(model, return_buffers=True):
    """
    Get the memory footprint of a model. This will return the memory footprint
    of the current model in bytes. Useful to benchmark the memory footprint of
    the current model and design some tests. Solution inspired from the
    PyTorch discussions:
    https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2

    Arguments:
        return_buffers (`bool`, *optional*, defaults to `True`): Whether to
            return the size of the buffer tensors in the computation of the
            memory footprint. Buffers are tensors that do not require gradients
            and not registered as parameters. E.g. mean and std in batch norm
            layers.
    """
    mem = sum(
        [
            param.nelement() * param.element_size()
            for param in model.parameters()
        ]
    )
    if return_buffers:
        mem_bufs = sum(
            [buf.nelement() * buf.element_size() for buf in model.buffers()]
        )
        mem = mem + mem_bufs
    return mem


def replace_linear_with_int8(
    model: "torch.nn.Module", do_not_convert: Optional[List[str]] = None
):
    do_not_convert = do_not_convert or ["lm_head"]

    names = [name for name, _ in model.named_modules()]

    for name in names:
        if any(name.startswith(e) for e in do_not_convert):
            continue

        module = model.get_submodule(name)

        if isinstance(module, torch.nn.Linear):
            new_module = QuantizedLinearInt8(layer=module)
            parent_name, module_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            delattr(parent, module_name)
            setattr(parent, module_name, new_module)

        torch.cuda.empty_cache()


class QuantizedLinearInt8(torch.nn.Module):
    """
    A simple but effictive implmenetion of Int8 quantization for linear layers.
    The weights are quantized and stored as Int8, which saves ~50% of the gpu
    memory. During the forwared pass, the weights are de-quantized back to fp16
    to do multiplication.

    Pros:
        - saves ~50% of the gpu memory
        - accurate quantization because only the weights are quantized, and
            the weights don't suffer from the "outliers" issue mentioned in
            the LLM.int8 paper; only the activations do.
        - high precision results beacuse the multiplication is done in fp16
        - much faster than LLM.int8
    Cons:
        - a bit slower because of the added computation of dequantization
            in each forward pass. In practice, the slowdown is not large
            because in the generation application, gpu utilization is not
            very high.
    """

    def __init__(self, layer, weight_bit_width: int = 8):
        super().__init__()

        self.bias = torch.nn.Parameter(
            layer.bias.clone(), requires_grad=layer.bias.requires_grad
        )

        self.weight_scale = torch.nn.Parameter(
            (
                layer.weight.abs().max(dim=-1).values
                / ((2 ** (weight_bit_width - 1)) - 1)
            ).half(),
        )

        self.weight = torch.nn.Parameter(
            torch.round(
                layer.weight.float() / self.weight_scale[:, None]
            ).char(),
            requires_grad=False,
        )

    def forward(self, x):

        # TODO: the following two lines can be combined in a single more
        # efficient command. One way to do so is torch.einsum (as in the
        # commented code below), but the order of operations in einsum is
        # arbitrary, so it can multiply `x * self.weight` before
        # `self.weight_scale`, which results in higher loss of precision.

        # The dequantization is two steps:
        # 1) `self.weight.half()`: which casts the weights back to fp16
        # 2) `* self.weight_scale[:, None]` multiply the weights by the
        #       quantization weight scales
        # Profiling shows that (1) is the main reason for the slowdown, and
        # that the overhead from (2) is very small

        weight = self.weight.half() * self.weight_scale[:, None]
        return torch.nn.functional.linear(x, weight, self.bias)

        # if x.dim() == 2:
        #     result = torch.einsum(
        #         "sh, dh, d -> sd", x, self.weight.half(), self.weight_scale
        #     )
        # elif x.dim() == 3:
        #     result = torch.einsum(
        #         "bsh, dh, d -> bsd", x, self.weight.half(), self.weight_scale
        #     )
        # else:
        #     # Not supported
        #     print(x)
        #     print(x.shape)
        #     assert False

        # if self.bias is not None:
        #     result += self.bias
        # return result


def convert_model_to_int8_on_gpu(model, device):
    """
    Quantize a model to int8 and move it to GPU using a simple method.
    """
    if "cuda" not in device:
        raise ValueError(
            f"Target device should be a gpu. Device {device} is not supported"
        )

    model.half()

    memory_before_quantization = get_memory_footprint(model)  # without lm_head

    # replace `Linear` with `QuantizedLinearInt8`
    replace_linear_with_int8(model)

    model.to(device=device)
    memory_after_quantization = get_memory_footprint(model)  # without lm_head

    saving = round(
        100 * memory_after_quantization / memory_before_quantization
    )
    memory_before_quantization = round(
        memory_before_quantization / 2**30, 2
    )  # rounding for printing
    memory_after_quantization = round(
        memory_after_quantization / 2**30, 2
    )  # rounding for printing

    print(
        f"Quantization memory - before: {memory_before_quantization} GB, "
        f"after: {memory_after_quantization} GB ({saving}% of the size before)"
    )
    return model


def main():
    model_name = "facebook/opt-350m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer.encode(
        "The sky is", return_tensors="pt"
    ).cuda()  # pyright: ignore

    model_nonint8 = AutoModelForCausalLM.from_pretrained(model_name)
    model_nonint8 = model_nonint8.half().cuda()

    model_int8 = AutoModelForCausalLM.from_pretrained(model_name)
    model_int8 = convert_model_to_int8_on_gpu(model_int8, device="cuda")

    with torch.inference_mode():
        logits_nonint8 = model_nonint8(input_ids=input_ids).logits
        logits_int8 = model_int8(input_ids=input_ids).logits
        print(torch.mean(abs(logits_int8 - logits_nonint8)).detach().item())
        assert_all_approx_close(
            logits_int8, logits_nonint8, rtol=0.01, atol=3.0e-2, count=30
        )

        generated_nonint8 = model_nonint8.generate(input_ids)
        generated_int8 = model_int8.generate(input_ids)
        print("=============")
        print(tokenizer.decode(generated_nonint8[0]))
        print("=============")
        print(tokenizer.decode(generated_int8[0]))
        print("=============")

        print(
            "Memory of non-int8 model: "
            f"{get_memory_footprint(model_nonint8) / 10**9}G"
        )
        print(
            f"Memory of int8 model: "
            f"{get_memory_footprint(model_int8) / 10**9}G"
        )


if __name__ == "__main__":
    main()
