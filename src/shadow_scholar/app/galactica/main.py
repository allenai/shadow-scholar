from contextlib import contextmanager
import re
from typing import Iterator, List, Literal, Tuple, Union, Dict, Optional
import logging
from ast import literal_eval

from shadow_scholar.cli import safe_import, cli, Argument
from .constants import INSTRUCTIONS, CSS

with safe_import():
    import torch
    import gradio as gr
    from transformers import (
        AutoTokenizer,
        OPTForCausalLM,
        OPTConfig,
        AutoModelForCausalLM
    )
    from accelerate import (
        init_empty_weights,
        infer_auto_device_map
    )


LOGGING = logging.getLogger(__name__)


def get_submodule(model: "torch.nn.Module", target: str) -> "torch.nn.Module":
    ...
    for part in target.split("."):
        if re.match(r"\d+", part):
            model = model[int(part)]    # pyright: ignore
        else:
            model = getattr(model, part)
    return model


def move_tensors(
    module: "torch.nn.Module",
    input: tuple,
    output: Union[Tuple["torch.Tensor", ...], "torch.Tensor"],
    device: str
) -> Union[Tuple["torch.Tensor", ...], "torch.Tensor"]:
    if isinstance(output, torch.Tensor):
        return output.to(device)
    return tuple(
        move_tensors(module, input, o, device)  # pyright: ignore
        for o in output
    )


def _gpu_mem() -> Dict[int, int]:
    """Return the amount of memory available on GPUs in MB, rounded to the
    nearest integer."""
    return {
        gid: round(
            torch.cuda.get_device_properties(gid).total_memory / 1024 / 1024
        ) for gid in range(torch.cuda.device_count())
    }


class _gl_model:
    @property
    def variants(cls) -> Dict[str, Tuple[str, 'torch.dtype']]:
        return {
            "mini": ("facebook/galactica-125m", torch.float32),
            "base": ("facebook/galactica-1.3b", torch.float32),
            "standard": ("facebook/galactica-6.7b", torch.float32),
            "large": ("facebook/galactica-30b", torch.float32),
            "huge": ("facebook/galactica-120b", torch.float16),
        }

    def __str__(self):
        return (
            f"model `{self.backbone}` on `{self.device}` "
            f"with type `{self.dtype}`"
        )

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        precision: Literal['full', 'mixed', 'int8'] = 'full',
        use_accelerate: bool = False,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        assert model_name in self.variants, (
            f"Unknown model name: {model_name}. "
            f"Available models: {', '.join(self.variants.keys())}"
        )
        self.backbone, self.dtype = self.variants[model_name]

        if precision == 'full':
            if self.dtype in {torch.float16, torch.int8, torch.bfloat16}:
                LOGGING.warning(
                    f"Model is only available in {self.dtype}. "
                    f"Ignoring `precision={precision}` argument."
                )
        elif precision == 'mixed':
            if self.dtype in {torch.int8, torch.float16}:
                LOGGING.warning(
                    f"Model is only available in {self.dtype}. "
                    f"Ignoring `precision={precision}` argument."
                )
            self.dtype = torch.bfloat16
        elif precision == 'int8':
            raise NotImplementedError("Int8 is not supported yet.")
        else:
            raise ValueError(f"Unknown precision: {precision}")

        print(f"Loading {self}...", end='', flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.backbone,
        )

        # from the galactica repo: https://github.com/paperswithcode/galai/blob/1a1e4486fd38c85c6b66b3c0b31ce24b0afd7613/galai/model.py#L154  # noqa: E501

        # setup padding
        self.tokenizer.pad_token_id = 1
        self.tokenizer.pad_token = "<pad>"
        self.tokenizer.padding_side = "left"

        # setup truncation
        self.tokenizer.truncation_side = "left"

        # setup special tokens
        self.tokenizer.bos_token_id = 0
        self.tokenizer.bos_token = "<s>"

        self.tokenizer.eos_token_id = 2
        self.tokenizer.eos_token = "</s>"

        self.tokenizer.unk_token = "<unk>"
        self.tokenizer.unk_token_id = 3

        if use_accelerate:
            config = OPTConfig.from_pretrained(self.backbone)
            with init_empty_weights():
                empty_model = AutoModelForCausalLM.from_config(config)

            device_map = infer_auto_device_map(
                empty_model,
                max_memory={
                    gid: f'{mem * .7:.0f}MB'
                    for gid, mem in _gpu_mem().items()
                },
                no_split_module_classes=[
                    "OPTDecoderLayer",
                    "Linear"
                    "Embedding",
                    "OPTLearnedPositionalEmbedding"
                    "LayerNorm"
                ],
                dtype=self.dtype,
            )

            # HACK: we actually want to have "lm_head" on the first GPU
            #       because its weights are tied to the embedding layer
            #       and we want to keep them on the same device.
            if 'lm_head' in device_map:
                device_map['lm_head'] = 0
            assert device_map.get(
                'model.decoder.embed_tokens', 0,
            ) == device_map.get(
                'model.decoder.embed_positions', 0,
            ) == device_map.get(
                'model.decoder.final_layer_norm', 0
            ) == device_map['lm_head'], \
                "Embedding layers and lm_head should be on the same device."

            self.model = OPTForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_path or self.backbone,
                device_map=device_map,
                offload_folder="offload",
                offload_state_dict=True,
                torch_dtype=self.dtype
            ).eval()    # pyright: ignore

        else:
            self.model = OPTForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_path or self.backbone,
                torch_dtype=self.dtype,
            ).to(self.device).eval()    # pyright: ignore

        print("done.", flush=True)

    @contextmanager
    def autocast(self) -> Iterator[bool]:
        try:
            if self.device == "cuda" and self.dtype != torch.float32:
                with torch.cuda.amp.autocast(enabled=True):  # pyright: ignore
                    yield True
            else:
                yield False
        finally:
            pass

    def __call__(
        self,
        text: str,
        tokenize_config: Union[dict, List[Tuple[str, str]], None] = None,
        generate_config: Union[dict, List[Tuple[str, str]], None] = None,
    ) -> str:

        if not isinstance(tokenize_config, dict):
            tokenize_config = {
                k: literal_eval(v) for k, v in (tokenize_config or [])
                if k.strip() != '' and v.strip() != ''
            }

        if not isinstance(generate_config, dict):
            generate_config = {
                k: literal_eval(v) for k, v in (generate_config or [])
                if k.strip() != '' and v.strip() != ''
            }

        batch = self.tokenizer(
            text,
            return_tensors="pt",
            **tokenize_config
        )
        casted_batch = batch.to(self.device)

        with self.autocast(), torch.no_grad():
            outputs = self.model.generate(
                input_ids=casted_batch.input_ids,
                attention_mask=casted_batch.attention_mask,
                **generate_config,
            )

        # following instructions from the galactica repo: https://github.com/paperswithcode/galai/blob/1a1e4486fd38c85c6b66b3c0b31ce24b0afd7613/galai/model.py#L308     # noqa: E501
        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )
        decoded = decoded.replace(
            self.tokenizer.eos_token, ""
        ).replace(self.tokenizer.pad_token, "")

        return decoded


@cli(
    "app.galactica",
    arguments=[
        Argument(
            '-m', '--model-name',
            default='facebook/galactica-125m',
            help='Pretrained model or path to local checkpoint'
        ),
        Argument(
            '-p', '--server-port',
            default=7860,
            help='Port to run the server on'
        ),
        Argument(
            '-n', '--server-name',
            default='localhost',
            help='Server address to run the gradio app at'
        ),
        Argument(
            '-e', '--precision',
            default='full',
            help='Precision to use for the model',
            choices=['full', 'mixed', 'int8']
        ),
        Argument(
            '-l', '--model-path',
            default=None,
            help='Path to local checkpoint',
        ),
        Argument(
            '-a', '--use-accelerate',
            default=False,
            action='store_true',
            help='Enable HuggingFace Accelerate to use on multiple GPUs'
        )

    ],
    requirements=['gradio', 'transformers', 'torch', 'accelerate']
)
def run_galactica_demo(
    model_name: str,
    server_port: int,
    server_name: str,
    precision: Literal['full', 'mixed', 'int8'],
    model_path: Optional[str] = None,
    use_accelerate: bool = False,
):

    gl_model = _gl_model(
        model_name=model_name,
        model_path=model_path,
        precision=precision,
        use_accelerate=use_accelerate,
    )

    with gr.Blocks(css=CSS) as demo:
        with gr.Row():
            gr.Markdown(f"# Galactica {model_name} Demo")
        with gr.Tab("Demo"):
            with gr.Row():
                gr.Markdown(
                    f"**Currently loaded**: {gl_model}\n" +
                    "Available models:\n" +
                    "\n".join(
                        f" - {k}: `{n}`"
                        for k, (n, _) in gl_model.variants.items()
                    )
                )
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        lines=5, label="Input", placeholder="Prompt text"
                    )
                    submit_button = gr.Button(label="Generate")
                    tokenizer_options = gr.inputs.Dataframe(
                        label=(
                            "Parameters for `Tokenizer.batch_encode()`; "
                            "values should be Python literals."
                        ),
                        headers=['Parameter', 'Value'],
                        type='array',
                        col_count=2,
                        default=[
                            ['max_length', '512'],
                            ['truncation', 'True'],
                            ['pad_to_multiple_of', '32'],
                        ]
                    )
                    generation_options = gr.inputs.Dataframe(
                        label=(
                            'Parameters for `GenerationMixin.generate()`; '
                            'values should be Python literals.'
                        ),
                        headers=['Parameter', 'Value'],
                        type='array',
                        col_count=2,
                        default=[
                            ['max_new_tokens', '128'],
                            ['do_sample', 'True'],
                            ['num_beams', '5'],
                            ['top_k', '25'],
                            ['temperature', '.7'],
                            ['top_p', '.9'],
                            ['no_repeat_ngram_size', '10'],
                            ['early_stopping', 'True'],
                            ['num_beam_groups', '1'],
                            ['diversity_penalty', '0.0'],
                        ]
                    )
                    hide_prompt = gr.Checkbox(
                        label="Hide input prompt in generation?",
                        bool=False
                    )
                with gr.Column():
                    output_text = gr.Textbox(
                        lines=5, label="Output", placeholder="Generated text"
                    )

            submit_button.click(
                fn=gl_model,
                inputs=[
                    input_text,
                    tokenizer_options,
                    generation_options,
                    hide_prompt
                ],
                outputs=[output_text]
            )
        with gr.Tab("Instructions"):
            with gr.Row():
                gr.Markdown(INSTRUCTIONS, elem_id="instructions")

    demo.launch(server_name=server_name, server_port=server_port)
