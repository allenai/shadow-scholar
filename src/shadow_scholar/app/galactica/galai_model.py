"""
Code from adapted from https://github.com/paperswithcode/galai/blob/1a1e4486fd38c85c6b66b3c0b31ce24b0afd7613/galai/model.py

Adapted by @soldni

"""  # noqa: E501

import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union, cast

from shadow_scholar.cli import safe_import

from .galai_utils import escape_custom_split_sequence

with safe_import():
    import psutil
    import torch
    from accelerate import infer_auto_device_map, init_empty_weights
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        OPTConfig,
        OPTForCausalLM,
        PreTrainedTokenizerBase,
    )


class Model:
    """
    Model class holding the GALACTICA models. We configure a class to
    encapsulate the HuggingFace model, the tokenizer, and the specific
    tokenization logic for GALACTICA. For low-level access, we recommend
    using the standard HuggingFace API.
    """

    model: "OPTForCausalLM"
    tokenizer: "PreTrainedTokenizerBase"

    def __init__(
        self,
        name: str,
        precision: Literal["full", "mixed"] = "full",
        tensor_parallel: bool = False,
    ):
        """
        Initializes a new model

        Args:
            name (str): The name of the model to load. One of
                `mini`, `base`, `standard`, `large`, `huge`.
            precision (str): The precision to use for the model. One of
                `full` or `mixed`. `full` uses 32-bit floating point
                numbers for the model weights and `mixed` uses 16-bit
                floating point numbers (bfloat16).
            tensor_parallel (bool): Whether to use tensor parallelism
                to load the model. This is only supported on NVIDIA
                GPUs at requires 2 or more GPUs.
        """

        self.name, self.dtype = self.variants[name]
        if precision == "mixed":
            self.dtype = torch.bfloat16

        self.is_loaded = False
        self.num_gpus = (
            torch.cuda.device_count()
            if tensor_parallel
            else (1 if torch.cuda.is_available() else 0)
        )
        self.tensor_parallel = tensor_parallel
        self.max_input_length = 2048
        self._master_port: Union[int, None] = None

        self._set_tokenizer(self.name)
        self._load_checkpoint(self.name)

    @property
    def variants(cls) -> Dict[str, Tuple[str, "torch.dtype"]]:
        return {
            "mini": ("facebook/galactica-125m", torch.float32),
            "base": ("facebook/galactica-1.3b", torch.float32),
            "standard": ("facebook/galactica-6.7b", torch.float32),
            "large": ("facebook/galactica-30b", torch.float32),
            "huge": ("facebook/galactica-120b", torch.float16),
        }

    def __str__(self):
        return f"model `{self.name}` with type `{self.dtype}`"

    def _load_checkpoint(self, checkpoint_path: str):
        if self.tensor_parallel:
            config = OPTConfig.from_pretrained(self.name)
            with init_empty_weights():
                empty_model = AutoModelForCausalLM.from_config(config)

            from ._old import _gpu_mem

            device_map = infer_auto_device_map(
                empty_model,
                max_memory={
                    gid: f"{mem * .7:.0f}MB" for gid, mem in _gpu_mem().items()
                },
                no_split_module_classes=[
                    "OPTDecoderLayer",
                    "Linear" "Embedding",
                    "OPTLearnedPositionalEmbedding" "LayerNorm",
                ],
                dtype=self.dtype,
            )

            # HACK: we actually want to have "lm_head" on the first GPU
            #       because its weights are tied to the embedding layer
            #       and we want to keep them on the same device.
            if "lm_head" in device_map:
                device_map["lm_head"] = 0
            assert (
                device_map.get(
                    "model.decoder.embed_tokens",
                    0,
                )
                == device_map.get(
                    "model.decoder.embed_positions",
                    0,
                )
                == device_map.get("model.decoder.final_layer_norm", 0)
                == device_map["lm_head"]
            ), "Embedding layers and lm_head should be on the same device."

            self.model = OPTForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.name,
                device_map=device_map,
                offload_folder="offload",
                offload_state_dict=True,
                torch_dtype=self.dtype,
            ).eval()  # pyright: ignore

        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = (
                OPTForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=self.name,
                    torch_dtype=self.dtype,
                )
                .to(device)  # pyright: ignore
                .eval()
            )

    def _buggy_load_checkpoint(self, checkpoint_path: str):
        """
        Loads the checkpoint for the model

        Parameters
        ----------
        checkpoint_path : str
            Path for the checkpoint (str)
        """

        # query available memory size of the GPUs we want to use.
        # If tensor_parallel is True, we just load the model's weights to RAM,
        # as it needs to be sliced by parallelformers before loading to VRAM.
        device_map = None
        max_memory: Dict[Union[str, int], int] = {}
        if self.num_gpus > 0 and not self.tensor_parallel:
            # based on https://github.com/huggingface/accelerate/blob/5315290b55ea9babd95a281a27c51d87b89d7c85/src/accelerate/utils/modeling.py#L274    # noqa: E501
            for i in range(self.num_gpus):
                _ = torch.tensor([0], device=i)
            for i in range(self.num_gpus):
                max_memory[i] = torch.cuda.mem_get_info(i)[0]
            device_map = "auto"
        max_memory["cpu"] = psutil.virtual_memory().available

        self.model = OPTForCausalLM.from_pretrained(  # pyright: ignore
            checkpoint_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
            max_memory=max_memory,
        )
        self.model.eval()  # pyright: ignore

        if self.tensor_parallel:
            self._parallelize()

    def _parallelize(self) -> None:
        """
        Parallelize the model for a tensor-parallel multi-GPU inference.
        """

        if self.num_gpus < 2:
            raise RuntimeError(
                "At least two GPUs are required to parallelize the model."
            )

        self._master_port = 13000 + (id(self.model) % 32749)

        custom_policies = None
        if (
            self.model.config.model_type == "opt"
            and not self.model.config.enable_bias
        ):
            from .galai_parallel_policy import OPTDecoderLayerPolicyNoBias

            custom_policies = [OPTDecoderLayerPolicyNoBias]

        from parallelformers import parallelize

        parallelize(
            self.model,
            num_gpus=self.num_gpus,
            fp16=self.dtype == torch.float16,
            master_port=self._master_port,
            custom_policies=custom_policies,
        )

    def _set_tokenizer(self, tokenizer_path: str):
        """
        Configures the tokenizer for the model

        Parameters
        ----------
        tokenizer_path : str
            Path for the tokenizer (str)
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # setup padding
        tokenizer.pad_token_id = 1
        tokenizer.pad_token = "<pad>"
        tokenizer.padding_side = "left"

        # setup truncation
        tokenizer.truncation_side = "left"

        # setup special tokens
        tokenizer.bos_token_id = 0
        tokenizer.bos_token = "<s>"

        tokenizer.eos_token_id = 2
        tokenizer.eos_token = "</s>"

        tokenizer.unk_token = "<unk>"
        tokenizer.unk_token_id = 3

        self.tokenizer = tokenizer

    def _tokenize(
        self, input_text: List[str], new_doc: bool
    ) -> "torch.LongTensor":
        """
        Apply custom preprocessing to input texts and tokenize them.

        Args:
            input_text (List[str]): Texts to be tokenized
            new_doc (bool): If True, prepends the end-of-document (</s>)
                token to each sequence.
        """
        texts = []
        for text in input_text:
            text = escape_custom_split_sequence(text)
            if not text:
                warnings.warn(
                    "Found an empty input text. Changing to "
                    + "end-of-document token instead.",
                    UserWarning,
                )
                text = self.tokenizer.eos_token
            texts.append(text)

        if new_doc:
            pad_token = self.tokenizer.pad_token
            texts = [pad_token + t for t in texts]

        encoded = self.tokenizer(
            texts,
            padding="longest",
            max_length=self.max_input_length,
            truncation=True,
            pad_to_multiple_of=8,
        )
        context_tokens = encoded["input_ids"]
        input_v = cast(
            torch.LongTensor,
            torch.LongTensor(context_tokens).to(self.model.device),
        )

        if new_doc:
            loc = input_v[:, 0] == self.tokenizer.pad_token_id
            input_v[loc, 0] = cast(int, self.tokenizer.eos_token_id)
        return input_v

    @torch.inference_mode()
    def generate(
        self,
        input_text: Union[str, List[str]],
        max_new_tokens: int = 128,
        new_doc: bool = False,
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        penalty_alpha: Optional[float] = None,
        num_beams: int = 1,
        num_return_sequences: int = 1,
        return_full_text: bool = True,
    ) -> Union[str, List[str], List[List[str]]]:
        """
        Generates text using the model

        Args:
            input_text (Union[str, List[str]]): Input context for the model
                to use for its generation, e.g. "Attention Is All You Need
                [START_REF]"
            max_length (int, optional): Maximum length in tokens of the
                generated text (including prompt). Only one of max_length
                and max_new_tokens should be specified. If neither is set,
                then max_new_tokens is set to 60.
            max_new_tokens (int, optional): Maximum length in tokens of the
                generated text (excluding prompt). Only one of max_length
                and max_new_tokens should be specified. If neither is set,
                then max_new_tokens is set to 60.
            new_doc (bool): If True, treats generation a new document,
                otherwise assumes generation could be anywhere within
                document. Use new_doc=True if you are generating documents,
                e.g. # Schwarzschild Radius, # Transformer (machine
                learning), Title: Transformers, A Survey. For general
                generation, set new_doc=False.
            top_p (float, optional): If set, the model will use nucleus
                sampling with this probability. This means that the model
                will sample from the most likely tokens until the sum of
                their probabilities exceeds this value. This is useful for
                generating more diverse outputs. Default is None, which
                means no nucleus sampling is used.
            top_k (int, optional): If set, the model will use top-k
                sampling with this value. This means that the model will
                sample from the top-k most likely tokens. This is useful
                for generating more diverse outputs. Default is None, which
                means no top-k sampling is used.
            penalty_alpha (float, optional): If set, the model will use
                penalty-based sampling with this value. This means that the
                model will sample from the most likely tokens, but will
                penalize tokens that have already been generated. This is
                useful for generating more diverse outputs. Default is
                None, which means no penalty-based sampling is used.
            num_beams (int, optional): If set, the model will use beam
                search with this value. This means that the model will
                generate this many sequences at once, and then return the
                best one. This is useful for generating more diverse
                outputs. Default is 1, which means no beam search is used.
            num_return_sequences (int, optional): If set, the model will
                return this many sequences. Default is 1, which means only
                one sequence is returned.
            return_full_text (bool, optional): If True, the model will
                return the full text of the generated sequence, including
                the prompt. If False, the model will return only the
                generated sequence. Default is True.
        """

        texts = [input_text] if isinstance(input_text, str) else input_text
        input_v = self._tokenize(texts, new_doc)
        options = {}
        if penalty_alpha is not None:
            options["penalty_alpha"] = float(penalty_alpha)
            options["top_k"] = float(top_k) if top_k is not None else 0
        elif top_p is not None:
            options["do_sample"] = True
            options["top_p"] = float(top_p)
        elif top_k is not None:
            options["do_sample"] = True
            options["top_k"] = float(top_k)

        out = self.model.generate(
            input_v,
            max_new_tokens=int(max_new_tokens),
            return_dict_in_generate=True,
            output_hidden_states=False,
            num_beams=int(num_beams),
            num_return_sequences=int(num_return_sequences),
            **options,
        )

        out_tokens = out["sequences"]  # pyright: ignore
        if not return_full_text:
            out_tokens = out_tokens[:, input_v.shape[1] :]
        # we keep special tokens such as [START_REF] or <work>
        decoded = self.tokenizer.batch_decode(
            out_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        # so we manually remove </s> and <pad>
        decoded = [
            text.replace(self.tokenizer.eos_token, "").replace(
                self.tokenizer.pad_token, ""
            )
            for text in decoded
        ]

        if num_return_sequences == 1:
            return decoded[0] if isinstance(input_text, str) else decoded
        if isinstance(input_text, str):
            return decoded
        else:
            return [
                decoded[
                    num_return_sequences * i : num_return_sequences * (i + 1)
                ]
                for i in range(len(texts))
            ]
