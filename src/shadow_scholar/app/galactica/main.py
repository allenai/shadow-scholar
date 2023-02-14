from contextlib import contextmanager
from typing import Tuple
import logging

from shadow_scholar.cli import safe_import, cli, Argument

with safe_import():
    import torch
    import gradio as gr
    from transformers import AutoTokenizer, OPTForCausalLM


LOGGING = logging.getLogger(__name__)


class _gl_model:

    @classmethod
    def get_model_spec(cls, model_name: str) -> Tuple[str, 'torch.dtype']:
        if model_name == "mini":
            return "facebook/galactica-125m", torch.float32
        elif model_name == "base":
            return "facebook/galactica-1.3b", torch.float32
        elif model_name == "standard":
            return "facebook/galactica-6.7b", torch.float32
        elif model_name == "large":
            return "facebook/galactica-30b", torch.float32
        elif model_name == "huge":
            return "facebook/galactica-120b", torch.float16
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def __init__(
        self,
        model_name: str,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.backbone, self.dtype = self.get_model_spec(model_name)

        print(f"Loading model {self.backbone} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.backbone
        )
        self.model = OPTForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.backbone, #device_map="auto"
        ).to(self.device).eval()    # pyright: ignore

        # self.model = AutoModelForCausalLM.from_pretrained(
        #     pretrained_model_name_or_path=pretrained_model_name_or_path,
        # )
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     pretrained_model_name_or_path=pretrained_model_name_or_path,
        #     pad_token_id=1,
        #     padding_side='left',
        #     model_max_length=4096,
        # )

    @contextmanager
    def autocast(self):
        try:
            if self.device == "cuda" and self.dtype != torch.float32:
                with torch.cuda.amp.autocast(enabled=True):  # pyright: ignore
                    yield
            else:
                yield
        finally:
            pass

    def __call__(self, text: str) -> str:
        with torch.no_grad(), self.autocast():
            batch = self.tokenizer(text, return_tensors="pt")
            batch = batch.to(self.device)
            outputs = self.model.generate(
                input_ids=batch.input_ids,
                max_length=4096
            )
        return self.tokenizer.decode(outputs[0])


@cli(
    "app.galactica",
    arguments=[
        Argument(
            '-m', '--model-name',
            default='facebook/galactica-125m',
            help='Pretrained model or path to local checkpoint'
        ),
    ],
    requirements=['gradio', 'transformers', 'torch', 'accelerate']
)
def run_galactica_demo(model_name: str):
    gl_model = _gl_model(model_name=model_name)

    demo = gr.Interface(fn=gl_model, inputs="text", outputs="text")

    demo.launch()
