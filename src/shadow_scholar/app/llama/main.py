import json
from ast import literal_eval
from typing import Optional

from shadow_scholar.cli import Argument, cli, safe_import

with safe_import():
    import gradio as gr

    from .facebook_llama.example import setup_model_parallel, load


class ModelWrapper:
    def __init__(
        self,
        name: str,
        model_name: str,
        model_base_path: str,
    ):
        local_rank, world_size = setup_model_parallel()
        if local_rank > 0:
            sys.stdout = open(os.devnull, 'w')

        generator = load(ckpt_dir, tokenizer_path, local_rank, world_size)
        prompts = ["The capital of Germany is the city of", "Here is my sonnet in the style of Shakespeare about an artificial intelligence:"]
        results = generator.generate(prompts, max_gen_len=256, temperature=temperature, top_p=top_p)

    def __str__(self) -> str:
        return str(self.model) + f" start_time={self.start_time}"

    def log(self, arguments, output):
        if self.logdir is None:
            return

        self.logdir.mkdir(parents=True, exist_ok=True)

        fn = f'{self.model.name.replace("/", "_")}_{self.start_time}.jsonl'
        with open(self.logdir / fn, "a") as f:
            f.write(json.dumps({"input": arguments, "output": output}) + "\n")

    def __call__(self, *args, **kwargs):
        arguments = self.signature.bind(*args, **kwargs).arguments

        if isinstance(opt := arguments.pop("extra_options", None), list):
            arguments["extra_options"] = {
                # evaluate strings as python literals
                k: literal_eval(v)
                for k, v in opt
                # no empty strings
                if k.strip() and v.strip()
            }

        arguments["top_k"] = (
            int(top_k) if (top_k := arguments.pop("top_k", None)) else None
        )

        arguments["top_p"] = (
            float(top_p) if (top_p := arguments.pop("top_p", None)) else None
        )

        arguments["penalty_alpha"] = (
            float(pa) if (pa := arguments.pop("penalty_alpha", None)) else None
        )

        output = self.model.generate(**arguments)
        self.log(arguments, output)
        return output


def draw_ui(model_name: str, model_wrapper: ModelWrapper):
    with gr.Blocks(css=CSS) as demo:
        with gr.Row():
            gr.Markdown(f"# LLaMA {model_name} Demo")
        # with gr.Tab("Demo"):
            # with gr.Row():
            #     gr.Markdown(
            #         f"**Currently loaded**: {gl_model}\n\n"
            #         + "Available models:\n"
            #         + "\n".join(
            #             f" - {k.capitalize()}: `{n}`"
            #             for k, (n, _) in gl_model.model.variants.items()
            #         )
            #     )
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    lines=5, label="Input", placeholder="Prompt text"
                )
                submit_button = gr.Button(label="Generate")

                max_new_tokens = gr.Number(
                    value=128,
                    label=(
                        "max_new_tokens: max number of new tokens "
                        "the model should generate"
                    ),
                )
                top_p = gr.Textbox(
                    value="",
                    label=(
                        "top_p: if set to float < 1, only the "
                        "smallest set of most probable tokens with "
                        "probabilities that add up to top_p or higher "
                        "are kept for generation."
                    ),
                )
                top_k = gr.Textbox(
                    value="",
                    label=(
                        "top_k: size of the candidate set that is "
                        "used to re-rank for contrastive search"
                    ),
                )
                penalty_alpha = gr.Textbox(
                    value="",
                    label=(
                        "penalty_alpha: degeneration penalty for "
                        "contrastive search"
                    ),
                )

                num_beams = gr.Number(
                    value=1,
                    label=(
                        "num_beams: number of beams for beam search. "
                        "1 means no beam search."
                    ),
                )
                num_return_sequences = gr.Number(
                    value=1,
                    label=(
                        "num_return_sequences: number of separate"
                        "computed returned sequences for each element "
                        "in the batch."
                    ),
                )
                return_full_text = gr.Checkbox(
                    label=(
                        "return_full_text: whether to return the full "
                        "text or just the newly generated text."
                    ),
                    value=True,
                )
                new_doc = gr.Checkbox(
                    label=(
                        "new_doc: whether the model should attempt "
                        "to generate a full document"
                    ),
                    value=False,
                )
                extra_options = gr.Dataframe(
                    label="Extra options to pass to model.generate()",
                    headers=["Parameter", "Value"],
                    col_count=2,
                    type="array",
                    interactive=True,
                )

            with gr.Column():
                output_text = gr.Textbox(
                    lines=25,
                    label="Output",
                    placeholder="Generated text",
                    interactive=False,
                )

        submit_button.click(
            fn=model_wrapper,
            inputs=[
                input_text,
                max_new_tokens,
                new_doc,
                top_p,
                top_k,
                penalty_alpha,
                num_beams,
                num_return_sequences,
                return_full_text,
                extra_options,
            ],
            outputs=[output_text],
        )
    return demo


@cli(
    "app.llama",
    arguments=[
        Argument(
            "-r",
            "--model-root",
            required=True,
            help="Path to directory containing model checkpoints",
        ),
        Argument(
            "-m",
            "--model-name",
            default="7B",
            choices=["7B", "13B", "30B", "66B"],
            help="Pretrained model or path to local checkpoint",
        ),
        Argument(
            "-p",
            "--server-port",
            default=7860,
            help="Port to run the server on",
        ),
        Argument(
            "-n",
            "--server-name",
            default="localhost",
            help="Server address to run the gradio app at",
        ),
        Argument(
            "-l",
            "--logdir",
            default=None,
            help="Directory to log inputs and outputs to",
        ),
    ],
    requirements=[
        "gradio",
        "torch>=1.13",
        "fairscale",
        "fire",
        "sentencepiece",
    ],
)
def run_llama_demo(
    model_root: str,
    model_name: str = "7B",
    server_port: int = 7860,
    server_name: str = "localhost",
    logdir: Optional[str] = None,
):

    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')


    # gl_model = ModelWrapper(
    #     name=model_name,
    #     precision=precision,
    #     tensor_parallel=parallelize,
    #     logdir=logdir,
    #     leftover_space=leftover_space,
    # )
