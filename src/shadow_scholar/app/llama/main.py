import datetime
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from time import sleep
from typing import Any, Dict, Literal, NamedTuple, Optional, Union

from shadow_scholar.cli import Argument, cli, safe_import

with safe_import():
    import fire
    import gradio as gr
    import requests
    import torch
    from fairscale.nn.model_parallel.initialize import (  # pyright: ignore
        initialize_model_parallel
    )
    from llama import (                                   # pyright: ignore
        LLaMA, ModelArgs, Tokenizer, Transformer
    )

NUM_GPUS_MAP = {
    "7B": 1,
    "13B": 2,
    "30B": 4,
    "65B": 8,
}


class MpEnv(NamedTuple):
    local_rank: int
    world_size: int

    @classmethod
    def auto(cls):
        return cls(
            local_rank=int(os.environ.get("LOCAL_RANK", -1)),
            world_size=int(os.environ.get("WORLD_SIZE", -1)),
        )


def configure_model_parallel() -> MpEnv:
    env = MpEnv.auto()

    torch.distributed.init_process_group("nccl")  # pyright: ignore
    initialize_model_parallel(env.world_size)
    torch.cuda.set_device(env.local_rank)

    # seed all processes with the same seed
    torch.manual_seed(42)
    return env


def load_model_and_tokenizer(
    llama_dir: Union[str, Path],
    model_name: str,
    env: MpEnv,
    seq_len: int = 1024,
    batch_size: int = 32,
) -> "LLaMA":
    start_load_op = time.time()

    llama_dir = Path(llama_dir)

    if not llama_dir.exists():
        raise ValueError(f"LLaMA directory does not exist: {llama_dir}")

    if model_name not in NUM_GPUS_MAP:
        raise ValueError(f"Invalid model name: {model_name}")

    model_path = llama_dir / model_name

    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    checkpoints = sorted(model_path.glob("*.pth"))
    if env.world_size != len(checkpoints):
        raise ValueError(
            f"Found {len(checkpoints)} shards, but world is {env.world_size}"
        )

    ckpt_path = checkpoints[env.local_rank]

    print(f"Loading {ckpt_path} to GPU {env.local_rank}...")
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    with open(Path(model_path) / "params.json", "r") as f:
        params = json.loads(f.read())

    # create model configuration
    params.update({"max_seq_len": seq_len, "max_batch_size": batch_size})
    model_args = ModelArgs(**params)

    # spin-up tokenizer
    tokenizer = Tokenizer(model_path=str(llama_dir / "tokenizer.model"))

    # create model and set parameters up
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)  # pyright: ignore
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    # wrap up model in a LLaMA object
    generator = LLaMA(model, tokenizer)

    # complete message
    delta = time.time() - start_load_op
    print(f"Loaded in {delta:.2f} s on GPU {env.local_rank}.")
    return generator


class UI:
    def __init__(
        self,
        model_name: str,
        server_name: str,
        ext_port: int,
        int_port: int,
    ):
        self.model_name = model_name
        self.server_name = server_name
        self.ext_port = ext_port
        self.int_port = int_port

    @property
    def rank(self) -> str:
        if torch.distributed.is_initialized():  # pyright: ignore
            return str(torch.distributed.get_rank())  # pyright: ignore
        else:
            return "null"

    def get(self, what: Literal["input", "output"]) -> dict:
        resp = requests.get(
            f"http://localhost:{self.int_port}/get/{what}?rank={self.rank}"
        )
        return resp.json()

    def set(self, what: Literal["input", "output"], value: dict):
        requests.post(
            f"http://localhost:{self.int_port}/set/{what}?rank={self.rank}",
            json=value,
        )

    def delete(self, what: Literal["input", "output"]):
        requests.get(
            f"http://localhost:{self.int_port}/delete/{what}?rank={self.rank}"
        )

    def runner(
        self, text: str, max_length: int, temperature: float, top_p: float
    ) -> dict:
        self.set(
            "input",
            {
                "text": text,
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
            },
        )
        output = {}
        while True:
            output = self.get("output")
            if output:
                break
            sleep(1)

        self.delete("output")
        return output["text"]

    def start_ui(self):
        demo = gr.Interface(
            fn=self.runner,
            inputs=[
                gr.Text(lines=10, label="Input"),
                gr.Slider(
                    minimum=1,
                    maximum=2048,
                    step=1,
                    value=256,
                    label="Max Length",
                ),
                gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.8,
                    label="Temperature",
                ),
                gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.95,
                    label="Top P",
                ),
            ],
            outputs=gr.Text(lines=10, label="Output"),
            title=f"LLaMA {self.model_name} Demo",
            # allow_flagging=False,
            # logdir=lg
        )
        demo.queue(concurrency_count=1)
        demo.launch(server_name=self.server_name, server_port=self.ext_port)

    def start_server(self):
        from flask import Flask, jsonify, request

        app = Flask(__name__)

        g: Dict[str, Any] = {"input": None, "output": None}

        def content(what: str) -> dict:
            return {**(g.get(what, None) or {})}

        @app.route("/get/<what>")
        def _get(what: Literal["input", "output"]):
            return jsonify(**content(what))

        @app.route("/set/<what>", methods=["POST"])
        def _set(what: Literal["input", "output"]):
            if request.json is not None:
                g[what] = dict(request.json)
            return jsonify(**content(what))

        @app.route("/delete/<what>")
        def _delete(what: Literal["input", "output"]):
            g[what] = {}
            return jsonify(**content(what))

        app.run()


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
            choices=list(NUM_GPUS_MAP.keys()),
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
            default="0.0.0.0",
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
        "requests",
        "flask",
    ],
)
def run_llama_demo(
    model_root: str,
    model_name: str = "7B",
    server_port: int = 7860,
    server_name: str = "localhost",
    logdir: Optional[Union[str, Path]] = None,
):
    num_gpus = NUM_GPUS_MAP[model_name]

    try:
        import llama  # noqa: F401    # pyright: ignore
    except ImportError:
        msg = (
            "LLaMA is not installed; you have to do so manually. "
            "Please run `pip install git+https://github.com/facebookresearch/"
            "llama.git@76066b1` and try again."
        )
        print(msg, file=sys.stderr)
        sys.exit(1)

    try:
        mp = configure_model_parallel()
    except ValueError:
        # something went wrong with model parallel setup
        mp = MpEnv(-1, -1)

    if mp.world_size < 0:
        message = (
            "This application is meant to be launched with "
            "`torch.distributed.launch`, but it appears that "
            "this is not the case. Please launch the application "
            "with the following command:\n"
            f"torchrun --nproc_per_node {num_gpus} {__file__} "
            f"--model-root {model_root} "
            f"--model-name {model_name} "
            f"--server-port {server_port} "
            f"--server-name {server_name} "
            f"--logdir {logdir}"
        )
        print(message, file=sys.stderr)
        sys.exit(1)

    if logdir:
        current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logdir = Path(f"{logdir}/{model_name}_{current_date}.jsonl")
        if mp.local_rank == 0:
            logdir.parent.mkdir(parents=True, exist_ok=True)

    ui = UI(
        model_name=model_name,
        server_name=server_name,
        ext_port=server_port,
        int_port=5000,
    )

    ps = []
    if mp.local_rank == 0:
        # start UI and communication server
        ps.append(multiprocessing.Process(target=ui.start_server))
        ps.append(multiprocessing.Process(target=ui.start_ui))

    for p in ps:
        p.start()

    print(f"Starting Llama demo, rank: {mp.local_rank}")

    try:
        # load models
        model_root = model_root.rstrip("/")
        generator = load_model_and_tokenizer(
            llama_dir=model_root,
            model_name=model_name,
            env=mp,
            seq_len=1024,
            batch_size=32,
        )

        torch.distributed.barrier()  # pyright: ignore

        while True:
            input_data = ui.get("input")
            if not input_data:
                sleep(1)
                continue

            if mp.local_rank == 0:
                content = json.dumps(input_data, indent=2)
                print(f"RANK {mp.local_rank}: {content}")

            text = input_data["text"].strip()

            if len(text) > 0:
                results = generator.generate(
                    [text],
                    max_gen_len=int(input_data["max_length"]),
                    temperature=float(input_data["temperature"]),
                    top_p=float(input_data["top_p"]),
                )
            else:
                results = [""]

            output_data = {"text": results[0]}

            if logdir is not None:
                with open(logdir, "a") as f:
                    data = json.dumps(
                        {"input": input_data, "output": output_data}, indent=2
                    )
                    f.write(data + "\n")

            if mp.local_rank == 0:
                data = json.dumps(output_data, indent=2)
                print(f"RANK {mp.local_rank}: {data}")

            if mp.local_rank == 0:
                ui.set("output", output_data)
                ui.delete("input")
                sleep(1)

            torch.distributed.barrier()  # pyright: ignore
    finally:
        for p in ps:
            p.terminate()
            p.join()
        if mp.local_rank == 0:
            gr.close_all()


if __name__ == "__main__":
    fire.Fire(run_llama_demo)
