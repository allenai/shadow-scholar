import datetime
import json
import multiprocessing
import sys
from pathlib import Path
from time import sleep
from typing import Any, Dict, Literal, Optional, Union

import requests

from shadow_scholar.cli import Argument, cli, safe_import

with safe_import():
    import fire
    import gradio as gr
    import torch

    from shadow_scholar.app.llama.facebook_llama.example import (
        load,
        setup_model_parallel,
    )


NUM_GPUS_MAP = {
    "7B": 1,
    "13B": 2,
    "30B": 4,
    "65B": 8,
}


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
    logdir: Optional[Union[str, Path]] = None,
):
    num_gpus = NUM_GPUS_MAP[model_name]

    try:
        local_rank, world_size = setup_model_parallel()
    except ValueError:
        # something went wrong with model parallel setup
        local_rank, world_size = -1, -1

    if world_size < 0:
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
        if local_rank == 0:
            logdir.parent.mkdir(parents=True, exist_ok=True)

    ui = UI(
        model_name=model_name,
        server_name=server_name,
        ext_port=server_port,
        int_port=5000,
    )

    ps = []
    if local_rank == 0:
        # start UI and communication server
        ps.append(multiprocessing.Process(target=ui.start_server))
        ps.append(multiprocessing.Process(target=ui.start_ui))

    for p in ps:
        p.start()

    print(f"Starting Llama demo, rank: {local_rank}")

    try:
        # load models
        model_root = model_root.rstrip("/")
        generator = load(
            ckpt_dir=f"{model_root}/{model_name}",
            tokenizer_path=f"{model_root}/tokenizer.model",
            local_rank=local_rank,
            world_size=world_size,
        )

        torch.distributed.barrier()  # pyright: ignore

        while True:
            input_data = ui.get("input")
            if not input_data:
                sleep(1)
                continue

            if local_rank == 0:
                print(f"RANK {local_rank}:", json.dumps(input_data, indent=2))

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

            if local_rank == 0:
                print(f"RANK {local_rank}:", json.dumps(output_data, indent=2))

            if local_rank == 0:
                ui.set("output", output_data)
                ui.delete("input")
                sleep(1)

            torch.distributed.barrier()  # pyright: ignore
    finally:
        for p in ps:
            p.terminate()
            p.join()
        if local_rank == 0:
            gr.close_all()


if __name__ == "__main__":
    fire.Fire(run_llama_demo)
