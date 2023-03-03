import os
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import List, Literal, Optional, Tuple, Union

from shadow_scholar.cli import Argument, cli, safe_import

with safe_import():
    import gradio as gr
    import nltk
    import numpy as np
    import openai
    import pandas as pd
    import requests
    import torch
    from transformers import AutoModel, AutoTokenizer


SEARCH_URI = "https://api.semanticscholar.org/graph/v1/paper/search/"
OPEN_AI_MODEL = "text-davinci-003"

QUERY_EXTRACTION_PROMPT = """\
Given the following prompt from a user, write one or more search queries \
to submit to an paper search engine to find relevant papers to answer the \
user information need. Fewer queries is better. Write at most 3. Make query \
rich in relevant keywords.

Prompt: "{prompt}"

Queries:
-\
"""
S2_PAPER_LINK = "https://api.semanticscholar.org/{sha}"

PromptType = List[Tuple[Union[str, None], Union[str, None]]]


class ACT(Enum):
    CHAT = 0
    ADD = 1
    REMOVE = 2


@dataclass
class Paper:
    sha1: str
    _cache: Optional[dict] = None
    _s2_api_key: str = os.environ.get("S2_API_KEY", "")
    # corpus_id: Optional[int] = None
    # title: Optional[str] = None
    # abstract: Optional[str] = None
    # paragraphs: Optional[List[str]] = None

    @classmethod
    def api_fields(cls) -> List[str]:
        return [
            "title",
            "abstract",
            "venue",
            "fieldsOfStudy",
            "authors",
            "isOpenAccess",
            "year",
        ]

    @property
    def cache(self) -> dict:
        if self._cache is None:
            url = (
                "https://api.semanticscholar.org/graph/v1/paper/"
                f"{self.sha1}?fields={','.join(self.api_fields())}"
                f"title,venue,abstract,fieldsOfStudy,authors,isOpenAccess,year"
            )
            header = {"x-api-key": self._s2_api_key}
            self._cache = requests.get(url, headers=header).json()
        return self._cache

    @cache.setter
    def cache(self, value: dict):
        if self._cache is not None:
            raise ValueError("Cache already set")
        self._cache = value

    @property
    def missing(self) -> bool:
        return self._cache is None

    @property
    def url(self) -> str:
        return "https://api.semanticscholar.org/{self.sha1}"

    @property
    def year(self) -> int:
        return int(self.cache["year"])

    @classmethod
    def from_url(cls, url, s2_api_key: Optional[str]):
        is_valid_url = re.match(
            r"https://www.semanticscholar.org/paper/.*?/([a-f0-9]+)", url
        )
        if not is_valid_url:
            raise ValueError("Invalid PDP URL")

        sha1 = is_valid_url.group(1)
        if s2_api_key:
            return cls(sha1=sha1, _s2_api_key=s2_api_key)
        else:
            return cls(sha1=sha1)

    @property
    def title(self) -> str:
        return self.cache["title"]

    @property
    def abstract(self) -> str:
        return self.cache["abstract"]

    @property
    def is_open_access(self) -> bool:
        return self.cache["isOpenAccess"]

    @property
    def venue(self) -> str:
        return self.cache["venue"]


@dataclass
class State:
    history: PromptType = field(default_factory=list)
    action: Literal[ACT.CHAT, ACT.ADD, ACT.REMOVE] = ACT.CHAT
    stack: List[Paper] = field(default_factory=list)
    s2_api_key: str = os.environ.get("S2_API_KEY", "")

    def _fetch_stack(self):
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/batch?"
            f"fields={','.join(Paper.api_fields())}"
        )
        header = {"x-api-key": self.s2_api_key}
        missing, locations = zip(
            [
                (i, paper.sha1)
                for i, paper in enumerate(self.stack)
                if paper.missing
            ]
        )
        data = {"ids": missing}
        data = requests.post(url, headers=header, json=data).json()
        for loc, cache in zip(locations, data):
            self.stack[loc].cache = cache  # pyright: ignore

    def table_stack(self, rows: Optional[List[str]] = None):
        html = []
        for paper in self.stack:
            if rows is not None and paper.sha1 not in rows:
                continue
            if paper.missing:
                self._fetch_stack()

            html.append(
                f'<div class="paper_row">'
                f'<p><a href="{paper.url}"><b>{paper.title}<b></a></p>'
                f"<p>{paper.abstract[:100]}...</p>"
                f"<p>{paper.venue}</p>"
                f"<p>{paper.year}</p>"
                "</div>"
            )


class ChatS2:
    def __init__(
        self,
        s2_key: str,
        openai_key: str,
        google_search_key: str,
        s2_endpoint: str = SEARCH_URI,
        s2_results_limit: int = 5,
        s2_search_fields: Optional[List[str]] = None,
        openai_model: str = OPEN_AI_MODEL,
        query_extraction_prompt: str = QUERY_EXTRACTION_PROMPT,
        query_extraction_max_tokens: int = 128,
        google_search_cx: str = "602714345f3a24773",
    ):
        self.s2_key = s2_key
        self.opeai_key = openai_key
        self.google_search_key = google_search_key
        self.s2_endpoint = s2_endpoint
        self.openai_model = openai_model
        self.query_extraction_prompt = query_extraction_prompt
        self.query_extraction_max_tokens = query_extraction_max_tokens
        self.s2_results_limit = s2_results_limit
        self.s2_search_fields = s2_search_fields or ["title", "abstract"]
        self.google_search_cx = google_search_cx

        openai.api_key = openai_key

    def google_search(self, query: str):
        # encode query with %20 for spaces, etc
        query = urllib.parse.quote(query)
        url = (
            f"https://www.googleapis.com/customsearch/v1"
            f"?key={self.google_search_key}"
            f"&cx={self.google_search_cx}"
            f"&q={query}"
        )
        response = requests.get(url).json()

        results = [
            Paper.from_url(url=item["link"], s2_api_key=self.s2_key)
            for item in response.get("items", [])
        ]
        return results

    def semantic_scholar_search(
        self, query: str, fields: List[str] = ["title"]
    ):
        query = query.replace(" ", "+")
        url = (
            f"{self.s2_endpoint}"
            f"?query={query}"
            f"&limit={self.s2_results_limit}"
            f'&fields={",".join(self.s2_search_fields)}'
        )
        headers = {"x-api-key": self.s2_key}
        response = requests.get(url, headers=headers).json()
        return response.get("data", [])

    def strip_and_remove_quotes(self, text: str) -> str:
        text = re.sub(r"^[\s\"\']+", "", text)
        text = re.sub(r"[\s\"\']+$", "", text)
        return text

    def extract_queries(self, prompt: str) -> List[str]:
        response = openai.Completion.create(
            engine=self.openai_model,
            prompt=self.query_extraction_prompt.format(prompt=prompt),
            max_tokens=self.query_extraction_max_tokens,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        text = response["choices"][0]["text"]  # pyright: ignore

        queries = [
            self.strip_and_remove_quotes(q)
            for q in text.split("-")
            if q.strip()
        ]
        return queries

    def __call__(
        self, prompt: str, history: State
    ) -> Tuple[PromptType, PromptType]:
        queries = self.extract_queries(prompt)

        output: PromptType = [(prompt, None)]
        for query in queries:
            output.append((None, f"Searching for **{query}**..."))

            # results = self.semantic_scholar_search(query)
            results = self.google_search(query)

            print(query, len(results))

            results_fmt = ["<ul>"]

            for result in results:
                title = result.get("title", None)
                sha = result.get("paperId", None)
                url = S2_PAPER_LINK.format(sha=sha)

                print(title, sha)

                if not (title and sha):
                    continue
                results_fmt.append(f'<li><a href="{url}">{title}</a></li>')

            results_fmt.append("</ul>")

            output.append((None, "\n".join(results_fmt)))
            time.sleep(0.2)

        # print('----')
        history.extend(output)

        return history, history


@cli(
    "app.chatS2",
    arguments=[
        Argument(
            "-sp",
            "--server-port",
            default=7860,
            help="Port to run the server on",
        ),
        Argument(
            "-sn",
            "--server-name",
            default="localhost",
            help="Server address to run the gradio app at",
        ),
        Argument(
            "-ok",
            "--openai-key",
            default=os.environ.get("OPENAI_API_KEY"),
            help="OpenAI API key",
        ),
        Argument(
            "-sk",
            "--s2-key",
            default=os.environ.get("S2_KEY"),
            help="Semantic Scholar API key",
        ),
        Argument(
            "-gk",
            "--google-custom-search-key",
            default=os.environ.get("GOOGLE_CUSTOM_SEARCH_API_KEY"),
        ),
    ],
    requirements=[
        "requests",
        # 'nltk',
        "transformers",
        "openai",
        # 'torch',
        # 'pandas',
        # 'accelerate',
    ],
)
def run_v2_demo(
    server_port: int,
    server_name: str,
    openai_key: str,
    s2_key: str,
    google_custom_search_key: str,
):
    assert openai_key is not None, "OpenAI API key is required"
    assert s2_key is not None, "Semantic Scholar API key is required"

    app = ChatS2(
        s2_key=s2_key,
        openai_key=openai_key,
        google_search_key=google_custom_search_key,
    )

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        state = gr.State(State())

        with gr.Row():
            txt = gr.Textbox(
                show_label=False, placeholder="Enter text and press enter"
            ).style(container=False)

            txt.submit(app, [txt, state], [chatbot, state])

    demo.launch(
        server_port=server_port, server_name=server_name, enable_queue=True
    )
