import json
import os
import re
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Literal, Optional, Tuple, Union

from shadow_scholar.cli import Argument, cli, safe_import

with safe_import():
    import gradio as gr
    import openai
    import requests
    from jinja2 import Template


SEARCH_URI = "https://api.semanticscholar.org/graph/v1/paper/search/"
S2_PAPER_LINK = "https://api.semanticscholar.org/{sha}"

OPEN_AI_MODEL = "text-davinci-003"

QUERY_EXTRACTION_TEMPLATE = """\
Given the following prompt from a user, write one or more search queries \
to submit to an paper search engine to find relevant papers to answer the \
user information need. Fewer queries is better. Write at most 3. Make query \
rich in relevant keywords.

Prompt: "{{ prompt }}"

Queries:
-\
"""


ANSWER_FORMULATION_TEMPLATE = """\
You are a scientific knowledge assistant. A user has given you the following \
prompt: "{{ prompt }}"

Using a search engine, you have found the following papers:
{% for paper in state.stack %}
[{{ paper.id }}] {{ paper.title }}
{{ paper.abstract }}
{% endfor %}\

Write a response to the user that answers their information need. Make sure \
to only use information from the previous results. When writing a claim, \
cite the relevant paper(s) using the format `[n]` (for example, `[1]` for \
the first paper). The list of results above is not exhaustive, so you need to \
make sure that the user knows that.

Response:\
"""

RESULT_TABLE_TEMPLATE = """\
<div class="paper_row">\
<p class="paper_link">\
[{{ paper.id }}] <a href="{{ paper.url }}" target="_blank">\
<b>{{ paper.title }}</b></a> ({{ paper.year }})</p>\
{% if paper.abstract %} \
<p class="paper_abstract">\
{{ paper.abstract }}\
</p>\
{% endif %}\
</div>\
"""

GRADIO_CSS = """
.paper_row {

    margin-bottom: .8em;
}
.paper_link {
    font-size: 1em;
}
.paper_abstract {
    font-size: .85em;
}
"""


PromptType = List[Tuple[Union[str, None], Union[str, None]]]


class ACT(Enum):
    CHAT = 0
    ADD = 1
    REMOVE = 2


@dataclass
class Paper:
    sha1: str
    id: Optional[int] = None
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

    def __str__(self) -> str:
        return json.dumps(
            {**(self._cache or {}), 'sha1': self.sha1},
            sort_keys=True
        )

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
        return f"https://api.semanticscholar.org/{self.sha1}"

    @property
    def year(self) -> int:
        return int(self.cache["year"])

    @classmethod
    def from_url(cls, url, s2_api_key: Optional[str]) -> Optional["Paper"]:
        is_valid_url = re.match(
            r"https://(www.)?semanticscholar.org/paper.*?/([a-f0-9]+)", url
        )
        # print(url)
        if not is_valid_url:
            return None

        sha1 = is_valid_url.group(2)
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

    def __post_init__(self):
        self.result_table_template = Template(RESULT_TABLE_TEMPLATE)

    def _fetch_stack(self):
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/batch?"
            f"fields={','.join(Paper.api_fields())}"
        )
        header = {"x-api-key": self.s2_api_key}
        locations, missing = zip(
            *(
                (i, paper.sha1)
                for i, paper in enumerate(self.stack)
                if paper.missing
            )
        )
        data = {"ids": missing}
        data = requests.post(url, headers=header, json=data).json()
        for loc, cache in zip(locations, data):
            self.stack[loc].cache = cache  # pyright: ignore

    def add_to_stack(self, elem_or_elems: Union[Paper, List[Paper]]):
        if not isinstance(elem_or_elems, list):
            elem_or_elems = [elem_or_elems]

        existing = set(paper.sha1 for paper in self.stack)

        for elem in elem_or_elems:
            if elem.sha1 not in existing:
                # we want ids to be 1-indexed
                elem.id = len(self.stack) + 1
                self.stack.append(elem)

    def table_stack(self, rows: Optional[Iterable[str]] = None):
        html = []
        for paper in self.stack:
            if rows is not None and paper.sha1 not in rows:
                continue
            if paper.missing:
                self._fetch_stack()
            html.append(self.result_table_template.render(paper=paper))
        return html


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
        query_extraction_template: str = QUERY_EXTRACTION_TEMPLATE,
        answer_template: str = ANSWER_FORMULATION_TEMPLATE,
        query_extraction_max_tokens: int = 128,
        google_search_cx: str = "602714345f3a24773",
        google_max_results: int = 3
    ):
        self.s2_key = s2_key
        self.opeai_key = openai_key
        self.google_search_key = google_search_key
        self.s2_endpoint = s2_endpoint
        self.openai_model = openai_model
        self.query_extraction_template = Template(query_extraction_template)
        self.answer_template = Template(answer_template)
        self.query_extraction_max_tokens = query_extraction_max_tokens
        self.s2_results_limit = s2_results_limit
        self.s2_search_fields = s2_search_fields or ["title", "abstract"]
        self.google_search_cx = google_search_cx
        self.google_max_results = google_max_results

        openai.api_key = openai_key

    def google_search(self, query: str):
        # encode query with %20 for spaces, etc
        query = urllib.parse.quote(query)
        url = (
            f"https://www.googleapis.com/customsearch/v1/siterestrict?"
            f"&key={self.google_search_key}"
            f"&cx={self.google_search_cx}"
            "&gl=us&lr=lang_en"
            f"&q={query}"
        )
        response = requests.get(url).json()

        print(response)

        results = [
            Paper.from_url(url=item["link"], s2_api_key=self.s2_key)
            for item in response.get("items", [])
        ]
        results_dict = {
            r.sha1 : (i, r) for i, r in enumerate(results) if r is not None
        }
        filtered_results = [
            r for _, r in sorted(results_dict.values(), key=lambda x: x[0])
        ]
        return filtered_results

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
        text = re.sub(r"^\-\s*", "", text)
        text = re.sub(r"[\"\']+", "", text)
        text = re.sub(r"(^\s+|\s+$)", "", text)
        return text

    def extract_queries(self, prompt: str) -> List[str]:
        response = openai.Completion.create(
            engine=self.openai_model,
            prompt=self.query_extraction_template.render(prompt=prompt),
            max_tokens=self.query_extraction_max_tokens,
            temperature=0.5,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        text = response["choices"][0]["text"]  # pyright: ignore

        queries = [
            self.strip_and_remove_quotes(q)
            for q in text.split("\n")
            if q.strip()
        ]
        return queries

    def formulate_answer(self, prompt: str, state: State) -> str:
        response = openai.Completion.create(
            engine=self.openai_model,
            prompt=self.answer_template.render(
                prompt=prompt, state=state
            ),
            max_tokens=256,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        text = response["choices"][0]["text"]   # pyright: ignore

        return text

    def __call__(
        self, prompt: str, state: Optional[State] = None
    ) -> Tuple[PromptType, State]:

        if state is None:
            state = State()

        state.history.append((None, prompt))

        queries = self.extract_queries(prompt)
        results_shas = set()

        for query in queries:
            state.history.append((f"Searching for **{query}**...", None))

            # results = self.semantic_scholar_search(query)
            results = self.google_search(query)

            state.add_to_stack(results)
            results_shas.update([paper.sha1 for paper in results])

        table_html = state.table_stack(rows=results_shas)
        state.history.append(('\n'.join(table_html), None))

        answer = self.formulate_answer(prompt, state)
        state.history.append((answer, None))

        return state.history, state


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
        "transformers",
        "openai",
        "jinja2",
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

    with gr.Blocks(css=GRADIO_CSS) as demo:
        chatbot = gr.Chatbot()
        state = gr.State(None)

        with gr.Row():
            txt = gr.Textbox(
                show_label=False, placeholder="Enter text and press enter"
            ).style(container=False)

            txt.submit(app, [txt, state], [chatbot, state])

    demo.launch(
        server_port=server_port, server_name=server_name, enable_queue=True
    )
