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
PROMPT_TXT = """\
Answer the question based on the context below.

Context: {context}

Question: {question}

Answer:\
"""


def cosine_similarity(A, B):
    dot_product = A @ B.T
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    similarity = dot_product / (norm_A * norm_B)
    return similarity


def search(query, s2_key, limit=20, fields=["title", "abstract"]):
    # space between the  query to be removed and replaced with +
    query = query.replace(" ", "+")
    url = f'{SEARCH_URI}?query={query}&limit={limit}&fields={",".join(fields)}'
    headers = {"Accept": "*/*", "X-API-Key": s2_key}
    response = requests.get(url, headers=headers)
    return response.json()


# function to preprocess the query and remove the stopwords
# before passing it to the search function
def preprocess_query(query):
    query = query.lower()
    # remove stopwords from the query
    stopwords = set(nltk.corpus.stopwords.words("english"))
    query = " ".join([word for word in query.split() if word not in stopwords])
    return query


class SpecterEmbeddings:
    def __init__(self, model_name="allenai/specter"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            .to("cuda" if torch.cuda.is_available() else "cpu")
            .eval()
        )

    def __call__(self, text):
        with torch.inference_mode():
            tokens = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.model.device)
            embeddings = self.model(**tokens).pooler_output
            return embeddings.detach().to("cpu", dtype=torch.float32).numpy()


def create_context(question, df, max_len=3800, size="davinci"):
    """
    Create a context for a question by finding the most
    similar context from the dataframe
    """

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context
    # until the context is too long
    for i, row in df.iterrows():

        # Add the length of the text to the current length
        cur_len += row["n_tokens"] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["title_abs"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
    df,
    model="text-davinci-003",
    question="What is the impact of creatine on cognition?",
    max_len=3800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None,
):
    """
    Answer a question based on the most similar context from the
    dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=PROMPT_TXT.format(context=context, question=question),
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


class App:
    def __init__(self, s2_key: str, openai_key: str):
        self.s2_key = s2_key
        self.openai_key = openai_key
        self.embeddings = SpecterEmbeddings()
        openai.api_key = openai_key
        nltk.download("stopwords")

    def __call__(self, query: str) -> str:
        # json to pandas dataframe
        search_results = search(preprocess_query(query), s2_key=self.s2_key)

        if search_results["total"] == 0:
            return "No results found - Try another query"
        else:
            df = pd.DataFrame(search_results["data"]).dropna()

        # merge columns title and abstract into a string separated by
        # tokenizer.sep_token and store it in a list

        df["title_abs"] = [
            d["title"]
            + self.embeddings.tokenizer.sep_token
            + (d.get("abstract") or "")
            for d in df.to_dict("records")
        ]
        df["n_tokens"] = df.title_abs.apply(
            lambda x: len(self.embeddings.tokenizer.encode(x))
        )

        # get embeddings for each document and query
        doc_embeddings = self.embeddings(list(df["title_abs"]))
        query_embeddings = self.embeddings(query)

        df["specter_embeddings"] = list(doc_embeddings)
        # find the cosine similarity between the query and the documents
        df["similarity"] = cosine_similarity(
            query_embeddings, doc_embeddings
        ).flatten()

        # sort the dataframe by similarity
        df.sort_values(by="similarity", ascending=False, inplace=True)

        return answer_question(df, question=query, debug=False)


@cli(
    "app.s2qa",
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
            required=True,
            help="OpenAI API key",
        ),
        Argument(
            "-sk",
            "--s2-key",
            required=True,
            help="Semantic Scholar API key",
        ),
    ],
    requirements=[
        "requests",
        "nltk",
        "transformers",
        "openai",
        "torch",
        "pandas",
        "accelerate",
    ],
)
def run_qa_demo(
    server_port: int,
    server_name: str,
    openai_key: str,
    s2_key: str,
):
    app = App(s2_key=s2_key, openai_key=openai_key)

    demo = gr.Interface(
        title="S2QA Demo",
        fn=app,
        inputs=[
            gr.inputs.Textbox(
                label="Question",
                lines=3,
                placeholder="What is the impact of creatine on cognition?",
            )
        ],
        outputs=[
            gr.Textbox(
                label="Answer",
                lines=3,
            )
        ],
    )

    demo.launch(
        server_port=server_port, server_name=server_name, enable_queue=True
    )
