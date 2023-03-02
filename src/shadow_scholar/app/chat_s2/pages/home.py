from typing import List

import pandas as pd
import streamlit as st

from ..db import all_forecite_versions, make_connector, read_and_search

FORECITE_TABLE_DESC = """\
Suffix|Description
:---:|---
`v0`|Initial version of ForeCite algorithm. Requires a noun phrase to occur \
both in the candidate paper (*i.e.*, the paper that supposedly introduces the \
concept) and the citing paper.
`v1-1`|Version of ForeCite algorithm that relaxes the requirement that the \
noun phrase occur in both the candidate paper and the citing paper. Instead, \
it requires that the noun phrase occur within ±298 characters from any \
citation in the citing paper. The value 298 was chosen because its the \
average size of a context in SciCite ± 2 standard deviations.
`v1-2`|Similar to `v1-1`, but requires the noun phrase in the citing to \
occur within a paragraph that contains at least one citation in the citing \
paper.
`v1-3`|Similar to `v1-2`, but instead of restricting where citation occurs, \
it restricts the list of all possible candidate terms to those that occur at \
least one in a title (or abstract) of any candidate or citing paper. If only \
titles are included, *this is the original version of ForeCite.*
`v2-1`|Combines `v0` and `v1-1`.
`v2-2`|Combines `v0` and `v1-2`.
`v2-3`|Combines `v0` and `v1-3`.
"""


@st.cache_resource(ttl=60 * 5, show_spinner="Connecting to database...")
def connect():
    conn = make_connector()
    return conn


@st.cache_data(ttl=60 * 5, show_spinner="Listing ForeCite versions...")
def all_dbs() -> List[str]:
    conn = connect()
    return all_forecite_versions(conn=conn)


def write():
    st.title("ForeCite Search")

    """Used to write the page in the app.py file"""
    st.write("Query the ForeCite database to find extracted terms.")
    st.write(
        "### Description of ForeCite versions \n"
        + "The table below describes the different versions of ForeCite. \n"
        + FORECITE_TABLE_DESC
    )
    st.write("<br>", unsafe_allow_html=True)

    conn = connect()

    version = st.selectbox(label="ForeCite version", options=all_dbs())

    term = st.text_input("Term lookup", placeholder="Enter a term")
    exact_match = st.checkbox("Exact match?", value=True)
    limit = st.number_input("Number results", min_value=1, value=10)
    score = st.slider("ForeCite score range", 0.0, 30.0, (0.0, 30.0), step=0.1)
    min_count = st.number_input("Min frequency of a term", min_value=0)
    max_count = st.number_input(
        "Max frequency of a term; if 0, no filtering", min_value=0
    )

    if st.button("Submit"):
        assert version is not None
        assert limit is not None

        try:
            with st.spinner("Querying..."):
                results = read_and_search(
                    forecite_version=version,
                    conn=conn,
                    text=str(term),
                    min_score=float(score[0]),
                    max_score=float(score[1]),
                    min_count=int(min_count),
                    max_count=int(max_count),
                    size=int(limit),
                    exact_match=exact_match,
                )
                for r in results:
                    cid = r["cited"]
                    url = f"https://api.semanticscholar.org/CorpusID:{cid}"
                    r["cited"] = f'<a href="{url}" rel="">{cid}</a>'
                    r.pop("version", None)
                    r.pop("search", None)
                    r["score"] = round(r["score"], 3)

                results_df = pd.DataFrame.from_records(results)
                results_html = results_df.to_html(escape=False)

            st.write(results_html, unsafe_allow_html=True)
            st.write("<br><br><br>", unsafe_allow_html=True)

        except ValueError as err:
            st.error(err)
