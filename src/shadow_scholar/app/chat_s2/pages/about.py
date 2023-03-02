import streamlit as st


def write():
    """Used to write the page in the app.py file"""
    st.title("Ahoy!")

    st.markdown(
        "This streamlit application provides forecite counters over "
        "a subset of the S2ORC dataset. The dataset is a collection of "
        "open-access scientific papers and their associated metadata. "
    )
