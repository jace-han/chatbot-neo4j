import streamlit as st

# Create the LLM
# Create the LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    openai_api_key=st.secrets["OPENROUTER_API_KEY"],
    model=st.secrets["LLM_MODEL"],
    openai_api_base="https://openrouter.ai/api/v1",
    max_tokens=400
)

# Create the Embedding model
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    openai_api_key=st.secrets["OPENROUTER_API_KEY"]
)