from dotenv import load_dotenv
load_dotenv()

import os
from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from helper import vector_db

RELEVANCE_THRESHOLD = 0.45

def get_store() -> Chroma:
    return vector_db()


class State(TypedDict):
    question: str
    docs: List[Document]
    answer: str


def retrieve(state: State) -> dict:
    scored = get_store().similarity_search_with_relevance_scores(state["question"], k=5)
    docs = [doc for doc, score in scored if score >= RELEVANCE_THRESHOLD]
    return {"docs": docs}


def generate(state: State) -> dict:
    # Label each chunk with its document and page so the model can cite it
    context = "\n\n".join(
        f"[{d.metadata.get('source_doc')}, page {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
        for d in state["docs"]
    )
    # TODO (yours): improve this prompt. Make sure it says "I don't know"
    # when the context doesn't contain the answer.
    prompt = (
    "You are answering questions about a financial document. "
    "Use only the context below. If the context does not contain the answer, "
    "reply exactly: 'I don't know. The uploaded documents do not contain this information.' "
    "Otherwise, answer concisely and cite the document and page for each claim.\n\n"
    f"Context:\n{context}\n\nQuestion: {state['question']}"
    )
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0,
    )
    return {"answer": llm.invoke(prompt).content}


def no_answer(state: State) -> dict:
    return {"answer": "No relevant content was found in the uploaded documents."}


def route_after_retrieve(state: State) -> str:
    return "generate" if state["docs"] else "no_answer"


builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("no_answer", no_answer)

builder.add_edge(START, "retrieve")
builder.add_conditional_edges("retrieve", route_after_retrieve)
builder.add_edge("generate", END)
builder.add_edge("no_answer", END)

graph = builder.compile()