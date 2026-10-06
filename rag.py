import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash"]

_embeddings = None
_llms = {}


# ---------- API key ----------
def get_api_key():
    key = os.getenv("GOOGLE_API_KEY", "")
    try:
        import streamlit as st
        if "GOOGLE_API_KEY" in st.secrets:
            key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    return str(key).strip().strip('"').strip("'").strip()


def key_info():
    key = get_api_key()
    if not key:
        return "No key found (length 0)"
    return f"Key starts with: {key[:4]} | length: {len(key)}"


# ---------- models ----------
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


def get_llm(model, fast=True):
    cache_key = (model, fast)
    if cache_key not in _llms:
        if fast:
            _llms[cache_key] = ChatGoogleGenerativeAI(
                model=model, api_key=get_api_key(), thinking_level="low"
            )
        else:
            _llms[cache_key] = ChatGoogleGenerativeAI(
                model=model, api_key=get_api_key()
            )
    return _llms[cache_key]


# ---------- document processing ----------
def build_vectorstore(pdf_path, file_name=None):
    pages = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    name = file_name or os.path.basename(pdf_path)
    for c in chunks:
        c.metadata["file_name"] = name
        c.metadata["total_pages"] = len(pages)
    return FAISS.from_documents(chunks, get_embeddings())


def get_all_chunks(vs):
    return [
        vs.docstore.search(vs.index_to_docstore_id[i])
        for i in range(vs.index.ntotal)
    ]


# ---------- prompts ----------
PROMPT = """You are a helpful assistant that answers questions about a PDF document.
You can only read the TEXT of the PDF. You cannot see colors, images, boxes, diagrams or layout.
If the question is about those visual things, say you can only read the text of the document.

Document info:
{info}

Context from the document:
{context}

Use ONLY the document info and context above. Be clear and concise.
If the answer is not there, say: "I couldn't find that in the document."

Question: {question}

Answer:"""

SUMMARY_PROMPT = """You are a helpful assistant. Below are excerpts taken from across a whole PDF document.
Write a clear, well-organized summary of what the document is about: its title or subject,
the main topics, and the key points. Use short bullet points. Use ONLY the excerpts.
You can only read text, not images or colors.

Document info:
{info}

Excerpts:
{context}

Request: {question}

Summary:"""

BROAD_WORDS = [
    "summary", "summarize", "summarise", "overview", "main points",
    "key points", "topics", "what is this document", "what is this file",
    "about this file", "about this document", "tell me about the",
]


def is_broad(question):
    q = question.lower()
    return any(w in q for w in BROAD_WORDS)


# ---------- asking Gemini ----------
def ask_llm(prompt):
    models = [MODEL_NAME] + [m for m in FALLBACK_MODELS if m != MODEL_NAME]
    last_error = None
    for model in models:
        fast = True
        for attempt in range(3):
            try:
                start = time.time()
                text = get_llm(model, fast).invoke(prompt).text
                print(f"[{model}] answered in {time.time() - start:.1f}s")
                return text
            except Exception as e:
                last_error = e
                msg = str(e)
                if fast and "thinking" in msg.lower():
                    fast = False
                    continue
                if any(w in msg for w in ["503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"]):
                    time.sleep(attempt + 1)
                else:
                    break
    raise last_error


def answer_question(vectorstore, question, k=4):
    all_chunks = get_all_chunks(vectorstore)
    first = all_chunks[0]
    info = (
        f"File name: {first.metadata.get('file_name', 'unknown')}\n"
        f"Total pages: {first.metadata.get('total_pages', 'unknown')}"
    )

    if is_broad(question):
        step = max(1, len(all_chunks) // 10)
        docs = all_chunks[::step][:10]
        template = SUMMARY_PROMPT
    else:
        docs = vectorstore.similarity_search(question, k=k)
        if first not in docs:
            docs = [first] + docs
        template = PROMPT

    context = "\n\n".join(d.page_content for d in docs)
    answer = ask_llm(template.format(info=info, context=context, question=question))
    return answer, docs


if __name__ == "__main__":
    print("Reading the PDF...")
    store = build_vectorstore("sample.pdf")
    question = "What is the title of the document?"
    answer, sources = answer_question(store, question)
    print("\nQuestion:", question)
    print("Answer:", answer)
    print("\nFound in pages:", [d.metadata.get("page", 0) + 1 for d in sources])