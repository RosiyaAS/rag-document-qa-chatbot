import html
import os
import tempfile
import streamlit as st
from rag import build_vectorstore, answer_question

st.set_page_config(page_title="DocChat AI", page_icon="📄", layout="wide")

CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 900px;}

.hero {
    padding: 1.6rem 1.8rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #06b6d4 100%);
    margin-bottom: 1.4rem;
}
.hero h1 {margin: 0; font-size: 2rem; color: white !important;}
.hero p {margin: 0.4rem 0 0 0; color: white !important; opacity: 0.92;}

.step-card {
    padding: 1rem 1.1rem;
    border-radius: 12px;
    background: rgba(124, 58, 237, 0.08);
    border: 1px solid rgba(124, 58, 237, 0.25);
    height: 100%;
}
.step-card .num {font-size: 1.5rem; font-weight: 700; color: #7c3aed;}
.step-card .title {font-weight: 600; margin-bottom: 0.2rem;}

.source-card {
    border-left: 4px solid #7c3aed;
    background: rgba(124, 58, 237, 0.08);
    padding: 0.6rem 0.9rem;
    border-radius: 8px;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
}
.source-page {font-weight: 600; color: #7c3aed; font-size: 0.8rem; margin-bottom: 0.2rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

SUGGESTIONS = [
    "Summarize this document",
    "What are the key points?",
    "What topics does it cover?",
]

# ---------- app memory ----------
defaults = {
    "store": None,
    "file_name": None,
    "messages": [],
    "pending": None,
    "pages": 0,
    "chunks": 0,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def set_pending(text):
    st.session_state.pending = text


def clear_chat():
    st.session_state.messages = []


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("## 📄 DocChat AI")
    st.caption("Chat with your PDF using RAG + Gemini")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded is not None and uploaded.name != st.session_state.file_name:
        with st.spinner("Reading and indexing your document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            store = build_vectorstore(tmp_path, uploaded.name)
            os.remove(tmp_path)
        st.session_state.store = store
        st.session_state.file_name = uploaded.name
        st.session_state.messages = []
        st.session_state.chunks = store.index.ntotal
        docs = store.docstore._dict.values()
        st.session_state.pages = len({d.metadata.get("page", 0) for d in docs})

    if st.session_state.store is not None:
        st.success(f"✅ {st.session_state.file_name}")
        c1, c2 = st.columns(2)
        c1.metric("Pages", st.session_state.pages)
        c2.metric("Chunks", st.session_state.chunks)
        st.button("🗑️ Clear chat", on_click=clear_chat)

    with st.expander("ℹ️ How it works"):
        st.markdown(
            "1. **Split** the PDF into small chunks\n"
            "2. **Embed** each chunk as numbers that capture meaning\n"
            "3. **Search** for the chunks closest to your question\n"
            "4. **Ask Gemini** to answer using only those chunks"
        )
    st.caption("Built by Rosiya A.S")

# ---------- header ----------
st.markdown(
    """
    <div class="hero">
        <h1>📄 DocChat AI</h1>
        <p>Upload a PDF and ask questions. Every answer comes only from your document.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- handle a new question ----------
typed = st.chat_input("Ask a question about your document...")
question = typed or st.session_state.pending
st.session_state.pending = None

if question and st.session_state.store is not None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Thinking..."):
        try:
            answer, docs = answer_question(st.session_state.store, question)
        except Exception as e:
            answer, docs = f"⚠️ Something went wrong: {e}", []
    sources = [
        {"page": d.metadata.get("page", 0) + 1, "text": d.page_content}
        for d in docs
    ]
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
elif question:
    st.warning("Please upload a PDF in the sidebar first.")

# ---------- main screen ----------
if st.session_state.store is None:
    st.markdown("### 👋 Get started in 3 steps")
    cols = st.columns(3)
    steps = [
        ("1", "Upload", "Add a PDF using the sidebar on the left."),
        ("2", "Ask", "Type any question about the document."),
        ("3", "Verify", "Open Sources to see the exact text used."),
    ]
    for col, (num, title, text) in zip(cols, steps):
        col.markdown(
            f'<div class="step-card"><div class="num">{num}</div>'
            f'<div class="title">{title}</div>{text}</div>',
            unsafe_allow_html=True,
        )
elif not st.session_state.messages:
    st.markdown("### 💡 Try asking")
    cols = st.columns(len(SUGGESTIONS))
    for col, q in zip(cols, SUGGESTIONS):
        col.button(q, on_click=set_pending, args=(q,), key=f"sugg_{q}")

for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    text = html.escape(s["text"][:400]).replace("\n", "<br>")
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="source-page">PAGE {s["page"]}</div>{text}...</div>',
                        unsafe_allow_html=True,
                    )


from rag import key_info

with st.expander("🔧 Debug (temporary)"):
    st.write(key_info())