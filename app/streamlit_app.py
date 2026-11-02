"""RegLens web demo.

Run (venv active, after `python scripts/ingest.py --index`):
    streamlit run app/streamlit_app.py

Streamlit re-runs this whole script on every click, top to bottom.
@st.cache_resource keeps slow things (the embedding model) loaded between runs.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reglens.config import get_settings  # noqa: E402
from reglens.generate import answer_question  # noqa: E402
from reglens.retrieval import retrieve  # noqa: E402

EXAMPLES = [
    "Berapa lama PUJK wajib menyelesaikan pengaduan konsumen?",
    "How often must a financial institution provide anti-fraud education?",
    "Kapan laporan rencana kegiatan literasi keuangan disampaikan ke OJK?",
    "Berapa harga bitcoin hari ini?",  # out of scope: should refuse
]

st.set_page_config(page_title="RegLens", page_icon="⚖️", layout="centered")


@st.cache_resource(show_spinner="Loading the search model (first time only)...")
def warm_up() -> bool:
    """Load the embedding model and the index once, so the first question isn't slow."""
    retrieve("pengaduan konsumen", top_k=1)
    return True


@st.cache_data(show_spinner=False, max_entries=100)
def ask(question: str):
    """Search + answer. Cached, so asking the same question twice is instant (and free)."""
    chunks = retrieve(question, top_k=get_settings().top_k)
    return chunks, answer_question(question, chunks)


# --- Header ---
st.title("⚖️ RegLens")
st.caption(
    "Ask about Indonesian financial regulations (OJK: POJK & SEOJK) in Bahasa Indonesia or English. "
    "Every answer cites the article it comes from, or says it doesn't know."
)

settings = get_settings()
if not settings.llm_ready:
    st.error("GEMINI_API_KEY is missing. Add it to your .env file and restart.")
    st.stop()
warm_up()

# --- Question input ---
st.write("Try an example:")
columns = st.columns(2)
clicked = None
for i, example in enumerate(EXAMPLES):
    if columns[i % 2].button(example, key=f"example-{i}", width="stretch"):
        clicked = example
if clicked:
    st.session_state.question = clicked  # fills the text box below

with st.form("ask"):
    question = st.text_area("Your question", key="question", height=90)
    submitted = st.form_submit_button("Ask", type="primary")

# Remember the last asked question, so the answer stays visible on later reruns
if (submitted or clicked) and question.strip():
    st.session_state.asked = question.strip()

# --- Answer ---
asked = st.session_state.get("asked")
if asked:
    with st.spinner("Searching the regulations and writing an answer..."):
        try:
            chunks, answer = ask(asked)
        except Exception as err:  # e.g. Gemini still overloaded after retries
            st.error(f"Something went wrong: {err}")
            st.stop()

    st.subheader("Answer")
    if answer.answered:
        st.success(answer.answer)
    else:
        st.warning(answer.answer)

    cited = {c.chunk_id for c in answer.citations}
    if cited:
        st.markdown("**Cited:** " + " · ".join(
            dict.fromkeys(f"{c.doc_title} — {c.pasal or 'Pembukaan'}" for c in answer.citations)
        ))

    st.subheader("Sources searched")
    st.caption("The numbers match the [n] references in the answer.")
    for number, chunk in enumerate(chunks, start=1):
        mark = "✅ " if chunk.chunk_id in cited else ""
        title = f"{mark}[{number}] {chunk.doc_title} — {chunk.pasal or 'Pembukaan'} (page {chunk.page_start})"
        with st.expander(title, expanded=chunk.chunk_id in cited):
            st.text(chunk.text)

# --- Footer ---
st.divider()
st.caption(
    f"Search: multilingual-e5-small + ChromaDB · Answers: {settings.gemini_chat_model} · "
    "Not legal advice: always check the official regulation text."
)
