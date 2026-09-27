"""Chat interface.  Run with:  streamlit run app.py"""
import streamlit as st

from src import config
from src.generator import generate_answer, llm_available
from src.retriever import Retriever

st.set_page_config(page_title="IT Helpdesk Assistant", page_icon="🛠️")


@st.cache_resource(show_spinner="Loading search index (first run builds it, ~1 min)...")
def get_retriever() -> Retriever:
    if not config.CHUNKS_FILE.exists():  # e.g. first run on Streamlit Cloud
        from src.ingest import build_index, load_chunks
        from src.retriever import load_embedder
        build_index(load_chunks(), load_embedder())
    return Retriever()


with st.sidebar:
    st.header("Settings")
    method = st.selectbox("Retrieval method", ["hybrid", "dense", "bm25"],
                          help="hybrid = keyword + semantic search combined")
    top_k = st.slider("Passages to retrieve", 1, 8, config.TOP_K)
    st.caption(f"LLM: `{config.LLM_MODEL}`" if llm_available()
               else "No LLM key set: showing retrieved passages only.")
    if st.button("Clear chat"):
        st.session_state.messages = []

st.title("🛠️ IT Helpdesk Assistant")
st.caption("Ask about passwords, VPN, Wi-Fi, Teams, printing and more. "
           "Answers come from the Harrowgate IT knowledge base.")

retriever = get_retriever()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("e.g. My account is locked, what do I do?")
if question:
    st.chat_message("user").markdown(question)
    chunks = retriever.search(question, k=top_k, method=method)

    with st.chat_message("assistant"):
        if llm_available():
            with st.spinner("Thinking..."):
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                try:
                    answer = generate_answer(question, chunks, history)
                except Exception as e:  # show API errors instead of crashing the app
                    answer = f"⚠️ The language model could not be reached: {e}"
        else:
            answer = "**Most relevant help articles:**\n\n" + "\n\n".join(
                f"**{c['title']} - {c['section']}**\n\n{c['text'].split(chr(10), 1)[1]}" for c in chunks[:2])
        st.markdown(answer)

        with st.expander("Sources"):
            for i, c in enumerate(chunks, start=1):
                st.markdown(f"**[{i}] {c['title']} - {c['section']}**  \n"
                            f"<small>score {c['score']:.3f} · `{c['id']}`</small>", unsafe_allow_html=True)

    st.session_state.messages += [{"role": "user", "content": question},
                                  {"role": "assistant", "content": answer}]
