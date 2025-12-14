import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import streamlit as st

from rag.retriever import retrieve_top_k
from rag.citations import build_context_with_citations, citations_to_ui_lines

from app.prompts import SYSTEM_BASE, explain_prompt, extract_prompt, qa_prompt
from app.generate import generate_text
from app.context import ChatTurn, trim_history, history_to_messages
from app.guards import (
    validate_report_input,
    validate_question_input,
    validate_retrieval_results,
    enforce_disclaimer,
)

st.set_page_config(page_title="HeyDoc AI - Radiology Report Copilot", layout="wide")

SAMPLE_REPORT = """CHEST X-RAY (PA AND LATERAL)
CLINICAL HISTORY: Shortness of breath.

FINDINGS: Mild patchy opacity in the right lower lung. No pleural effusion. No pneumothorax.
Cardiomediastinal silhouette is within normal limits.

IMPRESSION: Mild right lower lobe opacity may represent atelectasis versus early infection. Correlate clinically.
"""


def show_citations(citation_lines, citations):
    with st.expander("Show citations / sources"):
        for line in citation_lines:
            st.write(line)
        st.markdown("---")
        for c in citations:
            st.markdown(f"**[{c.cid}] {c.source} - page {c.page}** (score: {c.score:.3f})")
            st.caption(c.snippet)


def ensure_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "report_text" not in st.session_state:
        st.session_state.report_text = ""
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def main():
    ensure_session_state()

    # Header 
    c1, c2 = st.columns([0.07, 0.93], vertical_alignment="center")

    with c1:
        st.image(str(PROJECT_ROOT / "docs" / "logo.png"), width=60)

    with c2:
        st.markdown(
            """
            <div style="display:flex; align-items:center;">
                <span style="font-size:40px; font-weight:900; color:#2E86C1;">
                    HeyDocAI
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<i><b>Confused by radiology jargon? HeyDocAI is here to explain!</b></i>",
        unsafe_allow_html=True,
    )
    st.caption("Only explanations with citations. Not medical advice.")

    col_left, col_right = st.columns([1, 2], gap="large")

    # LEFT: input panel
    with col_left:
        st.subheader("Input Radiology Report")

        st.session_state.report_text = st.text_area(
            "Paste a radiology report here:",
            value=st.session_state.report_text,
            height=260,
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Load Sample Report"):
                st.session_state.report_text = SAMPLE_REPORT
                st.rerun()
        with c2:
            if st.button("Reset"):
                st.session_state.report_text = ""
                st.session_state.chat_history = []
                st.rerun()

        ok, err = validate_report_input(st.session_state.report_text)
        if not ok:
            st.warning(err)

        st.markdown("---")

        with st.expander("Advanced: Retrieval Settings (Pinecone)", expanded=False):
            st.caption(
                "These settings control how many reference chunks are retrieved and how strict the filtering is. "
                "Higher min score = stricter evidence quality."
            )
            top_k = st.slider("Top-K retrieved chunks", 6, 20, 12, 1)
            min_score = st.slider("Min similarity score", 0.30, 0.80, 0.50, 0.01)
            final_top_k = st.slider("Final chunks used in prompt", 2, 10, 6, 1)

    # RIGHT: tabs
    with col_right:
        tabs = st.tabs(["Explain", "Extract", "Evidence Q&A"])

        # TAB 1: Explain 
        with tabs[0]:
            st.subheader("Explain the report")
            level = st.selectbox("Explanation level", ["simple", "normal", "clinician"], index=1)

            if st.button("Generate Explanation"):
                ok, err = validate_report_input(st.session_state.report_text)
                if not ok:
                    st.error(err)
                else:
                    with st.spinner("Retrieving evidence and generating explanation..."):
                        t0 = time.time()
                        retrieved = retrieve_top_k(
                            query=f"Explain terms and phrases in this report: {st.session_state.report_text[:300]}",
                            top_k=top_k,
                            min_score=min_score,
                            final_top_k=final_top_k,
                        )

                        top_score = max([r.get("score", 0) for r in retrieved], default=0)
                        chunks_used = len(retrieved)

                        context_block, citations = build_context_with_citations(retrieved)
                        user_prompt = explain_prompt(level, st.session_state.report_text, context_block)
                        answer = generate_text(SYSTEM_BASE, user_prompt)
                        answer = enforce_disclaimer(answer)
                        t1 = time.time()

                    lat = t1 - t0
                    st.success(f"Done in {lat:.2f}s")
                    st.caption(f"Run stats: **{chunks_used} chunks used** | **top score {top_score:.3f}**")

                    with st.container(border=True):
                        st.markdown("### Explanation in Plain English")
                        st.write(answer)

                    citation_lines = citations_to_ui_lines(citations)
                    show_citations(citation_lines, citations)

        # TAB 2: Extract 
        with tabs[1]:
            st.subheader("Extract structured information (JSON)")
            if st.button("Extract Fields"):
                ok, err = validate_report_input(st.session_state.report_text)
                if not ok:
                    st.error(err)
                else:
                    with st.spinner("Extracting..."):
                        t0 = time.time()
                        user_prompt = extract_prompt(st.session_state.report_text)
                        answer = generate_text(SYSTEM_BASE, user_prompt)
                        t1 = time.time()

                    st.success(f"Done in {t1 - t0:.2f}s")

                    # Try to render JSON nicely
                    try:
                        parsed = json.loads(answer)
                        st.json(parsed)

                        # Pretty views (if keys exist)
                        findings = parsed.get("findings", [])
                        impression = parsed.get("impression", [])
                        entities = parsed.get("entities", [])

                        if findings:
                            with st.container(border=True):
                                st.markdown("### Findings")
                                if isinstance(findings, list):
                                    st.table({"findings": findings})
                                else:
                                    st.write(findings)

                        if impression:
                            with st.container(border=True):
                                st.markdown("### Impression")
                                if isinstance(impression, list):
                                    st.table({"impression": impression})
                                else:
                                    st.write(impression)

                        if entities:
                            with st.container(border=True):
                                st.markdown("### Key terms")
                                if isinstance(entities, list):
                                    st.write(" • " + " • ".join([str(e) for e in entities[:25]]))
                                else:
                                    st.write(entities)
                    except Exception:
                        st.warning("Model did not return valid JSON. Showing raw output:")
                        st.text(answer)

        # TAB 3: Evidence Q&A 
        with tabs[2]:
            st.subheader("Ask questions (answers must cite sources)")

            # Display history
            for turn in st.session_state.chat_history:
                with st.chat_message(turn.role):
                    st.write(turn.content)

            st.markdown("#### Suggested questions")
            sq1, sq2, sq3 = st.columns(3)

            # Suggested questions --> store in session
            with sq1:
                if st.button("Explain the impression"):
                    st.session_state.pending_question = "Explain the impression in simple terms."
            with sq2:
                if st.button("Is anything urgent?"):
                    st.session_state.pending_question = "Is there anything urgent or concerning in this report?"
            with sq3:
                if st.button("Define key terms"):
                    st.session_state.pending_question = "Define key terms mentioned in this report (e.g., opacity, atelectasis, effusion)."

            question = st.session_state.pending_question or st.chat_input("Ask a question about the report...")

            # If the question came from a suggested button, clear it after consuming
            if st.session_state.pending_question is not None and question == st.session_state.pending_question:
                st.session_state.pending_question = None

            if question is not None:
                ok_r, err_r = validate_report_input(st.session_state.report_text)
                ok_q, err_q = validate_question_input(question)

                if not ok_r:
                    st.error(err_r)
                    return
                if not ok_q:
                    st.error(err_q)
                    return

                # Add user message to chat
                st.session_state.chat_history.append(ChatTurn(role="user", content=question))
                with st.chat_message("user"):
                    st.write(question)

                with st.spinner("Retrieving evidence + answering..."):
                    t0 = time.time()
                    retrieved = retrieve_top_k(
                        query=question,
                        top_k=top_k,
                        min_score=min_score,
                        final_top_k=final_top_k,
                    )

                    ok_ev, err_ev = validate_retrieval_results(retrieved, min_results=2)
                    if not ok_ev:
                        assistant_reply = err_ev
                        assistant_reply = enforce_disclaimer(assistant_reply)
                        t1 = time.time()
                        with st.chat_message("assistant"):
                            st.write(assistant_reply)
                        st.session_state.chat_history.append(ChatTurn(role="assistant", content=assistant_reply))
                        st.info(f"Done in {t1 - t0:.2f}s (no sufficient evidence)")
                        return

                    context_block, citations = build_context_with_citations(retrieved)
                    user_prompt = qa_prompt(question, st.session_state.report_text, context_block)

                    # Context management: include last N turns
                    trimmed = trim_history(st.session_state.chat_history[:-1], max_turns=6)
                    history_msgs = history_to_messages(trimmed)

                    assistant_reply = generate_text(SYSTEM_BASE, user_prompt, history_messages=history_msgs)
                    assistant_reply = enforce_disclaimer(assistant_reply)
                    t1 = time.time()

                with st.chat_message("assistant"):
                    st.write(assistant_reply)

                st.session_state.chat_history.append(ChatTurn(role="assistant", content=assistant_reply))
                st.info(f"Done in {t1 - t0:.2f}s")

                citation_lines = citations_to_ui_lines(citations)

                with st.expander("Top evidence snippets (preview)", expanded=False):
                    for c in citations[:3]:
                        with st.container(border=True):
                            st.markdown(f"**[{c.cid}] {c.source} — page {c.page}** (score: {c.score:.3f})")
                            st.caption(c.snippet)
                        
                show_citations(citation_lines, citations)


if __name__ == "__main__":
    main()
