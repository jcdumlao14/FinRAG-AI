import sys
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from rag.pipeline import FinRAGPipeline
from monitoring.feedback import save_feedback


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FinRAG AI",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("📊 FinRAG AI")

st.markdown(
    """
### Financial Research Assistant

Ask questions about the indexed company financial reports.

FinRAG AI combines **vector search, BM25 lexical search,
and Hybrid Reciprocal Rank Fusion (RRF)** to retrieve
financial evidence before generating an answer with
**Gemini 2.5 Flash**.
"""
)

st.divider()


# ============================================================
# LOAD PIPELINE
# ============================================================

@st.cache_resource
def load_pipeline():

    return FinRAGPipeline()


try:

    with st.spinner(
        "Initializing FinRAG AI..."
    ):

        pipeline = load_pipeline()

except Exception as error:

    st.error(
        "Unable to initialize FinRAG AI."
    )

    st.exception(error)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About FinRAG AI")

    st.markdown(
        """
        **Retrieval**

        - Vector Search
        - BM25
        - Hybrid RRF

        **Generation**

        - Gemini 2.5 Flash

        **Knowledge Base**

        - Apple 2025
        - Microsoft 2025
        - NVIDIA 2026
        """
    )

    st.divider()

    st.markdown(
        """
        **Evaluation**

        Current retrieval benchmark:

        **100% Top-5 accuracy**

        on the 9-question evaluation set.
        """
    )

    st.divider()

    st.caption(
        "LLM Zoomcamp 2026 Capstone Project"
    )


# ============================================================
# SESSION STATE
# ============================================================

if "last_result" not in st.session_state:

    st.session_state[
        "last_result"
    ] = None


if "last_question" not in st.session_state:

    st.session_state[
        "last_question"
    ] = ""


if "feedback" not in st.session_state:

    st.session_state[
        "feedback"
    ] = None


# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader(
    "Ask a Financial Question"
)

question = st.text_area(
    "Enter your question",
    placeholder=(
        "Example: What was Apple's total net sales "
        "in fiscal year 2025?"
    ),
    height=100,
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.markdown("**Example questions:**")

example_questions = [
    "What was Apple's total net sales in fiscal year 2025?",
    "What was Microsoft's total revenue in fiscal year 2025?",
    "What was NVIDIA's revenue in fiscal year 2026?",
    "What was Tesla's total revenue in fiscal year 2025?",
]

for example in example_questions:

    st.caption(
        f"• {example}"
    )


# ============================================================
# ASK BUTTON
# ============================================================

ask_button = st.button(
    "🔎 Ask FinRAG AI",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a financial question."
        )

    else:

        with st.spinner(
            "Searching financial documents and generating a grounded answer..."
        ):

            try:

                result = pipeline.answer(
                    question.strip(),
                    top_k=5,
                )

                st.session_state[
                    "last_result"
                ] = result

                st.session_state[
                    "last_question"
                ] = question.strip()

                st.session_state[
                    "feedback"
                ] = None

            except Exception as error:

                st.error(
                    "An error occurred while processing the question."
                )

                st.exception(error)

                st.stop()


# ============================================================
# DISPLAY LAST ANSWER
# ============================================================

result = st.session_state.get(
    "last_result"
)

last_question = st.session_state.get(
    "last_question",
)


if result is not None:

    # --------------------------------------------------------
    # ANSWER
    # --------------------------------------------------------

    st.subheader(
        "💡 Answer"
    )

    st.success(
        result["answer"]
    )


    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    st.subheader(
        "📚 Retrieved Sources"
    )

    sources = result.get(
        "sources",
        [],
    )

    if sources:

        for i, source in enumerate(
            sources,
            start=1,
        ):

            with st.expander(
                f"Source {i}: "
                f"{source.get('company', 'Unknown')} "
                f"({source.get('year', 'Unknown')})"
            ):

                st.write(
                    f"**Company:** "
                    f"{source.get('company', 'Unknown')}"
                )

                st.write(
                    f"**Fiscal Year:** "
                    f"{source.get('year', 'Unknown')}"
                )

                st.write(
                    f"**Document:** "
                    f"`{source.get('filename', 'Unknown')}`"
                )

                st.write(
                    f"**Chunk:** "
                    f"`{source.get('chunk_id', 'Unknown')}`"
                )

    else:

        st.info(
            "No source information was returned."
        )


    # --------------------------------------------------------
    # FEEDBACK
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "📝 Was this answer helpful?"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "👍 Helpful",
            use_container_width=True,
            key="positive_feedback",
        ):

            save_feedback(
                question=last_question,
                answer=result["answer"],
                feedback="positive",
                sources=sources,
            )

            st.session_state[
                "feedback"
            ] = "positive"

    with col2:

        if st.button(
            "👎 Not Helpful",
            use_container_width=True,
            key="negative_feedback",
        ):

            save_feedback(
                question=last_question,
                answer=result["answer"],
                feedback="negative",
                sources=sources,
            )

            st.session_state[
                "feedback"
            ] = "negative"


    # --------------------------------------------------------
    # FEEDBACK CONFIRMATION
    # --------------------------------------------------------

    if st.session_state.get(
        "feedback"
    ) == "positive":

        st.success(
            "Thank you for your feedback! "
            "Your response has been recorded."
        )

    elif st.session_state.get(
        "feedback"
    ) == "negative":

        st.info(
            "Thank you. Your feedback has been recorded "
            "and will help improve FinRAG AI."
        )
