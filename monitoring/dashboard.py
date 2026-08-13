import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from monitoring.feedback import load_feedback


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FinRAG AI Monitoring",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("📊 FinRAG AI Monitoring Dashboard")

st.markdown(
    """
Monitor user queries, feedback, retrieved sources, and
application usage for the FinRAG AI financial research assistant.
"""
)

st.divider()


# ============================================================
# LOAD FEEDBACK
# ============================================================

records = load_feedback()


if not records:

    st.info(
        "No feedback data is available yet. "
        "Use the FinRAG AI application and submit feedback first."
    )

    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

df = pd.DataFrame(records)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
)

df["date"] = df["timestamp"].dt.date

df["feedback"] = (
    df["feedback"]
    .astype(str)
    .str.lower()
)


# ============================================================
# SUMMARY METRICS
# ============================================================

total_queries = len(df)

positive_feedback = (
    df["feedback"]
    .eq("positive")
    .sum()
)

negative_feedback = (
    df["feedback"]
    .eq("negative")
    .sum()
)

feedback_rate = (
    (positive_feedback / total_queries) * 100
    if total_queries
    else 0
)


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Queries",
        total_queries,
    )

with col2:

    st.metric(
        "👍 Positive",
        positive_feedback,
    )

with col3:

    st.metric(
        "👎 Negative",
        negative_feedback,
    )

with col4:

    st.metric(
        "Positive Feedback Rate",
        f"{feedback_rate:.1f}%",
    )


st.divider()


# ============================================================
# FEEDBACK DISTRIBUTION
# ============================================================

st.subheader("👍👎 Feedback Distribution")

feedback_counts = (
    df["feedback"]
    .value_counts()
    .rename_axis("feedback")
    .reset_index(name="count")
)

st.bar_chart(
    feedback_counts.set_index("feedback")
)


# ============================================================
# QUERIES OVER TIME
# ============================================================

st.subheader("📈 Queries Over Time")

queries_over_time = (
    df.groupby("date")
    .size()
    .rename("queries")
)

st.line_chart(
    queries_over_time
)


# ============================================================
# FEEDBACK OVER TIME
# ============================================================

st.subheader("📊 Feedback Over Time")

feedback_over_time = (
    df.groupby(
        ["date", "feedback"]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

st.line_chart(
    feedback_over_time
)


# ============================================================
# QUESTIONS BY COMPANY
# ============================================================

st.subheader("🏢 Questions by Company")

company_records = []

for _, row in df.iterrows():

    sources = row.get(
        "sources",
        [],
    )

    companies = set()

    if isinstance(
        sources,
        list,
    ):

        for source in sources:

            if isinstance(
                source,
                dict,
            ):

                company = source.get(
                    "company"
                )

                if company:
                    companies.add(
                        str(company)
                    )

    for company in companies:

        company_records.append(
            company
        )


if company_records:

    company_counts = (
        pd.Series(
            company_records
        )
        .value_counts()
        .rename_axis("company")
        .reset_index(
            name="queries"
        )
    )

    st.bar_chart(
        company_counts.set_index(
            "company"
        )
    )

else:

    st.info(
        "Company information is not available "
        "in the current feedback records."
    )


# ============================================================
# RETRIEVED SOURCES
# ============================================================

st.subheader("📚 Retrieved Sources by Document")

document_records = []

for _, row in df.iterrows():

    sources = row.get(
        "sources",
        [],
    )

    if isinstance(
        sources,
        list,
    ):

        for source in sources:

            if isinstance(
                source,
                dict,
            ):

                filename = source.get(
                    "filename"
                )

                if filename:

                    document_records.append(
                        str(filename)
                    )


if document_records:

    document_counts = (
        pd.Series(
            document_records
        )
        .value_counts()
        .rename_axis("document")
        .reset_index(
            name="retrieved_sources"
        )
    )

    st.bar_chart(
        document_counts.set_index(
            "document"
        )
    )

else:

    st.info(
        "Document information is not available "
        "in the current feedback records."
    )


# ============================================================
# RECENT QUESTIONS
# ============================================================

st.subheader("📝 Recent Questions")

recent_columns = [
    "timestamp",
    "question",
    "feedback",
]

available_columns = [
    column
    for column in recent_columns
    if column in df.columns
]

recent = (
    df[available_columns]
    .sort_values(
        "timestamp",
        ascending=False,
    )
    .head(10)
)

st.dataframe(
    recent,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "FinRAG AI Monitoring | LLM Zoomcamp 2026 Capstone Project"
)
