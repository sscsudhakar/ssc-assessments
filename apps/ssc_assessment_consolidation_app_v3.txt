# ======================================================================
# SSC ASSESSMENT CONSOLIDATION APP (V3)
#
# Enhancements vs earlier version:
#   a) Executive Summary, Quick Wins, Roadmap for P2P CoE are now
#      clearly driven by dimension scores + low/high patterns + answers.
#   b) Industry standards from config/industry_context.xlsx are loaded
#      and provided to AI; the text explicitly references benchmarks.
#   c) In addition to uploading client CSVs from the Client App, SSC
#      consultants can upload an additional notes file (CSV/TXT/DOCX).
#      These notes are also passed into AI to influence recommendations.
#
# Expected client CSV structure (from V8.1.1 client app):
#   columns:
#       assessment_name, dimension, index, question, answer,
#       ai_score, final_score, reason, weight, help_text, role
#
# Industry standards file:
#   config/industry_context.xlsx  (or ./industry_context.xlsx)
#   columns (example):
#       industry, dimension, benchmark_maturity,
#       best_practices, common_challenges,
#       automation_expectation, sla_expectations
#
# OpenAI:
#   Requires OPENAI_API_KEY in environment or Streamlit secrets.
# ======================================================================

import os
import json
import time

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests

# Optional DOCX support for SSC notes
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

st.set_page_config(
    page_title="SSC Assessment Consolidation App (V3)",
    layout="wide"
)

# ---------------------------------------------------------
# CONFIG: Industry standards file locations
# ---------------------------------------------------------
INDUSTRY_CONTEXT_CANDIDATES = [
    "config/industry_context.xlsx",
    "industry_context.xlsx",
]


# ---------------------------------------------------------
# Helper: Load Industry Context
# ---------------------------------------------------------
@st.cache_data
def load_industry_context():
    """
    Try to load industry_context.xlsx with columns like:
    industry, dimension, benchmark_maturity, best_practices,
    common_challenges, automation_expectation, sla_expectations
    """
    for path in INDUSTRY_CONTEXT_CANDIDATES:
        if os.path.exists(path):
            try:
                df = pd.read_excel(path)
                return df
            except Exception as e:
                st.sidebar.warning(f"Could not load industry context from '{path}': {e}")
                return None
    return None


def get_industry_context_for_ai(industry_name: str, dim_stats: pd.DataFrame, ctx_df: pd.DataFrame | None) -> str:
    """
    Build a consolidated, human-readable industry standards summary
    for the selected industry and the dimensions present in the data.
    """
    if ctx_df is None or ctx_df.empty or not industry_name:
        return (
            "No specific industry standards file was available. "
            "Use general best practices for each dimension."
        )

    lines = []
    lines.append(f"Industry: {industry_name}")
    lines.append("Industry benchmarks and practices by dimension:")

    try:
        for dim in dim_stats["dimension"].unique():
            subset = ctx_df[
                (ctx_df["industry"].str.lower() == industry_name.lower())
                & (ctx_df["dimension"].str.lower() == str(dim).lower())
            ]
            if subset.empty:
                continue

            lines.append(f"\nDimension: {dim}")
            for _, r in subset.iterrows():
                bm = r.get("benchmark_maturity", "")
                bp = r.get("best_practices", "")
                cc = r.get("common_challenges", "")
                auto = r.get("automation_expectation", "")
                sla = r.get("sla_expectations", "")

                if bm:
                    lines.append(f"- Benchmark Maturity: {bm}")
                if bp:
                    lines.append(f"- Best Practices: {bp}")
                if cc:
                    lines.append(f"- Common Challenges: {cc}")
                if auto:
                    lines.append(f"- Automation Expectation: {auto}")
                if sla:
                    lines.append(f"- SLA Expectations: {sla}")

        if len(lines) == 2:  # nothing added beyond header
            return (
                f"No matching industry standards found for industry '{industry_name}'. "
                "Use general best practices."
            )

        return "\n".join(lines)
    except Exception:
        return (
            "Industry standards could not be fully processed. "
            "Use general best practices and typical maturity patterns."
        )


# ---------------------------------------------------------
# Helper: OpenAI call with simple retry + friendly errors
# ---------------------------------------------------------
def call_openai_chat(prompt: str, max_retries: int = 3) -> str:
    """
    Call OpenAI Chat API with simple retry + backoff.
    Returns:
      - model output text on success
      - string starting with 'ERROR:' on failure (user-friendly message)
    """
    # Priority: Streamlit secrets if available
    api_key = None
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return "ERROR: AI key is not configured. Please contact SSC support."

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior management consultant specialising in P2P CoEs, "
                    "Finance transformations, and IT/ERP operating models. "
                    "You analyse consolidated maturity assessment data, identify quick wins, "
                    "and design practical roadmaps."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.25,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.HTTPError as http_err:
            status = http_err.response.status_code if http_err.response is not None else None

            if status == 429 and attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                st.warning(
                    f"AI rate limit reached (429). "
                    f"The system will retry automatically in {wait_time} seconds. Please wait…"
                )
                time.sleep(wait_time)
                continue

            return (
                "ERROR: The AI service returned an error. "
                "Possible reasons:\n"
                "- Too many users running consolidation simultaneously\n"
                "- API rate/quota limits reached\n"
                "- Temporary overload or configuration issue\n\n"
                "Please retry after 1–2 minutes or contact SSC Delivery CoE."
            )

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                st.warning(
                    f"AI request timed out. Retrying in {wait_time} seconds…"
                )
                time.sleep(wait_time)
                continue
            return "ERROR: The AI service is taking too long to respond. Please try again."

        except Exception as e:
            return f"ERROR: Unexpected issue while contacting AI – {e}"

    return "ERROR: Unable to reach the AI service after multiple retries."


# ---------------------------------------------------------
# Helper: Read SSC notes file (CSV/TXT/DOCX)
# ---------------------------------------------------------
def read_notes_file(uploaded_file) -> str:
    """
    Read SSC notes from an uploaded file and return as plain text.
    Supports CSV, TXT/MD, and DOCX (if python-docx is installed).
    """
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file)
            return df.to_string(index=False)
        except Exception as e:
            st.warning(f"Could not read notes CSV: {e}")
            return ""
    elif name.endswith(".txt") or name.endswith(".md"):
        try:
            return uploaded_file.read().decode("utf-8", errors="ignore")
        except Exception as e:
            st.warning(f"Could not read notes text file: {e}")
            return ""
    elif name.endswith(".docx"):
        if not DOCX_AVAILABLE:
            st.warning(
                "DOCX notes file uploaded, but 'python-docx' is not installed. "
                "Add 'python-docx' to requirements.txt or use CSV/TXT instead."
            )
            return ""
        try:
            doc = Document(uploaded_file)
            paras = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paras)
        except Exception as e:
            st.warning(f"Could not read DOCX file: {e}")
            return ""
    else:
        st.warning(
            "Unsupported notes format. Please upload CSV, TXT, MD, or DOCX."
        )
        return ""


# ---------------------------------------------------------
# Helper: Compute dimension-level stats
# ---------------------------------------------------------
def compute_dimension_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with:
    dimension, avg_ai_score, avg_final_score, question_count
    plus min/max ai_score for context.
    """
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby("dimension").agg(
        avg_ai_score=("ai_score", "mean"),
        avg_final_score=("final_score", "mean"),
        question_count=("index", "count"),
        min_ai_score=("ai_score", "min"),
        max_ai_score=("ai_score", "max"),
    ).reset_index()

    grouped["avg_ai_score"] = grouped["avg_ai_score"].round(2)
    grouped["avg_final_score"] = grouped["avg_final_score"].round(2)

    return grouped


# ---------------------------------------------------------
# Helper: Build AI prompt for EXEC SUMMARY + QUICK WINS + ROADMAP
# ---------------------------------------------------------
def build_consolidation_prompt(
    client_name: str,
    industry_name: str,
    dim_stats: pd.DataFrame,
    df_all: pd.DataFrame,
    industry_context_text: str,
    notes_text: str,
) -> str:
    """
    Build a rich prompt for AI that:
    - Includes dimension-level stats
    - Highlights lowest & highest maturity areas
    - Includes a sample of low-scoring questions/answers
    - Includes SSC notes
    - Includes industry standards text
    """
    if dim_stats is None or dim_stats.empty:
        dim_part = "No dimension statistics could be computed."
    else:
        dim_part = dim_stats.to_string(index=False)

    # Identify weakest and strongest dimensions by avg_ai_score
    weakest_dims = []
    strongest_dims = []
    if not dim_stats.empty:
        weakest_dims = (
            dim_stats.sort_values("avg_ai_score", ascending=True)
            .head(2)["dimension"]
            .tolist()
        )
        strongest_dims = (
            dim_stats.sort_values("avg_ai_score", ascending=False)
            .head(2)["dimension"]
            .tolist()
        )

    # Extract a limited set of low-scoring question records for context
    low_scoring = df_all[df_all["ai_score"] <= 2].copy()
    if low_scoring.empty:
        low_snippet = "No AI scores at or below 2. Most areas are at least mid-level maturity."
    else:
        cols_for_ai = [
            "dimension",
            "index",
            "question",
            "answer",
            "ai_score",
            "reason",
            "role",
        ]
        low_scoring = low_scoring[cols_for_ai].head(30)  # cap to avoid huge prompt
        low_snippet = low_scoring.to_json(orient="records", force_ascii=False)

    # Notes text
    notes_section = notes_text if notes_text.strip() else "No additional SSC notes were provided."

    # Build final prompt
    prompt = f"""
Client: {client_name or "Unknown Client"}
Industry: {industry_name or "Not specified"}

You are consolidating a P2P / Finance / CoE maturity assessment for this client.

1) Dimension-level maturity statistics (from 1 to 5):
{dim_part}

- Weakest dimensions by average AI score (from the above): {weakest_dims}
- Strongest dimensions by average AI score: {strongest_dims}

2) Sample of low-scoring questions (ai_score <= 2) with reasons and roles:
{low_snippet}

3) Industry standards and benchmarks:
{industry_context_text}

4) Additional SSC qualitative notes (interview findings, observations):
{notes_section}

Please synthesise all of this into a concise, consulting-style set of outputs:

A) Executive Summary (5–8 bullet points)
   - Mention overall maturity, key strengths, key risks,
     and how the client compares to typical industry benchmarks.

B) Quick Wins (4–8 items)
   - Actions that can be done within 4–8 weeks with low to medium effort
   - Clearly reference which dimension(s) they address, and if relevant,
     how they close the gap vs industry standards.

C) Longer-Term Roadmap to P2P CoE (6–18 months)
   - Organise into phases (e.g., Phase 1: Stabilise, Phase 2: Optimise,
     Phase 3: Transform).
   - Mention target maturity improvements per dimension where relevant.
   - Highlight digital/automation levers (e.g., workflow, RPA, analytics).

D) Final Recommendations / SSC Point of View
   - 4–6 bullets summarising what SSC should strongly recommend to the client
     (e.g., establish a P2P CoE, centralise AP, standardise policies, etc.)
   - Explicitly mention where the client is below, on par, or above industry
     and why this matters.

Format your answer in clear Markdown with headings:
- ## Executive Summary
- ## Quick Wins (4–8 weeks)
- ## Roadmap to P2P CoE (6–18 months)
- ## Final SSC Recommendations

Make the content specific to the data above. Do NOT use generic boilerplate.
    """
    return prompt


# ---------------------------------------------------------
# UI: Main App
# ---------------------------------------------------------
def main():
    st.title("SSC Assessment Consolidation App (V3)")
    st.caption(
        "Upload multiple AI-scored assessment CSVs from the Client App, "
        "optionally add SSC notes, and generate a consolidated view with "
        "Executive Summary, Quick Wins and P2P CoE Roadmap."
    )

    st.markdown("---")

    # Load industry context from Excel
    industry_df = load_industry_context()

    # 1) Upload client CSVs (multiple)
    st.subheader("Step 1 – Upload Client Assessment CSV Files")
    client_files = st.file_uploader(
        "Upload one or more CSV files generated by the SSC Client App",
        type=["csv"],
        accept_multiple_files=True,
    )

    if not client_files:
        st.info("Please upload at least one client CSV file to proceed.")
        return

    # Read and combine all CSVs
    dfs = []
    for f in client_files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = f.name
            dfs.append(df)
        except Exception as e:
            st.error(f"Failed to read '{f.name}': {e}")
            return

    df_all = pd.concat(dfs, ignore_index=True)

    required_cols = {
        "assessment_name",
        "dimension",
        "index",
        "question",
        "answer",
        "ai_score",
        "final_score",
        "reason",
        "weight",
        "help_text",
        "role",
    }
    missing = required_cols - set(df_all.columns)
    if missing:
        st.error(
            "Some required columns are missing in the uploaded CSVs: "
            + ", ".join(missing)
        )
        return

    # 2) Optional SSC notes upload
    st.subheader("Step 2 – Optional SSC Notes (Interviews & Observations)")
    notes_file = st.file_uploader(
        "Upload SSC notes file (optional) – CSV, TXT, MD, or DOCX",
        type=["csv", "txt", "md", "docx"],
        key="notes_uploader",
    )
    notes_text = read_notes_file(notes_file)

    if notes_text:
        with st.expander("Preview of SSC Notes (first 30 lines)", expanded=False):
            preview_lines = "\n".join(notes_text.splitlines()[:30])
            st.text(preview_lines)

    # 3) Choose client name & industry for consolidation
    st.subheader("Step 3 – Consolidation Context")

    client_name_default = ""
    if "assessment_name" in df_all.columns:
        # Not exactly client, but can help. Let user override.
        client_name_default = str(df_all["assessment_name"].iloc[0])

    client_name = st.text_input(
        "Client / Engagement Name (for narrative context)",
        value=client_name_default,
    )

    # Industry from standards file if available
    industry_name = ""
    if industry_df is not None and not industry_df.empty and "industry" in industry_df.columns:
        industries = sorted(industry_df["industry"].dropna().unique().tolist())
        industries.append("Other / Not Listed")
        selected = st.selectbox("Select Industry (for benchmark context)", industries)
        if selected == "Other / Not Listed":
            industry_name = st.text_input("Specify Industry", value="")
        else:
            industry_name = selected
    else:
        industry_name = st.text_input("Industry (free text – if standards file not found)", value="")

    st.markdown("---")

    # 4) Show aggregated stats & charts
    st.subheader("Step 4 – Review Consolidated Scores")

    df_all["ai_score"] = pd.to_numeric(df_all["ai_score"], errors="coerce")
    df_all["final_score"] = pd.to_numeric(df_all["final_score"], errors="coerce")

    dim_stats = compute_dimension_stats(df_all)

    if dim_stats.empty:
        st.warning("Could not compute dimension statistics. Check your CSV data.")
        return

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("**Dimension-level Maturity Summary**")
        st.dataframe(dim_stats, use_container_width=True)

    with col_right:
        fig = px.bar(
            dim_stats,
            x="dimension",
            y="avg_ai_score",
            title="Average AI Score by Dimension",
            labels={"avg_ai_score": "Average AI Score (1–5)", "dimension": "Dimension"},
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(
            dim_stats,
            x="dimension",
            y="question_count",
            title="Number of Questions per Dimension",
            labels={"question_count": "Question Count", "dimension": "Dimension"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # 5) Generate AI-based Executive Summary, Quick Wins, Roadmap, Recommendations
    st.subheader("Step 5 – Generate AI-Based Consolidated Narrative")

    # Build industry context text for AI
    industry_context_text = get_industry_context_for_ai(
        industry_name=industry_name,
        dim_stats=dim_stats,
        ctx_df=industry_df,
    )

    if st.button("Generate Consolidated Executive Summary & Recommendations"):
        with st.spinner("Calling AI to generate consolidated narrative..."):
            prompt = build_consolidation_prompt(
                client_name=client_name,
                industry_name=industry_name,
                dim_stats=dim_stats,
                df_all=df_all,
                industry_context_text=industry_context_text,
                notes_text=notes_text,
            )

            ai_text = call_openai_chat(prompt)

        if ai_text.startswith("ERROR:"):
            st.error(ai_text)
        else:
            st.markdown("---")
            st.markdown("### Consolidated Narrative")
            st.markdown(ai_text)


if __name__ == "__main__":
    main()
