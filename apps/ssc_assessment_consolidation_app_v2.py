
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import requests
import json

st.set_page_config(
    page_title="SSC Universal Assessment – Consolidation (V2)",
    layout="wide"
)

DEFAULT_TEMPLATE_PATH = "ssc_assessment_template.xlsx"

# Logo paths
SSC_LOGO_PATH = "ssc_logo.png"
CLIENT_LOGO_PATH = "client_logo.png"

# ---------------------------
# OPENAI HELPER
# ---------------------------

def call_openai_chat(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "ERROR: Missing OPENAI_API_KEY environment variable."

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
                    "You are an experienced management consultant. "
                    "You synthesise assessment results into executive summaries and recommendations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "ERROR: Timeout contacting OpenAI API. Please try again (network was too slow)."
    except Exception as e:
        return f"ERROR: {e}"


def generate_ai_recommendations(assessment_name: str, dim_scores_ai: dict, dim_scores_final: dict, description: str = "") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "AI recommendations not generated – missing OPENAI_API_KEY.\n"
            "Set your OpenAI key as an environment variable or in Streamlit secrets and rerun the app."
        )

    prompt = f"""
You are reviewing a consolidated maturity assessment called "{assessment_name}".
Scores are on a 1–5 scale (1 = very poor, 5 = world-class).

Short description of the assessment:
{description}

Dimension scores (averages across all stakeholders):
AI Average Scores (pre-consolidation):
{json.dumps(dim_scores_ai, indent=2)}

Consultant Final Consolidated Scores:
{json.dumps(dim_scores_final, indent=2)}

Using this information, produce a concise, CxO-ready output with the following sections:

1) Executive Summary
   - 3–5 bullet points capturing the overall maturity and big-picture story.

2) Key Gaps & Pain Points
   - 5–7 bullets grouped under the key dimensions (use the dimension names from the data).
   - Focus on structural and recurring issues, not one-off comments.

3) Quick-Win Recommendations (0–3 months)
   - 5–7 specific actions the client can take quickly with high impact and low effort.

4) Medium-Term Capability / CoE / Operating Model Initiatives (6–18 months)
   - 4–6 initiatives that build sustainable capability in this assessment area.
   - Each initiative should be 1–2 lines, outcome-oriented.

Keep the tone consulting-grade, practical, and suitable for inclusion in a client-facing deck.
Do NOT restate the numeric scores; interpret them.
"""

    raw = call_openai_chat(prompt)
    return raw

# ---------------------------
# TEMPLATE LOADER
# ---------------------------

def load_assessment_meta(file):
    try:
        xls = pd.ExcelFile(file)
        meta = pd.read_excel(xls, sheet_name="assessment_meta")
        meta_dict = {row["field"]: str(row["value"]) for _, row in meta.iterrows()}
        assessment_name = meta_dict.get("assessment_name", "SSC Assessment")
        description = meta_dict.get("description", "")
        return assessment_name, description
    except Exception:
        return "SSC Assessment", ""

# ---------------------------
# MAIN
# ---------------------------

def main():
    # --- Top branding with logos ---
    logo_cols = st.columns([1, 4, 1])

    with logo_cols[0]:
        if os.path.exists(SSC_LOGO_PATH):
            st.image(SSC_LOGO_PATH, use_column_width=True)
        else:
            st.write("")

    with logo_cols[1]:
        st.markdown(
            "<h1 style='text-align:center;color:#002060;font-family:Segoe UI, sans-serif;'>"
            "SSC Universal Assessment – Consolidation (V2)</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h4 style='text-align:center;color:gray;font-family:Segoe UI, sans-serif;'>"
            "Multi-Stakeholder Aggregation + Consultant Override + AI Recommendations</h4>",
            unsafe_allow_html=True,
        )

    with logo_cols[2]:
        if os.path.exists(CLIENT_LOGO_PATH):
            st.image(CLIENT_LOGO_PATH, use_column_width=True)
        else:
            st.write("")

    st.markdown("---")

    # Sidebar: optional template upload
    st.sidebar.markdown("### Assessment Template (Optional)")
    template_file = st.sidebar.file_uploader("Upload the same Assessment Template (.xlsx) used for the diagnostic", type=["xlsx"])
    if template_file is not None:
        assessment_name, assessment_description = load_assessment_meta(template_file)
    else:
        if os.path.exists(DEFAULT_TEMPLATE_PATH):
            assessment_name, assessment_description = load_assessment_meta(DEFAULT_TEMPLATE_PATH)
        else:
            assessment_name, assessment_description = "SSC Assessment", ""

    st.sidebar.markdown(f"**Assessment:** {assessment_name}")
    if assessment_description:
        st.sidebar.caption(assessment_description)

    st.markdown("#### Step 1: Upload Question-Level CSV Files from Individual Assessments")
    st.write(
        "Upload one or more CSV files exported from the **Universal Assessment App (V5)**. "
        "Each file should contain columns like: "
        "`assessment_name, dimension, index, question, answer, ai_score, final_score, reason, weight, help_text`."
    )

    uploaded_files = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload at least one CSV file to begin consolidation.")
        return

    all_dfs = []
    for f in uploaded_files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = f.name
            all_dfs.append(df)
        except Exception as e:
            st.error(f"Could not read file {f.name}: {e}")

    if not all_dfs:
        st.error("No valid CSV files could be read. Please check the format and try again.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)

    required_cols = {"dimension", "index", "question", "answer", "ai_score", "final_score"}
    missing = required_cols - set(df_all.columns)
    if missing:
        st.error(f"The uploaded files are missing required columns: {missing}")
        st.stop()

    st.success(f"Loaded {len(df_all)} rows from {len(uploaded_files)} file(s).")

    with st.expander("Preview Combined Question-Level Data", expanded=False):
        st.dataframe(df_all.head(50))

    # Step 2: Aggregated View by Dimension & Question
    st.markdown("#### Step 2: Aggregated View by Dimension & Question")

    agg_q = (
        df_all
        .groupby(["dimension", "index", "question"], as_index=False)
        .agg(
            n_responses=("final_score", "count"),
            avg_ai_score=("ai_score", "mean"),
            avg_final_score=("final_score", "mean"),
        )
    )

    st.dataframe(agg_q)

    # Step 3: Dimension-Level Consolidated Scores
    st.markdown("#### Step 3: Dimension-Level Consolidated Scores")

    agg_dim = (
        df_all
        .groupby("dimension", as_index=False)
        .agg(
            avg_ai_score=("ai_score", "mean"),
            avg_final_score=("final_score", "mean"),
            n_responses=("final_score", "count"),
        )
    )

    col1, col2 = st.columns(2)
    with col1:
        st.write("**AI Average Scores (from all responses)**")
        st.dataframe(agg_dim[["dimension", "avg_ai_score", "n_responses"]])
    with col2:
        st.write("**Consultant Average Scores (from all responses)**")
        st.dataframe(agg_dim[["dimension", "avg_final_score", "n_responses"]])

    # Build dictionaries
    dim_scores_ai = {row["dimension"]: row["avg_ai_score"] for _, row in agg_dim.iterrows()}
    dim_scores_final_default = {row["dimension"]: row["avg_final_score"] for _, row in agg_dim.iterrows()}
    dimensions = list(dim_scores_ai.keys())

    # Step 4: Consultant Override
    st.markdown("#### Step 4: Consultant Override – Final Dimension Maturity Scores")

    dim_scores_final = {}
    for dim in dimensions:
        ai_val = float(dim_scores_ai.get(dim, 0.0))
        default_final = float(dim_scores_final_default.get(dim, ai_val))
        st.write(f"**{dim}**")
        final_val = st.slider(
            f"{dim} – Final Consolidated Score (1–5)",
            min_value=1.0,
            max_value=5.0,
            value=float(round(default_final if default_final > 0 else 3.0)),
            step=0.1,
        )
        dim_scores_final[dim] = final_val

    # Step 5: Visualise
    st.markdown("#### Step 5: Visualise Consolidated Maturity")

    radar_df = pd.DataFrame({
        "Dimension": dimensions,
        "AI Score": [dim_scores_ai.get(d, 0.0) for d in dimensions],
        "Final Score": [dim_scores_final.get(d, 0.0) for d in dimensions],
    })

    fig_radar = px.line_polar(
        radar_df,
        r="Final Score",
        theta="Dimension",
        line_close=True,
        range_r=[0, 5],
        title="Consolidated Final Maturity (Consultant View)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    fig_bar = px.bar(
        radar_df.melt(id_vars="Dimension", value_vars=["AI Score", "Final Score"], var_name="Type", value_name="Score"),
        x="Dimension",
        y="Score",
        color="Type",
        barmode="group",
        range_y=[0, 5],
        title="AI vs Consultant Final Scores by Dimension"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Step 6: AI Recommendations
    st.markdown("#### Step 6: Generate AI Recommendations (CxO-Ready)")
    if st.button("Generate Consolidated AI Recommendations"):
        with st.spinner("Calling AI to generate executive summary and recommendations..."):
            recs = generate_ai_recommendations(assessment_name, dim_scores_ai, dim_scores_final, assessment_description)
        st.markdown("##### AI-Generated Narrative")
        if recs.startswith("ERROR:"):
            st.error(recs)
        else:
            st.write(recs)

    # Step 7: Export Data
    st.markdown("#### Step 7: Export Consolidated Data")
    st.write("Download the combined question-level dataset and aggregated views for offline analysis or archival.")

    csv_all = df_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Combined Question-Level CSV",
        data=csv_all,
        file_name="assessment_consolidated_question_level.csv",
        mime="text/csv"
    )

    csv_agg_q = agg_q.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Aggregated Question-Level CSV",
        data=csv_agg_q,
        file_name="assessment_aggregated_question_level.csv",
        mime="text/csv"
    )

    csv_agg_dim = agg_dim.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Aggregated Dimension-Level CSV",
        data=csv_agg_dim,
        file_name="assessment_aggregated_dimension_level.csv",
        mime="text/csv"
    )

    st.markdown(
        """
        <hr>
        <div style='text-align:center;color:gray;font-size:12px;'>
        SSC Universal Assessment Consolidation App • StrategyStack Consulting © 2025
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
