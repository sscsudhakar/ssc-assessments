# ======================================================================
# SSC UNIVERSAL ASSESSMENT APP (V7.1)
# - Config-driven routing (JSON-based)
# - Client self-assessment + AI pre-scoring
# - Consultant override + dashboard
# - Retry with backoff for OpenAI
# - User-friendly error messages
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import os
import json
import time

# ===========================
# BASIC APP CONFIG
# ===========================
st.set_page_config(
    page_title="SSC Universal Assessment App (V7.1)",
    layout="wide"
)

# ===========================
# CONFIG-DRIVEN ROUTING
# ===========================

# Where to look for the JSON config file (in your GitHub repo)
CONFIG_CANDIDATES = [
    "config/assessments_config.json",   # Recommended
    "assessments_config.json",          # Fallback
]

# SSC logo path – fixed (only the file content changes)
SSC_LOGO_PATH = "logos/ssc_logo.png"

# Fallback defaults if no config is found
FALLBACK_DEFAULT_TEMPLATES = [
    "ssc_assessment_template.xlsx",
    "templates/ssc_assessment_template_generic.xlsx",
]


def load_config() -> dict:
    """
    Load JSON config with structure:

    {
      "default_template_candidates": [...],
      "assessments": {
        "<template_key>": {
          "template_path": "templates/...xlsx",
          "client_key": "bain"
        },
        ...
      },
      "clients": {
        "<client_key>": {
          "display_name": "Bain Global Resources",
          "logo_path": "logos/bain_logo.png"
        },
        ...
      }
    }
    """
    for path in CONFIG_CANDIDATES:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                return cfg
            except Exception as e:
                st.sidebar.error(f"Failed to load config file '{path}': {e}")
                return {}
    # If nothing found
    st.sidebar.warning(
        "No config file found. Using fallback defaults only. "
        "Create 'config/assessments_config.json' to enable full routing."
    )
    return {}


def get_default_templates_from_config(cfg: dict):
    candidates = cfg.get("default_template_candidates")
    if isinstance(candidates, list) and candidates:
        return candidates
    return FALLBACK_DEFAULT_TEMPLATES


def resolve_template_path(cfg: dict, template_key: str):
    """Return the Excel path for a given template_key using config."""
    if not template_key:
        return None
    info = cfg.get("assessments", {}).get(template_key)
    if not info:
        return None
    return info.get("template_path")


def resolve_client_logo_path(cfg: dict, client_key: str):
    """Return logo path from config for a given client_key."""
    if not client_key:
        return None
    info = cfg.get("clients", {}).get(client_key)
    if not info:
        return None
    return info.get("logo_path")


def resolve_client_display_name(cfg: dict, client_key: str):
    """Return friendly display name for the client (if defined)."""
    info = cfg.get("clients", {}).get(client_key, {})
    return info.get("display_name", client_key or "")


# ===========================
# OPENAI HELPER (WITH RETRY + FRIENDLY ERRORS)
# ===========================

def call_openai_chat(prompt: str, max_retries: int = 3) -> str:
    """
    Call OpenAI Chat API with simple retry + backoff.
    Returns:
      - model output text on success
      - string starting with 'ERROR:' on failure (user-friendly message)
    """
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
                    "You are an experienced management consultant. "
                    "You specialise in maturity assessments across business, IT, CoEs, and transformations. "
                    "You MUST follow JSON formatting instructions exactly when requested."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.HTTPError as http_err:
            status = http_err.response.status_code if http_err.response is not None else None

            # Rate limit → retry with backoff
            if status == 429 and attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                st.warning(
                    f"AI rate limit reached (429). "
                    f"The system will retry automatically in {wait_time} seconds. Please wait…"
                )
                time.sleep(wait_time)
                continue

            # Other HTTP errors or final 429
            return (
                "ERROR: The AI service returned an error. "
                "Possible reasons:\n"
                "- Too many users running the scoring at the same time\n"
                "- API rate/quota limits reached for this account\n"
                "- Temporary overload or configuration issue\n\n"
                "Please retry after 1–2 minutes or contact SSC support."
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


# ===========================
# AI PRE-SCORING
# ===========================

def ai_pre_score_responses(assessment_name: str, client_responses: dict) -> dict:
    """
    Given:
      client_responses = {
        dimension: [
          {"index":..,"question":..,"answer":..,"weight":..,"help_text":..},
          ...
        ],
        ...
      }

    Return:
      ai_scores = {
        dimension: [
          {
            "index":..,
            "question":..,
            "answer":..,
            "ai_score": int,
            "reason": str,
            "weight":..,
            "help_text":..
          },
          ...
        ]
      }
      or {"error": "..."} if something goes wrong.
    """
    overall_result = {}

    base_instructions = """
You will receive a structured object containing responses to a maturity assessment questionnaire
for ONE dimension (e.g., People, Process, Technology, SLA & Performance, Governance, etc.).

For each answer, you must:
1) Assign a maturity score from 1 to 5 (1 = very poor / ad-hoc, 5 = world-class / fully optimised).
2) Provide a short reason (1–2 lines) explaining why you gave that score.

Important:
- Focus on what is actually described in the answer.
- If an answer indicates heavy manual workarounds, poor controls, lack of documentation, or recurring issues → Score 1–2.
- If partially standardised, with some gaps or inconsistency → Score 3.
- If well-documented, measured, mostly stable → Score 4–5 depending on how strong it sounds.
- If the answer is empty or non-informative, default to score 3 and reason "Insufficient detail; assumed mid-level maturity."

You MUST respond ONLY in valid JSON with this exact structure:

[
  {
    "index": <question_index_starting_from_1>,
    "question": "<question text>",
    "answer": "<client answer>",
    "ai_score": <integer 1-5>,
    "reason": "<short reason>"
  }
]

Do not include any other text outside the JSON array.
"""

    for dim, items in client_responses.items():
        if not items:
            overall_result[dim] = []
            continue

        prompt = (
            base_instructions
            + f"\n\nAssessment Name: {assessment_name}\n"
            + f"Dimension: {dim}\n\nHere is the client response data (as JSON):\n"
            + json.dumps(items, indent=2)
        )

        raw = call_openai_chat(prompt)
        if raw.startswith("ERROR:"):
            # Pass friendly error up to the UI
            return {"error": f"{dim} scoring failed: {raw}"}

        try:
            cleaned = raw.strip().strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            parsed = json.loads(cleaned)

            # Merge back weight/help_text from original items by index
            enriched = []
            for item in parsed:
                idx = item.get("index")
                base_item = next((x for x in items if x.get("index") == idx), {})
                item["weight"] = base_item.get("weight", 1)
                item["help_text"] = base_item.get("help_text", "")
                enriched.append(item)
            overall_result[dim] = enriched
        except Exception as e:
            return {
                "error": f"{dim} scoring failed: could not parse AI JSON: {e}\nRaw response was:\n{raw}"
            }

    return overall_result


# ===========================
# SESSION STATE HELPERS
# ===========================

def init_session_state():
    if "assessment_name" not in st.session_state:
        st.session_state["assessment_name"] = "SSC Assessment"
    if "assessment_description" not in st.session_state:
        st.session_state["assessment_description"] = ""
    if "dimensions" not in st.session_state:
        st.session_state["dimensions"] = []
    if "questions_df" not in st.session_state:
        st.session_state["questions_df"] = None
    if "client_responses" not in st.session_state:
        st.session_state["client_responses"] = {}
    if "ai_scoring" not in st.session_state:
        st.session_state["ai_scoring"] = None
    if "qa_document_text" not in st.session_state:
        st.session_state["qa_document_text"] = None
    if "dimension_weights" not in st.session_state:
        st.session_state["dimension_weights"] = {}
    if "template_key" not in st.session_state:
        st.session_state["template_key"] = ""
    if "client_key" not in st.session_state:
        st.session_state["client_key"] = ""
    if "config" not in st.session_state:
        st.session_state["config"] = {}


def read_routing_params():
    """Read ?template=...&client=... from the URL and store in session_state."""
    try:
        qp = st.experimental_get_query_params()
    except Exception:
        qp = {}

    template_key = qp.get("template", [""])[0].strip() if qp.get("template") else ""
    client_key = qp.get("client", [""])[0].strip() if qp.get("client") else ""

    if template_key:
        st.session_state["template_key"] = template_key
    if client_key:
        st.session_state["client_key"] = client_key


# ===========================
# LOADING ASSESSMENT TEMPLATE
# ===========================

def load_assessment_from_excel(file) -> tuple:
    """
    Expects an Excel file with:
      - sheet 'assessment_meta' (columns: field, value)
      - sheet 'questions' (columns: dimension, question, weight, help_text)
    Returns: (assessment_name, assessment_description, questions_df)
    """
    try:
        xls = pd.ExcelFile(file)
        meta = pd.read_excel(xls, sheet_name="assessment_meta")
        questions = pd.read_excel(xls, sheet_name="questions")

        meta_dict = {row["field"]: str(row["value"]) for _, row in meta.iterrows()}

        assessment_name = meta_dict.get("assessment_name", "SSC Assessment")
        assessment_description = meta_dict.get("description", "")
        questions_df = questions.copy()

        # Ensure required columns
        for col in ["dimension", "question"]:
            if col not in questions_df.columns:
                raise ValueError(f"Missing required column '{col}' in 'questions' sheet.")

        if "weight" not in questions_df.columns:
            questions_df["weight"] = 1
        if "help_text" not in questions_df.columns:
            questions_df["help_text"] = ""

        # Assign index per dimension for stable ordering
        questions_df["index"] = questions_df.groupby("dimension").cumcount() + 1

        return assessment_name, assessment_description, questions_df

    except Exception as e:
        st.error(f"Failed to load assessment template: {e}")
        return None, None, None


def select_template_file(uploaded_file, cfg: dict):
    """
    Decide which template to use, in this priority order:
    1) User-uploaded Excel (sidebar)
    2) Template from URL routing (?template=key) using config file
    3) Default generic template(s) from config or fallback list
    """
    # 1) Uploaded file always overrides everything
    if uploaded_file is not None:
        return uploaded_file

    # 2) Template key from URL
    template_key = st.session_state.get("template_key", "").strip()
    if template_key:
        mapped_path = resolve_template_path(cfg, template_key)
        if mapped_path and os.path.exists(mapped_path):
            return mapped_path
        else:
            st.sidebar.warning(
                f"Template key '{template_key}' was provided in URL, "
                "but the mapped file was not found. Falling back to default template."
            )

    # 3) Default candidates
    default_candidates = get_default_templates_from_config(cfg)
    for path in default_candidates:
        if os.path.exists(path):
            return path

    st.sidebar.error(
        "No assessment template available. "
        "Please upload a valid Excel template using the sidebar."
    )
    return None


# ===========================
# Q&A DOCUMENT BUILDER
# ===========================

def build_qa_document_text(
    assessment_name: str,
    client_name: str,
    assessor_name: str,
    responses: dict
) -> str:
    """Create a plain-text (Word-friendly) document with all questions and answers."""
    lines = []
    lines.append(f"{assessment_name} - Q&A Record")
    lines.append("=" * 60)
    lines.append("")
    if client_name:
        lines.append(f"Client / Entity : {client_name}")
    if assessor_name:
        lines.append(f"Respondent      : {assessor_name}")
    lines.append("")
    lines.append("Note: This document captures raw self-assessment inputs for reference and audit trail.")
    lines.append("")

    for dim in sorted(responses.keys()):
        dim_items = responses.get(dim, [])
        if not dim_items:
            continue
        lines.append("")
        lines.append(f"{dim} Dimension")
        lines.append("-" * (len(dim) + 11))
        lines.append("")
        for item in dim_items:
            idx = item.get("index")
            q = (item.get("question", "") or "").strip()
            a = (item.get("answer", "") or "").strip()
            help_text = (item.get("help_text", "") or "").strip()
            lines.append(f"Q{idx}. {q}")
            if help_text and help_text.lower() != "nan":
                lines.append(f"   [Hint: {help_text}]")
            if a:
                lines.append(f"   A: {a}")
            else:
                lines.append("   A: [No response provided]")
            lines.append("")

    return "\n".join(lines)


# ===========================
# UI SECTIONS
# ===========================

def render_sidebar(cfg: dict):
    st.sidebar.markdown("### SSC Universal Assessment")
    st.sidebar.markdown(
        "Use this app for any SSC diagnostic: CoE, IT, ERP, Pricing, Transformation, etc."
    )
    st.sidebar.markdown("---")

    # Show routing info
    template_key = st.session_state.get("template_key", "")
    client_key = st.session_state.get("client_key", "")
    if template_key or client_key:
        st.sidebar.markdown("**Routing Parameters Detected:**")
        if template_key:
            st.sidebar.write(f"- Template key: `{template_key}`")
        if client_key:
            st.sidebar.write(f"- Client key: `{client_key}`")
        st.sidebar.markdown("---")

    uploaded_template = st.sidebar.file_uploader(
        "Upload Assessment Template (.xlsx)", type=["xlsx"]
    )

    template_file = select_template_file(uploaded_template, cfg)
    if template_file is None:
        return

    assessment_name, assessment_description, questions_df = load_assessment_from_excel(template_file)
    if assessment_name is None or questions_df is None:
        return

    st.session_state["assessment_name"] = assessment_name
    st.session_state["assessment_description"] = assessment_description
    st.session_state["questions_df"] = questions_df
    st.session_state["dimensions"] = list(questions_df["dimension"].unique())

    st.sidebar.markdown(f"**Loaded Template:** {assessment_name}")
    if assessment_description:
        st.sidebar.caption(assessment_description)

    st.sidebar.markdown("---")
    st.sidebar.markdown("Weights for Overall Score (optional)")

    dims = st.session_state["dimensions"]
    weights = {}
    total_default = 0
    default_per_dim = int(100 / len(dims)) if dims else 0

    for dim in dims:
        w = st.sidebar.slider(f"{dim} Weight %", 0, 100, default_per_dim or 33, 1)
        weights[dim] = w
        total_default += w
    st.session_state["dimension_weights"] = weights
    if total_default != 100:
        st.sidebar.warning(f"Current total weight = {total_default}%. Consider adjusting to 100%.")


def render_client_assessment():
    assessment_name = st.session_state.get("assessment_name", "SSC Assessment")
    questions_df = st.session_state.get("questions_df", None)
    dimensions = st.session_state.get("dimensions", [])

    st.subheader("Client / Self-Assessment")
    st.caption(
        "This section is designed to be filled by the client or key stakeholders. "
        "No numeric scores are shown here – only descriptive inputs."
    )

    client_name = st.text_input("Client / Entity Name", value="")
    assessor_name = st.text_input("Name of person filling this assessment", value="")

    if questions_df is None or not dimensions:
        st.warning("No assessment template loaded. Please upload a valid Excel template in the sidebar.")
        return

    st.markdown(f"**Assessment:** {assessment_name}")
    st.markdown("---")

    responses = {}

    for dim in dimensions:
        st.markdown(f"### {dim}")
        dim_qs = questions_df[questions_df["dimension"] == dim].sort_values("index")
        dim_items = []
        for _, row in dim_qs.iterrows():
            idx = int(row["index"])
            q = str(row["question"])
            help_text = str(row.get("help_text", "") or "")
            weight = float(row.get("weight", 1) or 1)

            key = f"{dim}_q_{idx}"
            label = f"{dim} Q{idx}: {q}"
            if help_text and help_text.lower() != "nan":
                st.caption(f"Hint: {help_text}")
            answer = st.text_area(label, key=key, height=80)

            dim_items.append(
                {
                    "index": idx,
                    "question": q,
                    "answer": answer,
                    "weight": weight,
                    "help_text": help_text,
                }
            )
        responses[dim] = dim_items
        st.markdown("---")

    # Run AI pre-scoring
    if st.button("Run AI Pre-Scoring (Based on Answers)"):
        st.session_state["client_responses"] = responses
        with st.spinner("Calling AI to pre-score responses..."):
            ai_result = ai_pre_score_responses(assessment_name, responses)
        st.session_state["ai_scoring"] = ai_result
        if "error" in ai_result:
            st.error(ai_result["error"])
        else:
            st.success(
                "AI pre-scoring completed. Go to 'Consultant Review & Dashboard' tab "
                "to view and override scores."
            )

    # Q&A document creation and download
    st.markdown("### Download Q&A Record")
    if st.button("Prepare Q&A Document for Download"):
        st.session_state["client_responses"] = responses
        qa_text = build_qa_document_text(assessment_name, client_name, assessor_name, responses)
        st.session_state["qa_document_text"] = qa_text
        st.success("Q&A document prepared. Use the download button below.")

    qa_text = st.session_state.get("qa_document_text", None)
    if qa_text:
        safe_name = assessment_name.replace(" ", "_")
        file_name = f"{safe_name}_QA.txt"
        st.download_button(
            label="Download Q&A as Text File",
            data=qa_text.encode("utf-8"),
            file_name=file_name,
            mime="text/plain",
        )


def render_consultant_review_and_dashboard():
    assessment_name = st.session_state.get("assessment_name", "SSC Assessment")
    ai_scoring = st.session_state.get("ai_scoring", None)
    questions_df = st.session_state.get("questions_df", None)
    dimensions = st.session_state.get("dimensions", [])
    dim_weights = st.session_state.get("dimension_weights", {})

    st.subheader("Consultant Review & Dashboard")
    st.caption(
        "This section is for SSC consultants to review AI-pre-scored maturity, override scores, "
        "and generate the final dashboard. Use this primarily in single-stakeholder scenarios or "
        "as a pre-check before running multi-stakeholder consolidations."
    )

    if not ai_scoring or "error" in ai_scoring:
        st.warning("No valid AI scoring available. Please complete the Client Assessment and run AI pre-scoring first.")
        return

    if questions_df is None or not dimensions:
        st.warning("No assessment template loaded.")
        return

    dim_scores_ai = {}
    dim_scores_final = {}
    export_rows = []

    for dim in dimensions:
        st.markdown(f"### {dim} – Question-level Review")
        ai_items = ai_scoring.get(dim, [])
        rows = []

        if not isinstance(ai_items, list):
            st.error(f"AI scoring format for {dim} is invalid.")
            continue

        for item in ai_items:
            idx = item.get("index")
            question = item.get("question", "")
            answer = item.get("answer", "")
            ai_score = int(item.get("ai_score", 3))
            reason = item.get("reason", "")
            weight = item.get("weight", 1)
            help_text = item.get("help_text", "")

            override_key = f"{dim}_override_{idx}"
            default_value = ai_score
            final_score = st.selectbox(
                f"{dim} Q{idx} – Final Score (1–5)",
                options=[1, 2, 3, 4, 5],
                index=(default_value - 1) if 1 <= default_value <= 5 else 2,
                key=override_key,
            )

            st.write(f"**Question {idx}:** {question}")
            if help_text and str(help_text).lower() != "nan":
                st.caption(f"Hint: {help_text}")
            st.write(f"**Client Answer:** {answer if str(answer).strip() else '_No answer provided_'}")
            st.write(f"**AI Suggested Score:** {ai_score}  \n**AI Reason:** {reason}")
            st.write(f"**Consultant Final Score:** {final_score}")
            st.markdown("---")

            row = {
                "assessment_name": assessment_name,
                "dimension": dim,
                "index": idx,
                "question": question,
                "answer": answer,
                "ai_score": ai_score,
                "final_score": final_score,
                "reason": reason,
                "weight": weight,
                "help_text": help_text,
            }
            rows.append(row)
            export_rows.append(row)

        df_dim = pd.DataFrame(rows)
        if not df_dim.empty:
            dim_scores_ai[dim] = df_dim["ai_score"].mean()
            dim_scores_final[dim] = df_dim["final_score"].mean()
        else:
            dim_scores_ai[dim] = 0
            dim_scores_final[dim] = 0

    # Compute weighted overall score
    total_w = sum(dim_weights.get(d, 0) for d in dimensions)
    if total_w == 0:
        norm_weights = {d: 0 for d in dimensions}
    else:
        norm_weights = {d: dim_weights.get(d, 0) / total_w for d in dimensions}

    overall_weighted_final = sum(dim_scores_final[d] * norm_weights.get(d, 0) for d in dimensions)

    st.markdown("## Maturity Scores (AI vs Consultant Final)")
    cols = st.columns(len(dimensions) + 1)
    for i, dim in enumerate(dimensions):
        cols[i].metric(
            f"{dim} (Final)",
            f"{dim_scores_final[dim]:.2f} / 5",
            f"{dim_scores_ai[dim]:.2f} (AI)",
        )
    cols[-1].metric("Overall (Weighted Final)", f"{overall_weighted_final:.2f} / 5")

    # Radar chart
    st.markdown("### Radar View (Final Consultant Scores)")
    radar_df = pd.DataFrame(
        {
            "Dimension": dimensions,
            "Final Score": [dim_scores_final[d] for d in dimensions],
        }
    )
    fig_radar = px.line_polar(
        radar_df, r="Final Score", theta="Dimension", line_close=True, range_r=[0, 5]
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Bar chart AI vs Final
    st.markdown("### Bar Chart – Final Dimension Scores (AI vs Consultant)")
    bar_df = pd.DataFrame(
        {
            "Dimension": dimensions,
            "AI Score": [dim_scores_ai[d] for d in dimensions],
            "Final Score": [dim_scores_final[d] for d in dimensions],
        }
    )
    bar_df_melt = bar_df.melt(
        id_vars="Dimension",
        value_vars=["AI Score", "Final Score"],
        var_name="Type",
        value_name="Score",
    )
    fig_bar = px.bar(
        bar_df_melt,
        x="Dimension",
        y="Score",
        color="Type",
        barmode="group",
        range_y=[0, 5],
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Download option
    st.markdown("### Export Data")
    if export_rows:
        df_export = pd.DataFrame(export_rows)
        csv_data = df_export.to_csv(index=False).encode("utf-8")
        safe_name = assessment_name.replace(" ", "_")
        st.download_button(
            label="Download Question-Level Data as CSV",
            data=csv_data,
            file_name=f"{safe_name}_question_level.csv",
            mime="text/csv",
        )
    else:
        st.info("No question-level data available to export.")


# ===========================
# MAIN
# ===========================

def main():
    init_session_state()
    read_routing_params()

    # Load config once per run
    cfg = load_config()
    st.session_state["config"] = cfg

    # --- Top branding with logos ---
    logo_cols = st.columns([1, 4, 1])

    # Left: SSC logo
    with logo_cols[0]:
        if os.path.exists(SSC_LOGO_PATH):
            st.image(SSC_LOGO_PATH, use_column_width=True)
        else:
            st.write("")

    # Center: Title and client name
    client_key = st.session_state.get("client_key", "")
    client_logo_path = resolve_client_logo_path(cfg, client_key)
    client_display_name = resolve_client_display_name(cfg, client_key)

    with logo_cols[1]:
        title_html = (
            "<h1 style='text-align:center;color:#002060;font-family:Segoe UI, sans-serif;'>"
            "SSC Universal Assessment App (V7.1)</h1>"
        )
        subtitle_html = (
            "<h4 style='text-align:center;color:gray;font-family:Segoe UI, sans-serif;'>"
            "Client Self-Assessment + AI Pre-Scoring + Consultant Override + Config-Driven URL Routing</h4>"
        )
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(subtitle_html, unsafe_allow_html=True)
        if client_display_name and client_display_name != client_key:
            st.markdown(
                f"<p style='text-align:center;color:#555;font-style:italic;'>Client: {client_display_name}</p>",
                unsafe_allow_html=True,
            )

    # Right: client logo
    with logo_cols[2]:
        if client_logo_path and os.path.exists(client_logo_path):
            st.image(client_logo_path, use_column_width=True)
        else:
            st.write("")

    st.markdown("---")

    # Sidebar & content
    render_sidebar(cfg)

    tab_client, tab_consultant = st.tabs(
        ["Client Assessment", "Consultant Review & Dashboard"]
    )

    with tab_client:
        render_client_assessment()

    with tab_consultant:
        render_consultant_review_and_dashboard()

    # Footer
    st.markdown(
        """
        <hr>
        <div style='text-align:center;color:gray;font-size:12px;'>
        SSC Universal Assessment App (V7.1) - StrategyStack Consulting 2025
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
