# ======================================================================
# SSC UNIVERSAL ASSESSMENT APP (V8.1 - INDUSTRY + ROLE-AWARE SCORING)
# - Config-driven routing (JSON-based)
# - Client self-assessment only (no consultant tab)
# - Captures:
#       * Industry
#       * Respondent Role
# - "Submit Assessment" button:
#       -> Runs AI pre-scoring per dimension (industry + role-aware)
#       -> Builds Q&A record
#       -> Emails everything to StrategyStack (via SendGrid)
# - Optional industry_context.xlsx provides benchmarks per industry/dimension
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import os
import json
import time
import base64

# Try to import SendGrid (used for emailing results to SSC)
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import (
        Mail,
        Attachment,
        FileContent,
        FileName,
        FileType,
        Disposition,
    )
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False

# ===========================
# BASIC APP CONFIG
# ===========================
st.set_page_config(
    page_title="SSC Universal Assessment App (V8.1)",
    layout="wide"
)

# ===========================
# CONFIG-DRIVEN ROUTING
# ===========================

CONFIG_CANDIDATES = [
    "config/assessments_config.json",   # Recommended
    "assessments_config.json",          # Fallback
]

SSC_LOGO_PATH = "logos/ssc_logo.png"

FALLBACK_DEFAULT_TEMPLATES = [
    "ssc_assessment_template.xlsx",
    "templates/ssc_assessment_template_generic.xlsx",
]

INDUSTRY_CONTEXT_CANDIDATES = [
    "config/industry_context.xlsx",
    "industry_context.xlsx",
]


def load_config() -> dict:
    """Load JSON config for templates and clients."""
    for path in CONFIG_CANDIDATES:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                return cfg
            except Exception as e:
                st.sidebar.error(f"Failed to load config file '{path}': {e}")
                return {}
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
# INDUSTRY CONTEXT LOADING
# ===========================

def load_industry_context() -> pd.DataFrame | None:
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
    # Not critical – app can run without industry context
    return None


def get_industry_context_text(
    industry_name: str | None,
    dimension: str,
    ctx_df: pd.DataFrame | None
) -> str:
    """Build a human-readable industry context text for the AI prompt."""
    if ctx_df is None or not industry_name:
        return "No specific industry benchmarks provided. Use general best practices."

    try:
        subset = ctx_df[ctx_df["industry"].str.lower() == industry_name.lower()]
        if subset.empty:
            # Try any 'All' / blank industry rows for that dimension
            dim_rows = ctx_df[ctx_df["dimension"].str.lower() == dimension.lower()]
            if dim_rows.empty:
                return (
                    "No specific data found for this industry/dimension. "
                    "Use general P2P / process best practices."
                )
            rows = dim_rows
        else:
            # Filter by dimension if possible
            dim_rows = subset[subset["dimension"].str.lower() == dimension.lower()]
            rows = dim_rows if not dim_rows.empty else subset

        lines = []
        for _, r in rows.iterrows():
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
        if not lines:
            return (
                "Industry context could not be derived from the file. "
                "Use general best practices for this dimension."
            )
        return "\n".join(lines)
    except Exception:
        return (
            "Industry context could not be processed. "
            "Use general best practices and typical maturity patterns."
        )


# ===========================
# ROLE CONTEXT
# ===========================

def build_role_context(role: str) -> str:
    """
    Provide role-specific scoring expectations for the AI.
    This helps avoid penalising CFOs for not knowing low-level details,
    and avoids over-rewarding generic answers from operational roles.
    """
    if not role:
        return (
            "No specific role context provided. "
            "Apply standard expectations for a mid-senior process owner."
        )

    r_lower = role.lower()

    if "cfo" in r_lower or "finance director" in r_lower:
        return (
            "The respondent is a CFO / Finance Director. "
            "Expect more strategic, outcome-focused, governance-level answers. "
            "Do NOT penalise them for lack of technical or click-level detail. "
            "Score higher if they demonstrate clarity on controls, KPIs, risks, and overall oversight. "
            "Score lower only if the answer shows lack of awareness of major issues or complete absence of governance."
        )
    if "controller" in r_lower:
        return (
            "The respondent is a Financial Controller. "
            "Expect answers that link process execution, controls, month-end close, and financial integrity. "
            "They may not give full system detail, but should show awareness of key gaps and process health."
        )
    if "manager" in r_lower or "lead" in r_lower or "head" in r_lower:
        return (
            "The respondent is a Manager / Lead / Head for this area. "
            "Expect good understanding of end-to-end process, pain points, metrics, and team practices. "
            "Score lower if answers are overly generic and do not reflect day-to-day realities."
        )
    if "analyst" in r_lower or "specialist" in r_lower or "associate" in r_lower:
        return (
            "The respondent is an Analyst / Specialist. "
            "Expect more operational, step-by-step, and hands-on detail. "
            "If the answer is too high-level or generic, it may indicate limited understanding."
        )
    if "it" in r_lower or "erp" in r_lower or "technology" in r_lower:
        return (
            "The respondent is an IT / ERP / Technology role. "
            "Expect answers focusing on system capabilities, integrations, automation, controls, and technical constraints. "
            "Do NOT penalise them for missing pure business/commercial details."
        )

    return (
        "The respondent is in a business or support role. "
        "Expect answers aligned to their familiarity with the process. "
        "Avoid penalising them for details clearly outside their remit."
    )


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
# AI PRE-SCORING (INDUSTRY + ROLE-AWARE)
# ===========================

def ai_pre_score_responses(
    assessment_name: str,
    client_responses: dict,
    industry_name: str | None,
    respondent_role: str | None,
    industry_df: pd.DataFrame | None = None,
) -> dict:
    """
    Given:
      client_responses = {
        dimension: [
          {"index":..,"question":..,"answer":..,"weight":..,"help_text":..,"role":..},
          ...
        ],
        ...
      }

    Returns:
      ai_scores = {
        dimension: [
          {
            "index":..,
            "question":..,
            "answer":..,
            "ai_score": int,
            "reason": str,
            "weight":..,
            "help_text":..,
            "role":..
          },
          ...
        ]
      }
      or {"error": "..."} if something goes wrong.
    """
    overall_result = {}
    industry_name = industry_name or ""

    base_instructions = """
You will receive a structured object containing responses to a maturity assessment questionnaire
for ONE dimension (e.g., People, Process, Technology, SLA & Performance, Governance, etc.).

You must:
1) Assign a maturity score from 1 to 5 (1 = very poor/ad-hoc, 5 = world-class/fully optimised).
2) Provide a short reason (1–2 lines) explaining why you gave that score.

Use the following guidelines:
- If the answer describes heavy manual work, weak controls, lack of KPIs or repeated issues → Score 1–2.
- If partially standardised with some gaps or inconsistency → Score around 3.
- If well-documented, measured, stable and continuously improved → Score 4–5 depending on strength.
- If the answer is empty or non-informative, default to score 3 and reason "Insufficient detail; assumed mid-level maturity."

You will also be given:
- Industry benchmarks and context (where available),
- Respondent role and role-specific expectations.

Incorporate these into your scoring:
- Use industry benchmarks to calibrate what 'good' and 'average' look like.
- Use role expectations to avoid penalising people for details outside their remit.

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

    role_context = build_role_context(respondent_role or "")

    for dim, items in client_responses.items():
        if not items:
            overall_result[dim] = []
            continue

        industry_context_text = get_industry_context_text(industry_name, dim, industry_df)

        prompt = (
            base_instructions
            + f"\n\nAssessment Name: {assessment_name}\n"
            + f"Dimension: {dim}\n"
            + f"Industry: {industry_name or 'Not specified'}\n"
            + "Industry Benchmarks & Context:\n"
            + industry_context_text
            + "\n\nRespondent Role: "
            + (respondent_role or "Not specified")
            + "\nRole Context and Scoring Expectations:\n"
            + role_context
            + "\n\nHere is the client response data (as JSON):\n"
            + json.dumps(items, indent=2)
        )

        raw = call_openai_chat(prompt)
        if raw.startswith("ERROR:"):
            return {"error": f"{dim} scoring failed: {raw}"}

/*        try:
            cleaned = raw.strip().strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            parsed = json.loads(cleaned)

            # Merge back weight/help_text/role from original items by index
            enriched = []
            for item in parsed:
                idx = item.get("index")
                base_item = next((x for x in items if x.get("index") == idx), {})
                item["weight"] = base_item.get("weight", 1)
                item["help_text"] = base_item.get("help_text", "")
                item["role"] = base_item.get("role", respondent_role or "")
                enriched.append(item)
            overall_result[dim] = enriched 
*/
        try:
            cleaned = raw.strip().strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            parsed = json.loads(cleaned)

            # Merge back weight/help_text/role from original items by index
            enriched = []
            for item in parsed:
                idx = item.get("index")
                base_item = next((x for x in items if x.get("index") == idx), {})

                weight = base_item.get("weight", 1)
                ai_score = item.get("ai_score", None)

                item["weight"] = weight
                item["help_text"] = base_item.get("help_text", "")
                item["role"] = base_item.get("role", respondent_role or "")

                # --- NEW: compute final_score = ai_score * weight ---
                try:
                    if ai_score is not None:
                        item["final_score"] = float(ai_score) * float(weight)
                    else:
                        item["final_score"] = None
                except Exception:
                    # if anything weird, just keep it blank
                    item["final_score"] = None
                # ----------------------------------------------------

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
    if "industry_df" not in st.session_state:
        st.session_state["industry_df"] = None
    if "industry_name" not in st.session_state:
        st.session_state["industry_name"] = ""
    if "respondent_role" not in st.session_state:
        st.session_state["respondent_role"] = ""


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

        for col in ["dimension", "question"]:
            if col not in questions_df.columns:
                raise ValueError(f"Missing required column '{col}' in 'questions' sheet.")

        if "weight" not in questions_df.columns:
            questions_df["weight"] = 1
        if "help_text" not in questions_df.columns:
            questions_df["help_text"] = ""

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
    if uploaded_file is not None:
        return uploaded_file

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
    assessment_name,
    client_name,
    assessor_name,
    industry_name,
    respondent_role,
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
    if respondent_role:
        lines.append(f"Respondent Role : {respondent_role}")
    if industry_name:
        lines.append(f"Industry        : {industry_name}")
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
# EMAIL HELPERS – SENDGRID
# ===========================

def build_ai_scoring_csv(assessment_name, ai_result: dict) -> str:
    """
    Flatten ai_result into a CSV string.
    ai_result structure:
    {
      dimension: [
        {
          "index":..,
          "question":..,
          "answer":..,
          "ai_score":..,
          "reason":..,
          "weight":..,
          "help_text":..,
          "role":..
        }, ...
      ],
      ...
    }
    """
    if not ai_result:
        return ""

    rows = []
/*    for dim, items in ai_result.items():
        if not isinstance(items, list):
            continue
        for item in items:
            rows.append({
                "assessment_name": assessment_name,
                "dimension": dim,
                "index": item.get("index"),
                "question": item.get("question"),
                "answer": item.get("answer"),
                "ai_score": item.get("ai_score"),
                "reason": item.get("reason"),
                "weight": item.get("weight"),
                "help_text": item.get("help_text"),
                "role": item.get("role"),
            })
*/

    for dim, items in ai_result.items():
        if not isinstance(items, list):
            continue
        for item in items:
            rows.append({
                "assessment_name": assessment_name,
                "dimension": dim,
                "index": item.get("index"),
                "question": item.get("question"),
                "answer": item.get("answer"),
                "ai_score": item.get("ai_score"),
                # --- NEW: final_score column right after ai_score ---
                "final_score": item.get("final_score"),
                # ----------------------------------------------------
                "reason": item.get("reason"),
                "weight": item.get("weight"),
                "help_text": item.get("help_text"),
                "role": item.get("role"),
            })

    if not rows:
        return ""

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def send_assessment_to_ssc(
    assessment_name,
    client_name,
    assessor_name,
    industry_name,
    respondent_role,
    qa_text,
    ai_result=None,
    error_message=None,
):
    """
    Sends an email to SSC with:
    - Q&A text as attachment
    - AI scoring as CSV (if available)
    - Basic info in the email body
    Returns True if email sent successfully, False otherwise.
    """

    if not SENDGRID_AVAILABLE:
        st.error(
            "Internal configuration issue: email service is not available. "
            "Please contact SSC support."
        )
        return False

    try:
        sg_api_key = st.secrets["sendgrid"]["api_key"]
        to_email = st.secrets["sendgrid"]["to_email"]
        from_email = st.secrets["sendgrid"]["from_email"]
    except Exception:
        st.error(
            "Internal configuration error: email settings are missing. "
            "Please contact SSC support."
        )
        return False

    subject = f"[SSC Assessment] {assessment_name} – Submission from {client_name or 'Unknown Client'}"

    body_lines = [
        f"Assessment : {assessment_name}",
        f"Client     : {client_name or 'Not provided'}",
        f"Respondent : {assessor_name or 'Not provided'}",
        f"Role       : {respondent_role or 'Not provided'}",
        f"Industry   : {industry_name or 'Not provided'}",
        "",
        "This email was auto-generated by the SSC Universal Assessment App.",
    ]

    if error_message:
        body_lines.append("")
        body_lines.append("NOTE: AI pre-scoring could not be completed automatically.")
        body_lines.append(f"Error detail: {error_message}")

    body = "\n".join(body_lines)

    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject=subject,
        plain_text_content=body,
    )

    attachments = []

    qa_bytes = qa_text.encode("utf-8")
    qa_b64 = base64.b64encode(qa_bytes).decode()
    qa_attachment = Attachment(
        FileContent(qa_b64),
        FileName(f"{assessment_name.replace(' ', '_')}_QA.txt"),
        FileType("text/plain"),
        Disposition("attachment"),
    )
    attachments.append(qa_attachment)

    if ai_result and isinstance(ai_result, dict) and "error" not in ai_result:
        csv_str = build_ai_scoring_csv(assessment_name, ai_result)
        if csv_str:
            csv_bytes = csv_str.encode("utf-8")
            csv_b64 = base64.b64encode(csv_bytes).decode()
            csv_attachment = Attachment(
                FileContent(csv_b64),
                FileName(f"{assessment_name.replace(' ', '_')}_AI_Scoring.csv"),
                FileType("text/csv"),
                Disposition("attachment"),
            )
            attachments.append(csv_attachment)

    if attachments:
        message.attachment = attachments

    try:
        sg = SendGridAPIClient(sg_api_key)
        sg.send(message)
        return True
    except Exception as e:
        st.error(
            "Your responses were captured, but the system could not send them "
            f"to StrategyStack automatically. Please contact SSC support. (Email error: {e})"
        )
        return False


# ===========================
# UI SECTIONS
# ===========================

def render_sidebar(cfg: dict):
    st.sidebar.markdown("### SSC Universal Assessment")
    st.sidebar.markdown(
        "Use this app for any SSC diagnostic: CoE, IT, ERP, Pricing, Transformation, etc."
    )
    st.sidebar.markdown("---")

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
    st.sidebar.markdown("Weights for Overall Score (optional – SSC use)")

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
    industry_df = st.session_state.get("industry_df", None)

    st.subheader("Client / Self-Assessment")
    st.caption(
        "Please complete this questionnaire as accurately as possible. "
        "Once you click 'Submit Assessment', your responses will be securely "
        "shared with StrategyStack Consulting for review."
    )

    client_name = st.text_input("Client / Entity Name", value="")
    assessor_name = st.text_input("Name of person filling this assessment", value="")

    # Industry selection
    st.markdown("### Context")
    selected_industry = ""
    if industry_df is not None and not industry_df.empty and "industry" in industry_df.columns:
        industries = sorted(industry_df["industry"].dropna().unique().tolist())
        industries.append("Other / Not Listed")
        choice = st.selectbox("Industry", industries)
        if choice == "Other / Not Listed":
            custom = st.text_input("Please specify your industry", value="")
            selected_industry = custom.strip()
        else:
            selected_industry = choice
    else:
        selected_industry = st.text_input("Industry (free text)", value="")

    st.session_state["industry_name"] = selected_industry

    # Respondent role selection
    role_options = [
        "CFO / Finance Director",
        "Financial Controller",
        "P2P / AP Manager",
        "P2P / AP Analyst",
        "Procurement Head",
        "IT / ERP Lead",
        "Business Operations / Shared Services Lead",
        "Other",
    ]
    role_choice = st.selectbox("Your Role", role_options)
    if role_choice == "Other":
        role_text = st.text_input("Please specify your role", value="")
        respondent_role = role_text.strip()
    else:
        respondent_role = role_choice

    st.session_state["respondent_role"] = respondent_role

    st.markdown("---")

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
                    "role": respondent_role,
                }
            )
        responses[dim] = dim_items
        st.markdown("---")

    if st.button("Submit Assessment"):
        if not client_name or not assessor_name:
            st.error("Please fill in both 'Client / Entity Name' and 'Name of person filling this assessment' before submitting.")
            return

        st.session_state["client_responses"] = responses

        with st.spinner("Submitting your responses and running AI pre-scoring..."):
            ai_result = ai_pre_score_responses(
                assessment_name=assessment_name,
                client_responses=responses,
                industry_name=selected_industry,
                respondent_role=respondent_role,
                industry_df=industry_df,
            )

            qa_text = build_qa_document_text(
                assessment_name,
                client_name,
                assessor_name,
                selected_industry,
                respondent_role,
                responses
            )

            if "error" in ai_result:
                ok = send_assessment_to_ssc(
                    assessment_name=assessment_name,
                    client_name=client_name,
                    assessor_name=assessor_name,
                    industry_name=selected_industry,
                    respondent_role=respondent_role,
                    qa_text=qa_text,
                    ai_result=None,
                    error_message=ai_result["error"],
                )
                if ok:
                    st.warning(
                        "Thank you! Your responses have been submitted to StrategyStack. "
                        "However, AI scoring could not be completed automatically; "
                        "our consulting team will review manually."
                    )
            else:
                ok = send_assessment_to_ssc(
                    assessment_name=assessment_name,
                    client_name=client_name,
                    assessor_name=assessor_name,
                    industry_name=selected_industry,
                    respondent_role=respondent_role,
                    qa_text=qa_text,
                    ai_result=ai_result,
                    error_message=None,
                )
                if ok:
                    st.success(
                        "Thank you! Your responses have been submitted to StrategyStack. "
                        "Our consulting team will review the AI-scored results and "
                        "get back to you with insights."
                    )


# ===========================
# MAIN
# ===========================

def main():
    init_session_state()
    read_routing_params()

    cfg = load_config()
    st.session_state["config"] = cfg

    # Load industry context (optional)
    industry_df = load_industry_context()
    st.session_state["industry_df"] = industry_df

    logo_cols = st.columns([1, 4, 1])

    with logo_cols[0]:
        if os.path.exists(SSC_LOGO_PATH):
            st.image(SSC_LOGO_PATH, use_column_width=True)
        else:
            st.write("")

    client_key = st.session_state.get("client_key", "")
    client_logo_path = resolve_client_logo_path(cfg, client_key)
    client_display_name = resolve_client_display_name(cfg, client_key)

    with logo_cols[1]:
        title_html = (
            "<h1 style='text-align:center;color:#002060;font-family:Segoe UI, sans-serif;'>"
            "SSC Universal Assessment App (V8.1)</h1>"
        )
        subtitle_html = (
            "<h4 style='text-align:center;color:gray;font-family:Segoe UI, sans-serif;'>"
            "Client Self-Assessment • Industry & Role-Aware AI Scoring • Secure Submission</h4>"
        )
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(subtitle_html, unsafe_allow_html=True)
        if client_display_name and client_display_name != client_key:
            st.markdown(
                f"<p style='text-align:center;color:#555;font-style:italic;'>Client: {client_display_name}</p>",
                unsafe_allow_html=True,
            )

    with logo_cols[2]:
        if client_logo_path and os.path.exists(client_logo_path):
            st.image(client_logo_path, use_column_width=True)
        else:
            st.write("")

    st.markdown("---")

    render_sidebar(cfg)
    render_client_assessment()

    st.markdown(
        """
        <hr>
        <div style='text-align:center;color:gray;font-size:12px;'>
        SSC Universal Assessment App (V8.1) - StrategyStack Consulting 2025
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
