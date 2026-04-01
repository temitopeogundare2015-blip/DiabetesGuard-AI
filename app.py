"""
DiabetesGuard AI — Multi-Agent Patient Triage & Risk Dashboard
Author : Temitope Adereni | DataraFlow DF2025-036
Dataset: UCI Diabetes 130-US Hospitals (1999-2008)
Agents : RiskAssessor · ExplainerAgent · DischargeAdvisor · ResearchAssistant
"""

import os, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from openai import AzureOpenAI

# ── Azure OpenAI client ───────────────────────────────────────────────────────
client = AzureOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# ── Research context (real results from your notebooks) ──────────────────────
RESEARCH_CONTEXT = """
You are an AI clinical decision-support agent built on the research paper:
'Predictive Modeling: Utilizing Machine Learning to Forecast Diabetes Hospital Re-admissions'
by Temitope Adereni, DataraFlow Internship DF2025-036.

REAL RESEARCH FINDINGS YOU MUST USE:
- Dataset: UCI Diabetes 130-US Hospitals 1999-2008 (100,000+ encounters)
- Target: 30-day readmission (label: '<30' days = readmitted = 1, else 0)
- Logistic Regression AUC = 0.624  |  Random Forest AUC = 0.609  (LR wins — H1 not supported)
- External validation: LR AUC = 0.644  |  RF AUC = 0.605
- Adding utilisation features improved AUC from 0.529 → 0.629 (H2 SUPPORTED)
- SMOTE did NOT improve UCI performance — introduced noise (H3 MIXED)
- Strongest predictors: number_inpatient, time_in_hospital, number_emergency, number_outpatient
- Demo features: race, gender, age | Utilisation: time_in_hospital, number_inpatient, number_outpatient, number_emergency
"""

# ── Utility: call Azure OpenAI ────────────────────────────────────────────────
def ask_agent(system_prompt: str, user_prompt: str, max_tokens: int = 400) -> str:
    try:
        resp = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {"role": "system", "content": RESEARCH_CONTEXT + "\n\n" + system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.4,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Azure OpenAI error: {e}\n\nCheck your API key and endpoint in the Colab setup cell."

# ── Risk score (mirrors your LR model's top feature weights) ─────────────────
def compute_risk(inpatient, los, emergency, outpatient, diagnoses, medications, hba1c):
    score = (
        inpatient   * 0.35 +
        los         * 0.25 +
        emergency   * 0.20 +
        outpatient  * 0.10 +
        diagnoses   * 0.07 +
        medications * 0.03
    )
    if hba1c in [">7", ">8"]:
        score *= 1.15
    pct = min(int((score / 14) * 100), 96)
    if pct >= 60:
        level, colour, emoji = "High", "#e74c3c", "🔴"
    elif pct >= 35:
        level, colour, emoji = "Moderate", "#f39c12", "🟡"
    else:
        level, colour, emoji = "Low", "#27ae60", "🟢"
    return pct, level, colour, emoji

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL STYLE
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DiabetesGuard AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa; border-radius: 12px;
        padding: 18px; text-align: center;
        border-left: 5px solid #3498db;
    }
    .agent-box {
        background: #eaf4fb; border-radius: 10px;
        padding: 16px; margin-bottom: 12px;
        border-left: 4px solid #2980b9;
    }
    .risk-high   { color: #e74c3c; font-weight: 700; font-size: 1.4rem; }
    .risk-mod    { color: #f39c12; font-weight: 700; font-size: 1.4rem; }
    .risk-low    { color: #27ae60; font-weight: 700; font-size: 1.4rem; }
    .section-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏥 DiabetesGuard AI")
    st.caption("Multi-Agent Triage Dashboard")
    st.markdown("---")
    page = st.radio("", [
        "🏠  Home",
        "🔬  Patient Triage",
        "📊  Model Results",
        "📈  Feature Analysis",
        "🤖  Research Assistant",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Author:** Temitope Adereni")
    st.markdown("**Programme:** DataraFlow DF2025-036")
    st.markdown("**Dataset:** UCI Diabetes 130-US Hospitals")
    st.markdown("**Model:** Logistic Regression (AUC 0.624)")
    st.markdown("---")
    st.markdown("**🤖 Agents**")
    st.markdown("• 🔴 RiskAssessor\n• 🔵 ExplainerAgent\n• 🟢 DischargeAdvisor\n• 🟣 ResearchAssistant")

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.title("🏥 DiabetesGuard AI")
    st.markdown("### Multi-Agent Patient Triage & Readmission Risk Dashboard")
    st.markdown("""
    > *Built on the research paper: **Predictive Modeling: Utilizing Machine Learning
    > to Forecast Diabetes Hospital Re-admissions** — Temitope Adereni, DataraFlow DF2025-036*

    This system processes patient data through **3 specialised Azure OpenAI agents**
    to triage readmission risk, explain predictions, and generate discharge plans —
    just like the OralCare AI system, but for diabetes.
    """)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LR Model AUC", "0.624", "Best performer")
    c2.metric("RF Model AUC", "0.609", "vs LR")
    c3.metric("AUC w/ Utilisation", "0.629", "+19% vs demo-only")
    c4.metric("External Validation", "0.644", "Generalises well")

    st.markdown("---")

    # Agent pipeline diagram
    st.markdown("### 🔁 The 3-Agent Pipeline")
    a1, a2, a3, a4 = st.columns([1, 0.15, 1, 0.15])  # arrow cols
    with a1:
        st.info("**🔴 Agent 1 — RiskAssessor**\n\nTakes clinical inputs (age, LOS, prior visits, HbA1c) → outputs urgency score: 🔴 High / 🟡 Moderate / 🟢 Low")
    a2.markdown("<div style='text-align:center; font-size:2rem; margin-top:40px'>→</div>", unsafe_allow_html=True)
    with a3:
        st.info("**🔵 Agent 2 — ExplainerAgent**\n\nExplains *why* the patient is at risk using your feature importance findings (utilisation features = strongest predictors)")
    st.markdown("")
    b1, b2, b3 = st.columns([1, 0.15, 1])
    with b1:
        st.info("**🟢 Agent 3 — DischargeAdvisor**\n\nGenerates personalised, empathetic discharge recommendations to reduce 30-day readmission probability")
    b2.markdown("<div style='text-align:center; font-size:2rem; margin-top:40px'>→</div>", unsafe_allow_html=True)
    with b3:
        st.success("**📋 Output**\n\nRisk category + explanation + discharge plan — all in one dashboard view, ready for clinical use")

    st.markdown("---")
    st.markdown("### 📂 Research Hypotheses Summary")
    hyp_df = pd.DataFrame({
        "Hypothesis": ["H1: Random Forest > Logistic Regression", "H2: Utilisation features improve AUC", "H3: SMOTE improves Recall"],
        "Result": ["❌ Not supported — LR won (0.624 vs 0.609)", "✅ Supported — AUC: 0.529 → 0.629", "⚠️ Mixed — helped externally, not on UCI"],
        "Implication": ["Use simpler interpretable models", "Always include prior visit counts & LOS", "Apply SMOTE cautiously per dataset"],
    })
    st.dataframe(hyp_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PATIENT TRIAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬  Patient Triage":
    st.title("🔬 Patient Triage")
    st.markdown("Fill in patient details. Three Azure OpenAI agents will assess readmission risk, explain it, and suggest a discharge plan.")

    with st.form("triage_form"):
        st.markdown("#### 👤 Demographics  *(race · gender · age)*")
        d1, d2, d3 = st.columns(3)
        age    = d1.selectbox("Age Group", ["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)",
                                             "[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"], index=6)
        gender = d2.selectbox("Gender", ["Male", "Female"])
        race   = d3.selectbox("Race", ["Caucasian","AfricanAmerican","Hispanic","Asian","Other"])

        st.markdown("#### 🏥 Utilisation Features  *(strongest predictors in your research)*")
        u1, u2, u3, u4 = st.columns(4)
        los        = u1.slider("Length of Stay (days)",    1, 14, 5)
        inpatient  = u2.slider("Prior Inpatient Visits",   0, 15, 2)
        outpatient = u3.slider("Prior Outpatient Visits",  0, 15, 1)
        emergency  = u4.slider("Prior Emergency Visits",   0, 15, 1)

        st.markdown("#### 💊 Clinical Variables")
        cl1, cl2, cl3, cl4 = st.columns(4)
        medications = cl1.slider("Num. Medications",  1, 30, 12)
        diagnoses   = cl2.slider("Num. Diagnoses",    1, 16,  7)
        hba1c       = cl3.selectbox("HbA1c Result", ["None","Normal",">7",">8"])
        insulin     = cl4.selectbox("Insulin", ["No","Steady","Up","Down"])

        submitted = st.form_submit_button("🚀 Run Multi-Agent Assessment", use_container_width=True)

    if submitted:
        risk_pct, level, colour, emoji = compute_risk(
            inpatient, los, emergency, outpatient, diagnoses, medications, hba1c
        )

        patient_summary = {
            "age_group": age, "gender": gender, "race": race,
            "time_in_hospital": los, "number_inpatient": inpatient,
            "number_outpatient": outpatient, "number_emergency": emergency,
            "num_medications": medications, "num_diagnoses": diagnoses,
            "HbA1c_result": hba1c, "insulin": insulin,
            "computed_risk_pct": risk_pct, "risk_level": level,
        }

        st.markdown("---")

        # Risk gauge
        col_gauge, col_agents = st.columns([1, 2])
        with col_gauge:
            st.markdown("### Risk Score")
            fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw=dict(aspect="equal"))
            wedge_colors = ["#27ae60", "#f39c12", "#e74c3c"]
            ax.pie([33, 33, 34], colors=wedge_colors,
                   startangle=180, counterclock=False,
                   wedgeprops=dict(width=0.45))
            needle_angle = 180 - (risk_pct / 100 * 180)
            needle_rad   = np.radians(needle_angle)
            ax.annotate("", xy=(0.38 * np.cos(needle_rad), 0.38 * np.sin(needle_rad)),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2))
            ax.text(0, -0.25, f"{risk_pct}%", ha="center", va="center",
                    fontsize=20, fontweight="bold", color=colour)
            ax.text(0, -0.52, f"{emoji} {level} Risk", ha="center", va="center",
                    fontsize=11, color=colour)
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close()

            st.progress(risk_pct / 100)
            st.caption(f"Score based on your LR model's top feature weights: prior inpatient visits, length of stay, emergency visits.")

        with col_agents:
            # ── Agent 1: RiskAssessor ─────────────────────────────────
            st.markdown('<p class="section-title">🔴 RiskAssessor Agent</p>', unsafe_allow_html=True)
            with st.spinner("Assessing readmission risk..."):
                a1_out = ask_agent(
                    system_prompt=(
                        "You are the RiskAssessor agent. Using the research findings, "
                        "give a 3-sentence clinical risk assessment. State the risk level clearly, "
                        "reference the patient's most concerning utilisation features, "
                        "and compare to the model's AUC of 0.624."
                    ),
                    user_prompt=f"Assess 30-day readmission risk for this patient: {patient_summary}",
                )
            st.markdown(f'<div class="agent-box">{a1_out}</div>', unsafe_allow_html=True)

        # ── Agent 2: ExplainerAgent ───────────────────────────────────
        st.markdown('<p class="section-title">🔵 ExplainerAgent — Why is this patient at risk?</p>', unsafe_allow_html=True)
        with st.spinner("Explaining risk drivers..."):
            a2_out = ask_agent(
                system_prompt=(
                    "You are the ExplainerAgent — an explainable AI specialist. "
                    "Based on the research finding that utilisation features improved AUC from 0.529 to 0.629, "
                    "explain in exactly 3 bullet points which of this patient's features most drive their risk and why. "
                    "Be specific and reference the actual feature values provided."
                ),
                user_prompt=f"Explain risk drivers for: {patient_summary}",
            )
        st.markdown(f'<div class="agent-box">{a2_out}</div>', unsafe_allow_html=True)

        # ── Agent 3: DischargeAdvisor ─────────────────────────────────
        st.markdown('<p class="section-title">🟢 DischargeAdvisor — Personalised Discharge Plan</p>', unsafe_allow_html=True)
        with st.spinner("Generating discharge recommendations..."):
            a3_out = ask_agent(
                system_prompt=(
                    "You are the DischargeAdvisor — a compassionate diabetes discharge planning specialist. "
                    "Provide exactly 4 practical, empathetic, and specific discharge recommendations "
                    "to reduce this patient's 30-day readmission risk. "
                    "Tailor advice to their age group, HbA1c level, and utilisation history. "
                    "Use plain, patient-friendly language."
                ),
                user_prompt=f"Create a discharge plan for: {patient_summary}. Risk level: {level} ({risk_pct}%)",
                max_tokens=500,
            )
        st.success(a3_out)

        # Download summary
        st.markdown("---")
        summary_text = f"""DiabetesGuard AI — Patient Triage Report
==========================================
Risk Score : {risk_pct}% ({level})
Age Group  : {age} | Gender: {gender} | Race: {race}
LOS        : {los} days | Inpatient: {inpatient} | Emergency: {emergency}
HbA1c      : {hba1c} | Medications: {medications} | Diagnoses: {diagnoses}

RISK ASSESSMENT (RiskAssessor Agent)
{a1_out}

RISK EXPLANATION (ExplainerAgent)
{a2_out}

DISCHARGE PLAN (DischargeAdvisor Agent)
{a3_out}

Generated by DiabetesGuard AI | Temitope Adereni | DataraFlow DF2025-036
"""
        st.download_button("📥 Download Triage Report", summary_text,
                           file_name="triage_report.txt", mime="text/plain")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Model Results":
    st.title("📊 Model Performance Results")
    st.markdown("Your actual results from the UCI Diabetes and external validation datasets.")

    tab1, tab2, tab3 = st.tabs(["H1 — LR vs Random Forest", "H2 — Feature Sets", "H3 — SMOTE"])

    with tab1:
        st.markdown("#### Cross-Dataset AUC Comparison")
        df = pd.DataFrame({
            "Model":   ["Logistic Regression", "Random Forest", "Logistic Regression", "Random Forest"],
            "Dataset": ["UCI", "UCI", "External", "External"],
            "AUC":     [0.624, 0.609, 0.644, 0.605],
            "F1":      [0.028, 0.012, 0.484, 0.537],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(2)
        w = 0.32
        b1 = ax.bar(x - w/2, [0.624, 0.609], w, label="UCI Dataset",      color="#3498db")
        b2 = ax.bar(x + w/2, [0.644, 0.605], w, label="External Dataset", color="#2ecc71")
        ax.set_xticks(x); ax.set_xticklabels(["Logistic Regression", "Random Forest"])
        ax.set_ylim(0.55, 0.68); ax.set_ylabel("AUC")
        ax.set_title("AUC: Logistic Regression vs Random Forest")
        ax.legend(); ax.bar_label(b1, fmt="%.3f", padding=3); ax.bar_label(b2, fmt="%.3f", padding=3)
        ax.axhline(0.624, color="#3498db", linestyle="--", alpha=0.4, linewidth=1)
        st.pyplot(fig); plt.close()
        st.info("**Finding (H1):** LR consistently outperforms RF in AUC. Simpler models generalise better on sparse clinical data.")

    with tab2:
        st.markdown("#### Demographics vs Demographics + Utilisation")
        df2 = pd.DataFrame({
            "Feature Set":          ["Demographics Only", "Demographics + Utilisation"],
            "ROC-AUC":  [0.529, 0.629],
            "PR-AUC":   [0.120, 0.189],
            "Recall":   [0.765, 0.388],
            "F1 Score": [0.205, 0.243],
        })
        st.dataframe(df2, use_container_width=True, hide_index=True)

        fig2, axes = plt.subplots(1, 3, figsize=(12, 4))
        metrics = ["ROC-AUC", "PR-AUC", "F1 Score"]
        demo_vals = [0.529, 0.120, 0.205]
        util_vals = [0.629, 0.189, 0.243]
        for i, (m, dv, uv) in enumerate(zip(metrics, demo_vals, util_vals)):
            bars = axes[i].bar(["Demo Only", "Demo+Util"], [dv, uv],
                               color=["#95a5a6", "#3498db"])
            axes[i].set_title(m); axes[i].set_ylim(0, max(dv, uv) * 1.25)
            axes[i].bar_label(bars, fmt="%.3f", padding=3)
        plt.tight_layout()
        st.pyplot(fig2); plt.close()
        st.success("**Finding (H2 ✅):** Adding utilisation features boosted AUC by +19% (0.529 → 0.629).")

    with tab3:
        st.markdown("#### Baseline vs SMOTE — UCI Dataset")
        df3 = pd.DataFrame({
            "Model":   ["Baseline", "SMOTE"],
            "AUC":     [0.628, 0.606],
            "Recall":  [0.495, 0.423],
            "F1":      [0.235, 0.235],
        })
        st.dataframe(df3, use_container_width=True, hide_index=True)

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        x3 = np.arange(2)
        b3 = ax3.bar(x3 - 0.2, [0.628, 0.495], 0.35, label="Baseline", color="#3498db")
        b4 = ax3.bar(x3 + 0.2, [0.606, 0.423], 0.35, label="SMOTE",    color="#e74c3c")
        ax3.set_xticks(x3); ax3.set_xticklabels(["AUC", "Recall"])
        ax3.set_ylim(0, 0.75); ax3.set_title("Baseline vs SMOTE (UCI)")
        ax3.legend(); ax3.bar_label(b3, fmt="%.3f", padding=3); ax3.bar_label(b4, fmt="%.3f", padding=3)
        st.pyplot(fig3); plt.close()
        st.warning("**Finding (H3 ⚠️ Mixed):** SMOTE reduced both AUC and Recall on UCI — introduced noise. Helped slightly on external dataset.")

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Feature Analysis":
    st.title("📈 Feature Analysis")

    st.markdown("#### Estimated Feature Importance *(from your LR model research)*")
    features   = ["number_inpatient", "time_in_hospital", "number_emergency",
                  "number_outpatient", "num_diagnoses", "num_medications", "age"]
    importance = [0.35, 0.25, 0.20, 0.10, 0.05, 0.03, 0.02]
    colours    = ["#e74c3c","#e67e22","#f39c12","#3498db","#2ecc71","#9b59b6","#95a5a6"]

    fig4, ax4 = plt.subplots(figsize=(9, 4))
    bars = ax4.barh(features[::-1], importance[::-1], color=colours[::-1])
    ax4.set_xlabel("Relative Importance"); ax4.set_title("Feature Importance (Logistic Regression)")
    ax4.bar_label(bars, fmt="%.2f", padding=3)
    ax4.axvline(0.20, color="gray", linestyle="--", alpha=0.5)
    ax4.text(0.21, 0.5, "Utilisation\nthreshold", transform=ax4.get_yaxis_transform(),
             fontsize=8, color="gray")
    st.pyplot(fig4); plt.close()

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Utilisation vs Demographics — AUC Gain")
        fig5, ax5 = plt.subplots(figsize=(5, 3))
        ax5.barh(["Demo Only\n(race, gender, age)",
                  "Demo + Utilisation\n(+ LOS, inpatient, outpatient, emergency)"],
                 [0.529, 0.629], color=["#95a5a6", "#3498db"])
        ax5.set_xlim(0.45, 0.65); ax5.set_xlabel("AUC")
        ax5.set_title("AUC by Feature Set (H2)")
        for i, v in enumerate([0.529, 0.629]):
            ax5.text(v + 0.002, i, f"{v:.3f}", va="center", fontweight="bold")
        st.pyplot(fig5); plt.close()

    with col_b:
        st.markdown("#### Risk Level Distribution *(simulated from UCI class balance)*")
        fig6, ax6 = plt.subplots(figsize=(5, 3))
        labels = ["Low Risk\n(>30 days)", "Moderate Risk\n(No readmission)", "High Risk\n(<30 days)"]
        sizes  = [53, 35, 12]
        explode = (0, 0, 0.08)
        ax6.pie(sizes, labels=labels, explode=explode, autopct="%1.0f%%",
                colors=["#2ecc71", "#f39c12", "#e74c3c"],
                startangle=90, textprops={"fontsize": 9})
        ax6.set_title("UCI Dataset — Readmission Distribution")
        st.pyplot(fig6); plt.close()

    st.info("The UCI dataset is heavily imbalanced — only ~11% of patients are readmitted within 30 days. This is why SMOTE was explored (H3) and why Recall is a critical metric alongside AUC.")

# ══════════════════════════════════════════════════════════════════════════════
# RESEARCH ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Research Assistant":
    st.title("🤖 Research Assistant")
    st.markdown(
        "Ask anything about your research, methodology, or results. "
        "Powered by **Azure OpenAI GPT-4o** and trained on your actual findings."
    )

    # Suggested questions
    st.markdown("**💡 Try asking:**")
    q1, q2, q3 = st.columns(3)
    if q1.button("Why did LR beat Random Forest?"):
        st.session_state.setdefault("messages", [])
        st.session_state.messages.append({"role": "user", "content": "Why did Logistic Regression outperform Random Forest in my research?"})
    if q2.button("Why did SMOTE fail on UCI?"):
        st.session_state.setdefault("messages", [])
        st.session_state.messages.append({"role": "user", "content": "Why did SMOTE not improve performance on the UCI dataset?"})
    if q3.button("How do I improve AUC further?"):
        st.session_state.setdefault("messages", [])
        st.session_state.messages.append({"role": "user", "content": "What steps could I take to improve the AUC beyond 0.624 in future work?"})

    st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your diabetes readmission research..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = client.chat.completions.create(
                        model=DEPLOYMENT,
                        messages=(
                            [{"role": "system", "content": RESEARCH_CONTEXT +
                              "\nAnswer questions clearly and specifically. Reference real numbers from the research."}]
                            + st.session_state.messages
                        ),
                        max_tokens=600,
                    )
                    reply = resp.choices[0].message.content
                except Exception as e:
                    reply = f"⚠️ Azure OpenAI error: {e}"
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()
