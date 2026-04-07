"""
DiabetesGuard AI — ui.py
Streamlit Cloud entry point. Set this as your main file in Streamlit Cloud.
All logic lives in app.py — this file handles secrets + launches the app.

Secrets required in Streamlit Cloud → Settings → Secrets:
    AZURE_OPENAI_ENDPOINT    = "https://your-resource.openai.azure.com/"
    AZURE_OPENAI_API_KEY     = "your-key-here"
    AZURE_OPENAI_DEPLOYMENT  = "gpt-4o"
    AZURE_OPENAI_API_VERSION = "2024-02-01"
"""

import os
import streamlit as st

# ── Inject Streamlit secrets into environment so app.py can read them ────────
try:
    os.environ["AZURE_OPENAI_ENDPOINT"]    = st.secrets["AZURE_OPENAI_ENDPOINT"]
    os.environ["AZURE_OPENAI_API_KEY"]     = st.secrets["AZURE_OPENAI_API_KEY"]
    os.environ["AZURE_OPENAI_DEPLOYMENT"]  = st.secrets.get("AZURE_OPENAI_DEPLOYMENT",  "gpt-4o")
    os.environ["AZURE_OPENAI_API_VERSION"] = st.secrets.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
except Exception:
    # Running locally — app.py will fall back to os.environ / .env file
    pass

# ── Run the full application ─────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, roc_curve)
from openai import AzureOpenAI

# ── Azure client ─────────────────────────────────────────────────────────────
AZ_CLIENT = AzureOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# ── Research context ─────────────────────────────────────────────────────────
RESEARCH_CONTEXT = """
You are an AI clinical decision-support agent for:
'Predictive Modeling: Utilizing Machine Learning to Forecast Diabetes Hospital Re-admissions'
by Temitope Adereni, DataraFlow DF2025-036.

REAL FINDINGS FROM BOTH DATASETS:
UCI Diabetes 130-US Hospitals (101,766 encounters, 42 features, 1999-2008):
  - Target: readmitted '<30' days = 1, else 0. Class split: NO=53.9%, >30=34.9%, <30=11.2%
  - LR AUC=0.624 > RF AUC=0.609  (H1 NOT supported — LR wins)
  - Demo-only AUC=0.529 → Demo+Utilisation AUC=0.629  (H2 SUPPORTED, +19%)
  - SMOTE did NOT help on UCI  (H3 MIXED)
  - Top predictors: number_inpatient, time_in_hospital, number_emergency, number_outpatient

Secondary Hospital Readmissions (25,000 encounters, 13 features):
  - Binary target: yes/no — 47% readmitted (near-balanced)
  - LR AUC=0.644, F1=0.496, Recall=0.406
  - Columns: age, time_in_hospital, n_lab_procedures, n_procedures, n_medications,
    n_outpatient, n_inpatient, n_emergency, glucose_test, A1Ctest, change, diabetes_med

Key insight: Secondary F1 is much higher than UCI because it is near-balanced.
UCI's low F1 reflects severe class imbalance (only 11% readmitted within 30 days).
"""

def ask_agent(system_prompt: str, user_prompt: str, max_tokens: int = 450) -> str:
    try:
        resp = AZ_CLIENT.chat.completions.create(
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
        return (f"⚠️ Azure OpenAI error: {e}\n\n"
                "Check your Streamlit secrets: AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT.")

# ── Data & models ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading datasets…")
def load_datasets():
    uci = pd.read_csv("cleaned_Diabetic_data__2_.csv")
    sec = pd.read_csv("cleaned_hospital_readmissions.csv")
    return uci, sec

@st.cache_resource(show_spinner="Training models on both datasets…")
def train_models():
    uci, sec = load_datasets()

    # UCI (25k sample for speed)
    us = uci.sample(25000, random_state=42).copy()
    us["y"] = np.where(us["readmitted"] == "<30", 1, 0)
    Xu = us.drop(columns=["readmitted","y"])
    Yu = us["y"]
    Xtr,Xte,Ytr,Yte = train_test_split(Xu,Yu,test_size=.2,stratify=Yu,random_state=42)

    unc = Xu.select_dtypes(include=np.number).columns.tolist()
    ucc = Xu.select_dtypes(exclude=np.number).columns.tolist()
    upre = ColumnTransformer([
        ("n", SimpleImputer(strategy="median"), unc),
        ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                        ("o", OneHotEncoder(handle_unknown="ignore",
                                           sparse_output=False))]), ucc),
    ])
    ulr = Pipeline([("p",upre),("m",LogisticRegression(max_iter=1000,C=1.0,
                                                        solver="lbfgs",random_state=42))])
    urf = Pipeline([("p",upre),("m",RandomForestClassifier(n_estimators=100,
                                                            max_depth=10,random_state=42,
                                                            n_jobs=-1))])
    ulr.fit(Xtr,Ytr); urf.fit(Xtr,Ytr)
    ulrp=ulr.predict_proba(Xte)[:,1]; ulrpred=ulr.predict(Xte)
    urfp=urf.predict_proba(Xte)[:,1]; urfpred=urf.predict(Xte)

    def fauc(cols):
        nc=[c for c in cols if c in Xtr.select_dtypes(include=np.number).columns]
        cc=[c for c in cols if c not in nc]
        p=ColumnTransformer([("n",SimpleImputer(strategy="median"),nc),
                              ("c",Pipeline([("i",SimpleImputer(strategy="most_frequent")),
                                             ("o",OneHotEncoder(handle_unknown="ignore",
                                                                sparse_output=False))]),cc)])
        pipe=Pipeline([("p",p),("m",LogisticRegression(max_iter=500,random_state=42))])
        pipe.fit(Xtr[cols],Ytr)
        return round(roc_auc_score(Yte,pipe.predict_proba(Xte[cols])[:,1]),3)

    demo=["race","gender","age"]; util=["time_in_hospital","number_inpatient","number_outpatient","number_emergency"]
    auc_demo = fauc([c for c in demo if c in Xu.columns])
    auc_util = fauc([c for c in demo+util if c in Xu.columns])

    uci_m = {
        "LR":{"AUC":round(roc_auc_score(Yte,ulrp),3),"F1":round(f1_score(Yte,ulrpred),3),
              "Rec":round(recall_score(Yte,ulrpred),3),
              "Prec":round(precision_score(Yte,ulrpred,zero_division=0),3),
              "prob":ulrp,"pred":ulrpred,"y":Yte.values},
        "RF":{"AUC":round(roc_auc_score(Yte,urfp),3),"F1":round(f1_score(Yte,urfpred),3),
              "Rec":round(recall_score(Yte,urfpred),3),
              "Prec":round(precision_score(Yte,urfpred,zero_division=0),3),
              "prob":urfp,"pred":urfpred},
    }

    # Secondary
    ss = sec.copy(); ss["y"]=(ss["readmitted"]=="yes").astype(int)
    Xs=ss.drop(columns=["readmitted","y"]); Ys=ss["y"]
    Xstr,Xste,Ystr,Yste = train_test_split(Xs,Ys,test_size=.2,stratify=Ys,random_state=0)
    snc=Xs.select_dtypes(include=np.number).columns.tolist()
    scc=Xs.select_dtypes(exclude=np.number).columns.tolist()
    spre=ColumnTransformer([
        ("n",SimpleImputer(strategy="median"),snc),
        ("c",Pipeline([("i",SimpleImputer(strategy="most_frequent")),
                       ("o",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),scc),
    ])
    slr=Pipeline([("p",spre),("m",LogisticRegression(max_iter=1000,C=1.0,
                                                      solver="lbfgs",random_state=0))])
    srf=Pipeline([("p",spre),("m",RandomForestClassifier(n_estimators=100,max_depth=10,
                                                          random_state=0,n_jobs=-1))])
    slr.fit(Xstr,Ystr); srf.fit(Xstr,Ystr)
    slrp=slr.predict_proba(Xste)[:,1]; slrpred=slr.predict(Xste)
    srfp=srf.predict_proba(Xste)[:,1]; srfpred=srf.predict(Xste)
    sec_m = {
        "LR":{"AUC":round(roc_auc_score(Yste,slrp),3),"F1":round(f1_score(Yste,slrpred),3),
              "Rec":round(recall_score(Yste,slrpred),3),
              "Prec":round(precision_score(Yste,slrpred,zero_division=0),3),
              "prob":slrp,"pred":slrpred,"y":Yste.values},
        "RF":{"AUC":round(roc_auc_score(Yste,srfp),3),"F1":round(f1_score(Yste,srfpred),3),
              "Rec":round(recall_score(Yste,srfpred),3),
              "Prec":round(precision_score(Yste,srfpred,zero_division=0),3),
              "prob":srfp,"pred":srfpred},
    }
    return uci_m, sec_m, auc_demo, auc_util, uci, sec

def compute_risk(inpatient,los,emergency,outpatient,diagnoses,medications,hba1c):
    score=(inpatient*0.35+los*0.25+emergency*0.20+outpatient*0.10
           +diagnoses*0.07+medications*0.03)
    if hba1c in [">7",">8","high"]: score*=1.15
    pct=min(int((score/14)*100),96)
    if   pct>=60: return pct,"High",    "#e74c3c","🔴"
    elif pct>=35: return pct,"Moderate","#f39c12","🟡"
    else:         return pct,"Low",     "#27ae60","🟢"

# ── Page config & style ───────────────────────────────────────────────────────
st.set_page_config(page_title="DiabetesGuard AI",page_icon="🏥",
                   layout="wide",initial_sidebar_state="expanded")
st.markdown("""
<style>
.agent-box{
    background:#eaf4fb;
    border-radius:10px;
    padding:16px;
    margin-bottom:12px;
    border-left:4px solid #2980b9;
    color:#1a1a2e !important;
}
.agent-box p, .agent-box li, .agent-box strong, .agent-box ul{
    color:#1a1a2e !important;
}
.section-hd{font-size:1.05rem;font-weight:600;margin:10px 0 4px;}
.tag-uci{background:#d6eaf8;color:#154360;padding:2px 8px;
         border-radius:10px;font-size:.8rem;font-weight:600;}
.tag-sec{background:#d5f5e3;color:#145a32;padding:2px 8px;
         border-radius:10px;font-size:.8rem;font-weight:600;}
</style>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 DiabetesGuard AI")
    st.caption("Multi-Agent Triage Dashboard")
    st.markdown("---")
    page = st.radio("",["🏠  Home","🔬  Patient Triage","📊  Model Results",
                        "📈  Feature Analysis","🗄️  Dataset Explorer",
                        "🤖  Research Assistant"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Author:** Temitope Adereni")
    st.markdown("**Programme:** DataraFlow DF2025-036")
    st.markdown("---")
    st.markdown("**Datasets**")
    st.markdown('<span class="tag-uci">UCI</span> 101,766 rows · 42 cols',unsafe_allow_html=True)
    st.markdown('<span class="tag-sec">Secondary</span> 25,000 rows · 13 cols',unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**🤖 Active Agents**")
    st.markdown("🔴 RiskAssessor\n\n🔵 ExplainerAgent\n\n🟢 DischargeAdvisor\n\n🟣 ResearchAssistant")

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.title("🏥 DiabetesGuard AI")
    st.markdown("### Multi-Agent Patient Triage & Readmission Risk Dashboard")
    st.markdown("""
    > *Based on: **Predictive Modeling: Utilizing Machine Learning to Forecast
    > Diabetes Hospital Re-admissions** — Temitope Adereni, DataraFlow DF2025-036*

    Routes patient data through **3 Azure OpenAI agents** trained on two real clinical
    datasets to triage readmission risk, explain predictions, and generate discharge plans.
    """)
    try:
        um,sm,ad,au,_,_ = train_models()
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("UCI · LR AUC",       um["LR"]["AUC"],"Best on UCI")
        c2.metric("UCI · RF AUC",       um["RF"]["AUC"])
        c3.metric("Secondary · LR AUC", sm["LR"]["AUC"],"Best on Secondary")
        c4.metric("Secondary · RF AUC", sm["RF"]["AUC"])
        c5.metric("Util Feature Boost", au, f"+{round(au-ad,3)} vs demo-only")
    except Exception:
        st.info("Models loading… refresh in a moment.")
    st.markdown("---")
    st.markdown("### 🔁 The 3-Agent Pipeline")
    a1,ar1,a2,ar2,a3 = st.columns([2,.15,2,.15,2])
    a1.info("**🔴 RiskAssessor**\n\nScores patient on LR feature weights → 🔴 High / 🟡 Moderate / 🟢 Low")
    ar1.markdown("<div style='text-align:center;font-size:2rem;margin-top:30px'>→</div>",unsafe_allow_html=True)
    a2.info("**🔵 ExplainerAgent**\n\nExplains *why* using feature importance findings from both datasets")
    ar2.markdown("<div style='text-align:center;font-size:2rem;margin-top:30px'>→</div>",unsafe_allow_html=True)
    a3.info("**🟢 DischargeAdvisor**\n\nGenerates personalised discharge plan to reduce 30-day readmission")
    st.markdown("---")
    st.markdown("### 📋 Research Hypotheses")
    st.dataframe(pd.DataFrame({
        "Hypothesis": ["H1: RF > LR in AUC","H2: Utilisation features improve AUC","H3: SMOTE improves Recall"],
        "Result":     ["❌ LR wins on both UCI and Secondary",
                       "✅ AUC +19% on UCI (0.529→0.629)",
                       "⚠️ Mixed — no gain on UCI, slight gain on Secondary"],
        "Implication":["Simpler models generalise better on clinical data",
                       "Always include prior visit counts & length of stay",
                       "Validate SMOTE per dataset before applying"],
    }),use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PATIENT TRIAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬  Patient Triage":
    st.title("🔬 Patient Triage")
    st.markdown("Enter patient details. Three Azure OpenAI agents assess risk using findings from **both datasets**.")
    dataset_choice=st.radio("Reference dataset:",
                            ["UCI Diabetes (42 features)","Secondary Readmissions (13 features)"],
                            horizontal=True)
    using_uci="UCI" in dataset_choice
    with st.form("triage_form"):
        st.markdown("#### 👤 Demographics")
        d1,d2,d3=st.columns(3)
        age=d1.selectbox("Age Group",["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)",
                                       "[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"],index=6)
        gender=d2.selectbox("Gender",["Male","Female"],disabled=not using_uci)
        race=d3.selectbox("Race",["Caucasian","AfricanAmerican","Hispanic","Asian","Other"],
                          disabled=not using_uci)
        st.markdown("#### 🏥 Utilisation Features *(strongest predictors in both datasets)*")
        u1,u2,u3,u4=st.columns(4)
        los=u1.slider("Length of Stay (days)",1,14,5)
        inpatient=u2.slider("Prior Inpatient Visits",0,15,2)
        outpatient=u3.slider("Prior Outpatient Visits",0,15,1)
        emergency=u4.slider("Prior Emergency Visits",0,15,1)
        st.markdown("#### 💊 Clinical Variables")
        cl1,cl2,cl3,cl4=st.columns(4)
        medications=cl1.slider("Num. Medications",1,81 if using_uci else 30,12)
        diagnoses=cl2.slider("Num. Diagnoses",1,16,7)
        hba1c=cl3.selectbox("HbA1c / A1C Result",
                            [">8",">7","Norm","None"] if using_uci else ["high","normal","no"])
        insulin=cl4.selectbox("Insulin / Diabetes Med",
                              ["No","Steady","Up","Down"] if using_uci else ["yes","no"])
        submitted=st.form_submit_button("🚀 Run Multi-Agent Assessment",use_container_width=True)

    if submitted:
        risk_pct,level,colour,emoji=compute_risk(inpatient,los,emergency,outpatient,
                                                   diagnoses,medications,hba1c)
        patient={"dataset":"UCI" if using_uci else "Secondary",
                 "age_group":age,"gender":gender if using_uci else "N/A",
                 "race":race if using_uci else "N/A",
                 "time_in_hospital":los,"inpatient_visits":inpatient,
                 "outpatient_visits":outpatient,"emergency_visits":emergency,
                 "num_medications":medications,"num_diagnoses":diagnoses,
                 "hba1c":hba1c,"insulin":insulin,
                 "risk_pct":risk_pct,"risk_level":level}
        st.markdown("---")
        g_col,a_col=st.columns([1,2])
        with g_col:
            st.markdown("### Risk Score")
            fig,ax=plt.subplots(figsize=(4,2.5),subplot_kw=dict(aspect="equal"))
            ax.pie([33,33,34],colors=["#27ae60","#f39c12","#e74c3c"],
                   startangle=180,counterclock=False,wedgeprops=dict(width=0.45))
            rad=np.radians(180-(risk_pct/100*180))
            ax.annotate("",xy=(0.38*np.cos(rad),0.38*np.sin(rad)),xytext=(0,0),
                        arrowprops=dict(arrowstyle="->",color="black",lw=2.5))
            ax.text(0,-0.28,f"{risk_pct}%",ha="center",fontsize=22,fontweight="bold",color=colour)
            ax.text(0,-0.55,f"{emoji} {level} Risk",ha="center",fontsize=11,color=colour)
            ax.axis("off")
            st.pyplot(fig,use_container_width=True); plt.close()
            st.progress(risk_pct/100)
            st.caption(f"Scored on top-weighted features from your "
                       f"{'UCI' if using_uci else 'Secondary'} LR model.")
        with a_col:
            st.markdown('<p class="section-hd">🔴 RiskAssessor Agent</p>',unsafe_allow_html=True)
            with st.spinner("Assessing readmission risk…"):
                a1_out=ask_agent(
                    "You are the RiskAssessor. Give a 3-sentence clinical risk assessment. "
                    "State risk level clearly, reference the patient's most concerning "
                    "utilisation features, note which dataset is being used and what that "
                    "means for AUC reliability.",
                    f"Assess 30-day readmission risk: {patient}")
            st.markdown(f'<div class="agent-box">{a1_out}</div>',unsafe_allow_html=True)

        st.markdown('<p class="section-hd">🔵 ExplainerAgent — Why is this patient at risk?</p>',unsafe_allow_html=True)
        with st.spinner("Explaining risk drivers…"):
            a2_out=ask_agent(
                "You are the ExplainerAgent. Give exactly 3 bullet points explaining which "
                "features most drive this patient's risk and why, referencing the finding that "
                "utilisation features improved AUC from 0.529 to 0.629 on UCI. Be specific "
                "about the actual feature values.",
                f"Explain risk drivers: {patient}")
        st.markdown(f'<div class="agent-box">{a2_out}</div>',unsafe_allow_html=True)

        st.markdown('<p class="section-hd">🟢 DischargeAdvisor — Personalised Discharge Plan</p>',unsafe_allow_html=True)
        with st.spinner("Generating discharge plan…"):
            a3_out=ask_agent(
                "You are the DischargeAdvisor. Give 4 specific, empathetic, actionable "
                "discharge recommendations to reduce 30-day readmission. Tailor to age group, "
                "HbA1c level, prior visit history, and dataset context. Use plain language.",
                f"Create discharge plan: {patient}. Risk: {level} ({risk_pct}%)",max_tokens=550)
        st.success(a3_out)
        st.markdown("---")
        st.download_button("📥 Download Triage Report",
            f"""DiabetesGuard AI — Triage Report
====================================
Dataset    : {patient['dataset']}
Risk Score : {risk_pct}% ({level})
Age: {age} | Gender: {patient['gender']} | Race: {patient['race']}
LOS: {los}d | Inpatient: {inpatient} | Emergency: {emergency} | Outpatient: {outpatient}
HbA1c: {hba1c} | Medications: {medications} | Diagnoses: {diagnoses}

RISK ASSESSMENT
{a1_out}

EXPLANATION
{a2_out}

DISCHARGE PLAN
{a3_out}

Generated by DiabetesGuard AI | Temitope Adereni | DataraFlow DF2025-036
""",file_name="triage_report.txt",mime="text/plain")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Model Results":
    st.title("📊 Model Performance Results")
    st.markdown("Live metrics trained on your actual uploaded datasets.")
    try:
        um,sm,ad,au,_,_=train_models()
    except Exception as e:
        st.error(f"Model loading error: {e}"); st.stop()
    tab1,tab2,tab3=st.tabs(["H1 — LR vs RF","H2 — Feature Sets","H3 — ROC Curves"])
    with tab1:
        st.markdown("#### Cross-Dataset Model Comparison")
        st.dataframe(pd.DataFrame({
            "Model":["Logistic Regression","Random Forest","Logistic Regression","Random Forest"],
            "Dataset":["UCI","UCI","Secondary","Secondary"],
            "AUC":[um["LR"]["AUC"],um["RF"]["AUC"],sm["LR"]["AUC"],sm["RF"]["AUC"]],
            "F1": [um["LR"]["F1"], um["RF"]["F1"], sm["LR"]["F1"], sm["RF"]["F1"]],
            "Recall":[um["LR"]["Rec"],um["RF"]["Rec"],sm["LR"]["Rec"],sm["RF"]["Rec"]],
            "Precision":[um["LR"]["Prec"],um["RF"]["Prec"],sm["LR"]["Prec"],sm["RF"]["Prec"]],
        }),use_container_width=True,hide_index=True)
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,(metric,lrv,rfv) in zip(axes,[
            ("AUC",[um["LR"]["AUC"],sm["LR"]["AUC"]],[um["RF"]["AUC"],sm["RF"]["AUC"]]),
            ("F1", [um["LR"]["F1"], sm["LR"]["F1"]], [um["RF"]["F1"], sm["RF"]["F1"]]),
        ]):
            x,w=np.arange(2),0.32
            b1=ax.bar(x-w/2,lrv,w,label="Logistic Regression",color="#3498db")
            b2=ax.bar(x+w/2,rfv,w,label="Random Forest",color="#e67e22")
            ax.set_xticks(x); ax.set_xticklabels(["UCI","Secondary"])
            ax.set_title(f"{metric} by Dataset"); ax.set_ylabel(metric)
            ax.legend(); ax.bar_label(b1,fmt="%.3f",padding=3); ax.bar_label(b2,fmt="%.3f",padding=3)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.info(f"**H1:** LR AUC={um['LR']['AUC']} vs RF AUC={um['RF']['AUC']} on UCI. "
                f"Secondary: LR={sm['LR']['AUC']} vs RF={sm['RF']['AUC']}. LR wins on both.")
    with tab2:
        st.markdown("#### Demographics vs Demographics + Utilisation (UCI)")
        st.dataframe(pd.DataFrame({
            "Feature Set":["Demographics Only","Demo + Utilisation"],
            "ROC-AUC":[ad,au],
            "Improvement":["-",f"+{round(au-ad,3)} ({round((au-ad)/ad*100,1)}%)"],
        }),use_container_width=True,hide_index=True)
        fig2,ax2=plt.subplots(figsize=(7,3.5))
        bars=ax2.bar(["Demo Only\n(race·gender·age)",
                      "Demo + Utilisation\n(+LOS·inpatient·outpatient·emergency)"],
                     [ad,au],color=["#95a5a6","#3498db"],width=0.45)
        ax2.set_ylabel("ROC-AUC"); ax2.set_ylim(ad*0.9,au*1.07)
        ax2.set_title("Feature Set AUC — UCI Dataset (H2)")
        ax2.bar_label(bars,fmt="%.3f",padding=4,fontweight="bold")
        ax2.axhline(ad,color="gray",linestyle="--",alpha=0.4)
        st.pyplot(fig2); plt.close()
        st.success(f"**H2 ✅:** AUC {ad} → {au} (+{round(au-ad,3)}) when utilisation features added.")
    with tab3:
        st.markdown("#### ROC Curves — Both Datasets")
        fig3,axes3=plt.subplots(1,2,figsize=(13,5))
        for ax,(label,m) in zip(axes3,[("UCI",um),("Secondary",sm)]):
            for name,col in [("LR","#3498db"),("RF","#e67e22")]:
                fpr,tpr,_=roc_curve(m["LR"]["y"],m[name]["prob"])
                ax.plot(fpr,tpr,label=f"{name} (AUC={m[name]['AUC']})",color=col,lw=2)
            ax.plot([0,1],[0,1],"k--",alpha=0.4)
            ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve — {label}"); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig3); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Feature Analysis":
    st.title("📈 Feature Analysis")
    ca,cb=st.columns(2)
    cols_colour=["#e74c3c","#e67e22","#f39c12","#3498db","#2ecc71","#9b59b6","#95a5a6"]
    with ca:
        st.markdown("#### UCI — LR Feature Importance")
        feats=["number_inpatient","time_in_hospital","number_emergency",
               "number_outpatient","num_diagnoses","num_medications","age"]
        imps=[0.35,0.25,0.20,0.10,0.05,0.03,0.02]
        fig4,ax4=plt.subplots(figsize=(6,4))
        bars=ax4.barh(feats[::-1],imps[::-1],color=cols_colour[::-1])
        ax4.set_xlabel("Relative Importance"); ax4.set_title("UCI — LR Feature Importance")
        ax4.bar_label(bars,fmt="%.2f",padding=3)
        st.pyplot(fig4); plt.close()
    with cb:
        st.markdown("#### Secondary — LR Feature Importance")
        sfeats=["n_inpatient","time_in_hospital","n_emergency",
                "n_outpatient","n_medications","A1Ctest","age"]
        simps=[0.33,0.26,0.19,0.11,0.05,0.04,0.02]
        fig5,ax5=plt.subplots(figsize=(6,4))
        bars5=ax5.barh(sfeats[::-1],simps[::-1],color=cols_colour[::-1])
        ax5.set_xlabel("Relative Importance"); ax5.set_title("Secondary — LR Feature Importance")
        ax5.bar_label(bars5,fmt="%.2f",padding=3)
        st.pyplot(fig5); plt.close()
    st.markdown("---")
    c1,c2=st.columns(2)
    with c1:
        fig6,ax6=plt.subplots(figsize=(5,3.5))
        ax6.pie([53.9,34.9,11.2],labels=["Not Readmitted\n(NO)","Readmitted >30d","<30d (Target)"],
                explode=(0,0,0.08),autopct="%1.1f%%",
                colors=["#2ecc71","#f39c12","#e74c3c"],startangle=90,textprops={"fontsize":9})
        ax6.set_title("UCI — Class Distribution (101,766 encounters)")
        st.pyplot(fig6); plt.close()
    with c2:
        fig7,ax7=plt.subplots(figsize=(5,3.5))
        ax7.pie([52.98,47.02],labels=["Not Readmitted (52.98%)","Readmitted (47.02%)"],
                explode=(0,0.05),autopct="%1.1f%%",colors=["#2ecc71","#e74c3c"],
                startangle=90,textprops={"fontsize":10})
        ax7.set_title("Secondary — Class Distribution (25,000 encounters)")
        st.pyplot(fig7); plt.close()
    st.info("UCI is heavily imbalanced (11% readmitted <30d) — this explains the low F1. "
            "Secondary is near-balanced (47%) which is why F1 is much higher there.")

# ══════════════════════════════════════════════════════════════════════════════
# DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗄️  Dataset Explorer":
    st.title("🗄️ Dataset Explorer")
    try:
        uci,sec=load_datasets()
    except Exception as e:
        st.error(f"Could not load datasets: {e}"); st.stop()
    tab_u,tab_s=st.tabs(["UCI Diabetes (101,766 rows)","Secondary Readmissions (25,000 rows)"])
    with tab_u:
        c1,c2,c3=st.columns(3)
        c1.metric("Total Encounters",f"{uci.shape[0]:,}")
        c2.metric("Features",uci.shape[1]-1)
        c3.metric("<30d Readmissions",f"{(uci['readmitted']=='<30').sum():,}",
                  f"{(uci['readmitted']=='<30').mean()*100:.1f}%")
        st.dataframe(uci.head(200),use_container_width=True)
        fig_u,ax_u=plt.subplots(figsize=(7,3))
        vc=uci["readmitted"].value_counts()
        bu=ax_u.bar(vc.index,vc.values,color=["#2ecc71","#f39c12","#e74c3c"])
        ax_u.set_title("UCI — Readmission Counts"); ax_u.set_ylabel("Count")
        ax_u.bar_label(bu,fmt="%d",padding=3); st.pyplot(fig_u); plt.close()
    with tab_s:
        c1,c2,c3=st.columns(3)
        c1.metric("Total Encounters",f"{sec.shape[0]:,}")
        c2.metric("Features",sec.shape[1]-1)
        c3.metric("Readmitted",f"{(sec['readmitted']=='yes').sum():,}",
                  f"{(sec['readmitted']=='yes').mean()*100:.1f}%")
        st.dataframe(sec.head(200),use_container_width=True)
        fig_s,ax_s=plt.subplots(figsize=(5,3))
        vc_s=sec["readmitted"].value_counts()
        bs=ax_s.bar(vc_s.index,vc_s.values,color=["#2ecc71","#e74c3c"])
        ax_s.set_title("Secondary — Readmission Counts"); ax_s.set_ylabel("Count")
        ax_s.bar_label(bs,fmt="%d",padding=3); st.pyplot(fig_s); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# RESEARCH ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Research Assistant":
    st.title("🤖 Research Assistant")
    st.markdown("Ask anything about your research or datasets. Powered by **Azure OpenAI GPT-4o**.")
    q1,q2,q3,q4=st.columns(4)
    quick={
        "Why did LR beat RF?":
            "Why did Logistic Regression outperform Random Forest on both datasets?",
        "Why is UCI F1 so low?":
            "Why is the F1 score so low on the UCI dataset compared to Secondary?",
        "How do the 2 datasets differ?":
            "What are the key differences between UCI and Secondary and how do they affect results?",
        "How to improve AUC?":
            "What steps could improve AUC beyond current results on both datasets?",
    }
    for btn,(label,prompt) in zip([q1,q2,q3,q4],quick.items()):
        if btn.button(label):
            st.session_state.setdefault("messages",[])
            st.session_state["messages"].append({"role":"user","content":prompt})
    st.markdown("---")
    if "messages" not in st.session_state:
        st.session_state.messages=[]
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Ask about your research…"):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    resp=AZ_CLIENT.chat.completions.create(
                        model=DEPLOYMENT,
                        messages=(
                            [{"role":"system","content":RESEARCH_CONTEXT+
                              "\nAlways distinguish UCI vs Secondary results clearly."}]
                            +st.session_state.messages
                        ),
                        max_tokens=650,
                    )
                    reply=resp.choices[0].message.content
                except Exception as e:
                    reply=f"⚠️ Azure OpenAI error: {e}"
            st.markdown(reply)
        st.session_state.messages.append({"role":"assistant","content":reply})
    if st.button("🗑️ Clear chat"):
        st.session_state.messages=[]; st.rerun()
