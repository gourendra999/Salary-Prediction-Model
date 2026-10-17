from pathlib import Path
import html
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from scripts.salary_features import RAW_FEATURES, add_engineered_features, city_to_tier

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "salary_model.pkl"

st.set_page_config(
    page_title="Compensation Intelligence | HR Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { color-scheme: light; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background:#f5f7fb; color:#172b4d; }
[data-testid="stHeader"] { background:rgba(245,247,251,.94); }
[data-testid="stSidebar"] { background:#fff; border-right:1px solid #e4eaf2; }
[data-testid="stSidebar"] > div:first-child { padding-top:1rem; }
[data-testid="stSidebar"] * { color:#24364b; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#6b7a90; }
.block-container { padding-top:1.35rem; padding-bottom:3rem; max-width:1600px; }
.brand-lockup { display:flex; align-items:center; gap:.72rem; padding:.25rem .1rem 1rem; margin-bottom:1rem; border-bottom:1px solid #edf0f5; }
.brand-mark { width:42px; height:42px; border-radius:12px; display:flex; align-items:center; justify-content:center; background:#eaf1ff; color:#2458c6; font-family:'Manrope',sans-serif; font-size:1rem; font-weight:800; }
.brand-name { color:#182b49; font-family:'Manrope',sans-serif; font-weight:800; font-size:.96rem; line-height:1.2; }
.brand-sub { color:#8491a5; font-size:.7rem; margin-top:.2rem; letter-spacing:.06em; }
.sidebar-section { color:#91a0b3; font-size:.68rem; font-weight:800; text-transform:uppercase; letter-spacing:.12em; margin:.7rem 0 .5rem; }
.topline { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin:0 0 .45rem; color: #1F2937; }
.breadcrumb { color: #1F2937 !important; font-size:.74rem; font-weight:500; letter-spacing:.06em; text-transform:uppercase; }
.status-pill { display:inline-flex; align-items:center; gap:.4rem; padding:.36rem .65rem; color: #1F2937 !important; background: #F3F4F6 border: 1px solid #D1D5DB; border-radius:999px; font-size:.71rem; font-weight:700; white-space:nowrap; }
.status-dot { width:6px; height:6px; border-radius:50%; background:#19a566; display:inline-block; background-color:#16A34A;}
.page-title { color:#172b4d; font-family:'Manrope',sans-serif; font-weight:800; font-size:2.05rem; letter-spacing:-.055em; margin:.05rem 0 .2rem; line-height:1.2; }
.page-subtitle { color:#6f7f95; font-size:.92rem; margin:0 0 1.1rem; max-width:850px; line-height:1.55; }
.panel { background:#fff; border:1px solid #e4eaf2; border-radius:16px; padding:1.1rem 1.2rem; box-shadow:0 2px 8px rgba(22,42,75,.025); }
.panel-label { color:#7a889d; font-size:.68rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; }
.panel-title { color:#1d3557; font-family:'Manrope',sans-serif; font-weight:800; font-size:1.02rem; margin:.15rem 0 .3rem; }
.panel-description { color:#7b899c; font-size:.82rem; line-height:1.55; margin:0; }
.section-heading { color:#1c3354; font-family:'Manrope',sans-serif; font-size:1.12rem; font-weight:800; letter-spacing:-.025em; margin:.25rem 0 .2rem; }
.section-caption { color:#77859a; font-size:.81rem; margin:0 0 .8rem; }
.kpi-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.8rem; margin:.9rem 0 1.15rem; }
.kpi-card { background:#fff; border:1px solid #e4eaf2; border-radius:15px; padding:1rem 1.05rem; min-height:116px; overflow:hidden; box-shadow:0 2px 8px rgba(22,42,75,.025); }
.kpi-top { display:flex; justify-content:space-between; align-items:center; gap:.5rem; }
.kpi-label { color:#75849a; font-size:.75rem; font-weight:700; }
.kpi-icon { width:30px; height:30px; border-radius:9px; display:flex; align-items:center; justify-content:center; background:#eef4ff; color:#3267cc; font-size:.92rem; font-weight:800; }
.kpi-value { color:#18365f; font-family:'Manrope',sans-serif; font-size:1.55rem; font-weight:800; letter-spacing:-.045em; line-height:1.25; margin:.65rem 0 .25rem; overflow-wrap:anywhere; }
.kpi-foot { color:#8895a8; font-size:.71rem; }
.kpi-accent { height:3px; width:38px; background:#3c72d9; border-radius:99px; margin-top:.55rem; }
.salary-banner { background:linear-gradient(112deg,#fff 0%,#f5f8ff 100%); border:1px solid #dbe6f7; border-radius:18px; padding:1.2rem 1.4rem; box-shadow:0 3px 12px rgba(22,42,75,.03); display:flex; justify-content:space-between; align-items:center; gap:1rem; }
.salary-banner-label { color:#6f8098; font-size:.74rem; text-transform:uppercase; letter-spacing:.09em; font-weight:800; }
.salary-banner-value { color:#19478f; font-family:'Manrope',sans-serif; font-size:2.5rem; font-weight:800; letter-spacing:-.06em; line-height:1.25; margin:.18rem 0; }
.salary-banner-note { color:#74849b; font-size:.8rem; }
.salary-banner-icon { background:#e8f0ff; color:#2b61c8; width:56px; height:56px; flex:0 0 56px; border-radius:16px; display:flex; align-items:center; justify-content:center; font-size:1.5rem; }
.insight-card { background:#fff; border:1px solid #e4eaf2; border-radius:15px; padding:1rem 1.05rem; min-height:120px; box-shadow:0 2px 8px rgba(22,42,75,.02); }
.insight-icon { width:32px; height:32px; border-radius:10px; display:flex; align-items:center; justify-content:center; color:#2b61c8; background:#eef4ff; font-size:.95rem; margin-bottom:.7rem; }
.insight-title { color:#213956; font-weight:800; font-size:.9rem; margin:0 0 .28rem; }
.insight-text { color:#7a899d; font-size:.79rem; line-height:1.5; margin:0; }
.profile-chip { display:inline-block; padding:.36rem .58rem; margin:.2rem .18rem .1rem 0; background:#f2f5fa; color:#52647c; border:1px solid #e4eaf2; border-radius:8px; font-size:.72rem; font-weight:650; }
.stButton > button[kind="primary"] { background:#285fca; color:#ffffff; border:1px solid #285fca; border-radius:10px; min-height:2.75rem; font-weight:750; box-shadow:0 3px 8px rgba(40,95,202,.14); transition:all .15s ease; }
.stButton > button[kind="primary"]:hover { background:#1e4fae; border-color:#1e4fae; box-shadow:0 5px 12px rgba(40,95,202,.2); }
.stButton > button { border-radius:9px; }
.stTabs [data-baseweb="tab-list"] { gap:.35rem; border-bottom:1px solid #e2e8f1; }
.stTabs [data-baseweb="tab"] { height:2.6rem; color:#718097; font-weight:700; padding:0 .8rem; }
.stTabs [aria-selected="true"] { color:#245dc4 !important; border-bottom-color:#2f68d2 !important; }
div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label { color:#445770; font-size:.8rem; font-weight:700; }
[data-testid="stDataFrame"] { border:1px solid #e4eaf2; border-radius:12px; overflow:hidden; }
[data-testid="stDownloadButton"] button { border-radius:9px; font-weight:700; }
hr { border-color:#e5eaf2; margin:1.3rem 0; }
[data-testid="stCaptionContainer"] { color:#8491a4; }
.empty-state { background:#fff; border:1px dashed #cfd9e8; border-radius:18px; padding:2rem; text-align:center; }
.empty-icon { width:58px; height:58px; border-radius:17px; margin:0 auto .9rem; display:flex; align-items:center; justify-content:center; background:#eef4ff; color:#2d64ca; font-size:1.45rem; }
.empty-title { color:#1d3657; font-family:'Manrope',sans-serif; font-size:1.22rem; font-weight:800; }
.empty-copy { color:#79889e; font-size:.85rem; max-width:610px; margin:.4rem auto 0; line-height:1.6; }
@media (max-width: 900px) { .kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .page-title { font-size:1.7rem; } .salary-banner-value { font-size:2rem; } }
@media (max-width: 560px) { .kpi-grid { grid-template-columns:1fr; } .salary-banner { align-items:flex-start; } .salary-banner-value { font-size:1.8rem; } .topline { align-items:flex-start; flex-direction:column; } }
</style>
""", unsafe_allow_html=True)


EDUCATION_DEGREES = {
    "Bachelor's Degrees": [
        "B.Tech (Computer Science)", "B.Tech (Electronics)", "B.Tech (Mechanical)",
        "B.Tech (Civil)", "B.E. (Computer Engineering)", "B.Sc (Computer Science)",
        "B.Sc (Mathematics)", "B.Sc (Statistics)", "BCA (Computer Applications)",
        "BBA (Business Administration)", "B.Com (Commerce)", "B.A. (Economics)",
        "B.A. (English Literature)", "B.Pharm (Pharmacy)", "MBBS (Medicine)",
        "B.Arch (Architecture)", "BDS (Dental Surgery)", "B.Sc (Nursing)", "LLB (Law)",
    ],
    "Master's Degrees": [
        "M.Tech (Computer Science)", "M.Tech (Data Science)", "M.Tech (AI & ML)",
        "M.Tech (Electronics)", "M.E. (Software Engineering)", "M.Sc (Computer Science)",
        "M.Sc (Data Science)", "M.Sc (Statistics)", "MCA (Computer Applications)",
        "MBA (Finance)", "MBA (Marketing)", "MBA (HR & Operations)",
        "MBA (Business Analytics)", "M.Com (Commerce)", "M.A. (Economics)",
        "M.Pharm (Pharmacy)", "MDS (Dental Surgery)", "M.Sc (Biotechnology)",
        "LLM (Law)", "M.Arch (Architecture)",
    ],
    "PhD / Doctoral": [
        "PhD (Computer Science)", "PhD (Data Science & AI)", "PhD (Electrical Engineering)",
        "PhD (Mechanical Engineering)", "PhD (Business Management)", "PhD (Economics)",
        "PhD (Mathematics)", "PhD (Statistics)", "PhD (Biotechnology)", "PhD (Physics)",
    ],
}
EDU_LEVEL = {}
for degree in EDUCATION_DEGREES["Bachelor's Degrees"]: EDU_LEVEL[degree] = "Bachelor"
for degree in EDUCATION_DEGREES["Master's Degrees"]: EDU_LEVEL[degree] = "Master"
for degree in EDUCATION_DEGREES["PhD / Doctoral"]: EDU_LEVEL[degree] = "Phd"

SPECIALIZATIONS = {
    "B.Tech (Computer Science)": ["General","Artificial Intelligence & ML","Data Science","Cybersecurity","Cloud Computing","Full Stack Development","IoT & Embedded Systems","Blockchain Technology","Computer Vision"],
    "B.Tech (Electronics)": ["General","VLSI Design","Embedded Systems","Signal Processing","IoT & Automation","Robotics"],
    "B.Tech (Mechanical)": ["General","Robotics & Automation","Thermal Engineering","Manufacturing & Production","Automobile Engineering","CAD/CAM Design"],
    "B.Tech (Civil)": ["General","Structural Engineering","Environmental Engineering","Construction Management","Transportation Engineering"],
    "B.E. (Computer Engineering)": ["General","AI & Data Science","Cybersecurity","Cloud & DevOps","Software Engineering"],
    "B.Sc (Computer Science)": ["General","Data Science & AI","Cybersecurity","Cloud Computing","Full Stack Development","Game Development"],
    "B.Sc (Mathematics)": ["General","Data Science","Statistics & Analytics","Computational Mathematics","Actuarial Science"],
    "B.Sc (Statistics)": ["General","Data Analytics","Biostatistics","Actuarial Statistics","Business Analytics"],
    "BCA (Computer Applications)": ["General","Data Science & Artificial Intelligence","Cloud Computing","Cybersecurity","Mobile App Development","Full Stack Development","UI/UX Design"],
    "BBA (Business Administration)": ["General","Finance & Banking","Digital Marketing","HR Management","International Business","Business Analytics","Entrepreneurship"],
    "B.Com (Commerce)": ["General","Accounting & Finance","Taxation","Banking & Insurance","E-Commerce"],
    "B.A. (Economics)": ["General","Econometrics","International Economics","Development Economics","Financial Economics"],
    "B.A. (English Literature)": ["General","Content & Journalism","Communication & Media"],
    "B.Pharm (Pharmacy)": ["General","Clinical Research","Pharmaceutical Chemistry","Drug Regulatory Affairs"],
    "MBBS (Medicine)": ["General","Clinical Practice"],
    "B.Arch (Architecture)": ["General","Urban Planning","Interior Design","Sustainable Architecture"],
    "BDS (Dental Surgery)": ["General","Orthodontics","Prosthodontics"],
    "B.Sc (Nursing)": ["General","Critical Care Nursing","Community Health Nursing"],
    "LLB (Law)": ["General","Corporate Law","Criminal Law","Intellectual Property Law","Cyber Law"],
    "M.Tech (Computer Science)": ["General","Artificial Intelligence","Machine Learning","Data Science & Big Data","Cybersecurity","Cloud Computing","Computer Vision","NLP & Text Analytics","Distributed Systems"],
    "M.Tech (Data Science)": ["General","Deep Learning & AI","Business Analytics","Bioinformatics","Financial Data Science","Healthcare Analytics"],
    "M.Tech (AI & ML)": ["General","Computer Vision","NLP & Conversational AI","Reinforcement Learning","Generative AI","Robotics & Autonomous Systems"],
    "M.Tech (Electronics)": ["General","VLSI & Chip Design","Signal & Image Processing","Embedded AI"],
    "M.E. (Software Engineering)": ["General","DevOps & Cloud","Software Architecture","Agile & Quality Engineering"],
    "M.Sc (Computer Science)": ["General","AI & Data Science","Cybersecurity","Cloud & Infrastructure"],
    "M.Sc (Data Science)": ["General","Machine Learning Engineering","Statistical Modelling","Financial Analytics","Healthcare Data Science"],
    "M.Sc (Statistics)": ["General","Actuarial Science","Biostatistics","Econometrics","Data Analytics"],
    "MCA (Computer Applications)": ["General","AI & Machine Learning","Cloud Computing & DevOps","Cybersecurity","Full Stack Development","Mobile & App Development"],
    "MBA (Finance)": ["General","Investment Banking","Fintech & Digital Banking","Risk Management","Private Equity & Venture Capital","Corporate Finance","Wealth Management"],
    "MBA (Marketing)": ["General","Digital Marketing & Analytics","Brand Management","Product Management","Sales & CRM"],
    "MBA (HR & Operations)": ["General","HR Analytics","Talent Acquisition","Operations & Supply Chain"],
    "MBA (Business Analytics)": ["General","Data Science & AI","Financial Analytics","Marketing Analytics","Operations Analytics"],
    "M.Com (Commerce)": ["General","Accounting & Taxation","Financial Markets","E-Commerce"],
    "M.A. (Economics)": ["General","Econometrics & Forecasting","Development Economics","Financial Economics"],
    "M.Pharm (Pharmacy)": ["General","Clinical Pharmacology","Pharmaceutical Technology","Drug Regulatory Affairs"],
    "MDS (Dental Surgery)": ["General","Orthodontics & Dentofacial","Oral & Maxillofacial Surgery"],
    "M.Sc (Biotechnology)": ["General","Bioinformatics","Medical Biotechnology","Agricultural Biotechnology"],
    "LLM (Law)": ["General","Corporate & Commercial Law","International Law","Intellectual Property Law","Cyber & Technology Law"],
    "M.Arch (Architecture)": ["General","Urban Design & Planning","Sustainable Architecture","Interior Architecture"],
    "PhD (Computer Science)": ["General","AI & Deep Learning","Computer Vision","NLP & Language Models","Distributed Systems","Cybersecurity Research"],
    "PhD (Data Science & AI)": ["General","Generative AI & Foundation Models","Reinforcement Learning","Explainable AI (XAI)","Healthcare AI"],
    "PhD (Electrical Engineering)": ["General","Power Electronics","VLSI & Semiconductor","Signal Processing"],
    "PhD (Mechanical Engineering)": ["General","Robotics & Control Systems","Thermal & Fluid Sciences","Advanced Manufacturing"],
    "PhD (Business Management)": ["General","Strategic Management","Organisational Behaviour","Business Analytics"],
    "PhD (Economics)": ["General","Financial Economics","Behavioural Economics","Development Economics"],
    "PhD (Mathematics)": ["General","Computational Mathematics","Optimization & Operations Research","Applied Mathematics"],
    "PhD (Statistics)": ["General","Bayesian Statistics","Biostatistics","Computational Statistics"],
    "PhD (Biotechnology)": ["General","Genomics & Proteomics","Drug Discovery","Agricultural Biotech"],
    "PhD (Physics)": ["General","Quantum Computing","Condensed Matter Physics","Astrophysics & Cosmology"],
}
SECTORS_ROLES = {
    "Software Development": ["Software Developer","Full Stack Developer","Backend Developer","Frontend Developer","Mobile App Developer","iOS Developer","Android Developer","React Developer","Python Developer","Java Developer","Game Developer","Embedded Systems Developer"],
    "Data & AI": ["Data Analyst","Data Scientist","ML Engineer","AI Research Scientist","NLP Engineer","Computer Vision Engineer","Business Intelligence Analyst","Data Engineer","Big Data Engineer","Deep Learning Engineer","Quantitative Analyst"],
    "Cloud & Infrastructure": ["Cloud Engineer","DevOps Engineer","SRE (Site Reliability Engineer)","AWS Solutions Architect","Azure Cloud Architect","GCP Engineer","Kubernetes Engineer","Infrastructure Engineer","Platform Engineer"],
    "Cyber Security": ["Cyber Security Analyst","Security Engineer","Penetration Tester","SOC Analyst","Cloud Security Architect","Network Security Engineer","Information Security Manager","Forensics Analyst"],
    "IT Services": ["System Administrator","Technical Support Engineer","Network Administrator","IT Project Manager","ERP Consultant (SAP)","ERP Consultant (Oracle)","Scrum Master","IT Business Analyst","Technical Architect"],
    "Finance & Banking": ["Financial Analyst","Investment Banker","Risk Analyst","Credit Analyst","Portfolio Manager","Financial Controller","Wealth Manager","Compliance Officer","Fintech Product Manager","Actuary"],
    "Healthcare & Pharma": ["Medical Officer","Pharmaceutical Researcher","Clinical Data Analyst","Hospital Administrator","Medical Device Engineer","Regulatory Affairs Manager","Healthcare IT Consultant","Biomedical Engineer"],
    "E-Commerce & Retail": ["E-Commerce Manager","Category Manager","Supply Chain Manager","Growth Hacker","Product Manager","Logistics Manager","Inventory Analyst","Performance Marketing Manager"],
    "Consulting": ["Management Consultant","Strategy Consultant","IT Consultant","HR Consultant","Financial Consultant","Operations Consultant","Digital Transformation Consultant","Risk & Compliance Consultant"],
    "Marketing & Advertising": ["Digital Marketing Manager","SEO Specialist","Content Strategist","Brand Manager","Performance Marketing Analyst","Social Media Manager","Marketing Analytics Manager","Product Marketing Manager"],
    "Manufacturing & Automotive": ["Mechanical Engineer","Production Manager","Quality Assurance Engineer","Automation Engineer","Robotics Engineer","Supply Chain Analyst","R&D Engineer","Plant Manager"],
    "Telecom": ["Network Engineer","RF Engineer","Telecom Software Developer","5G Solutions Architect","VoIP Engineer","Telecom Project Manager"],
    "Education & EdTech": ["Data Science Instructor","Curriculum Developer","EdTech Product Manager","Academic Researcher","University Professor","E-Learning Developer"],
    "Media & Entertainment": ["UI/UX Designer","Graphic Designer","Video Game Designer","Content Producer","VFX Artist","Digital Media Manager"],
    "Energy & Utilities": ["Energy Analyst","Electrical Engineer","Power Systems Engineer","Renewable Energy Consultant","Environmental Engineer","Oil & Gas Engineer"],
}


COMPANY_SIZES = ["Startup", "Medium", "Enterprise", "MNC (Multinational)"]
CITY_TIERS = ["Tier 1 (Metro)", "Tier 2 (Mid-Sized)", "Tier 3 (Smaller Cities)"]
PLACEHOLDER = "-- Select --"


@st.cache_resource(show_spinner="Loading compensation model…")
def load_model(model_path: str):
    return joblib.load(model_path)


def make_plot_layout(fig, height=340):
    fig.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=10, r=12, t=20, b=10),
        paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="#ffffff",
        font=dict(family="DM Sans, sans-serif", color="#60718a", size=11),
        hoverlabel=dict(bgcolor="#172b4d", font_size=12, font_family="DM Sans, sans-serif", font_color="#ffffff"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#e7edf5", tickfont=dict(size=10), automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor="#edf1f6", zeroline=False, linecolor="#e7edf5", tickfont=dict(size=10), automargin=True)
    return fig


def make_kpi(label, value, foot, icon):
    return f'''<div class="kpi-card"><div class="kpi-top"><div class="kpi-label">{html.escape(label)}</div><div class="kpi-icon">{icon}</div></div><div class="kpi-value">{html.escape(str(value))}</div><div class="kpi-foot">{html.escape(foot)}</div><div class="kpi-accent"></div></div>'''


def profile_chips(profile):
    labels = [profile["Job Role"], profile["Job Sector"], profile["Education Degree"], profile["Company Size"], profile["City Tier"]]
    return "".join(f'<span class="profile-chip">{html.escape(str(x))}</span>' for x in labels)


st.markdown("""
<div class="topline">
<div class="breadcrumb">People analytics&nbsp;&nbsp; / &nbsp;&nbsp;Compensation intelligence</div>
<div class="status-pill"><span class="status-dot"></span> Predictive analytics workspace</div></div>
<div class="page-title">Compensation Intelligence</div>
<div class="page-subtitle">A focused workspace for salary estimation, compensation scenario planning, and career progression analysis.</div>
""", unsafe_allow_html=True)

MODEL_METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
metadata = {}
if MODEL_METADATA_PATH.exists():
    try:
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata = {}

with st.sidebar:
    st.markdown('''<div class="brand-lockup"><div class="brand-mark">CI</div><div><div class="brand-name">Compensation<br>Intelligence</div><div class="brand-sub">HR ANALYTICS SUITE</div></div></div>''', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Candidate inputs</div>', unsafe_allow_html=True)
    st.caption("Set a consistent candidate profile to run a salary scenario.")
    degree_group = st.selectbox("Education category", [PLACEHOLDER] + list(EDUCATION_DEGREES.keys()))
    degree = st.selectbox("Highest degree", [PLACEHOLDER] + EDUCATION_DEGREES.get(degree_group, []))
    specialization_options = SPECIALIZATIONS.get(degree, ["General"]) if degree != PLACEHOLDER else []
    specialization = st.selectbox("Specialization", [PLACEHOLDER] + specialization_options)
    experience = st.slider("Work experience", 0.0, 30.0, 0.0, 0.5, format="%.1f yrs")
    sector = st.selectbox("Job function / sector", [PLACEHOLDER] + list(SECTORS_ROLES.keys()))
    role = st.selectbox("Target job role", [PLACEHOLDER] + SECTORS_ROLES.get(sector, []))
    st.markdown('<div class="sidebar-section">Organization & market</div>', unsafe_allow_html=True)
    company_size = st.selectbox("Company size", [PLACEHOLDER] + COMPANY_SIZES)
    city_tier = st.selectbox("Work location tier", [PLACEHOLDER] + CITY_TIERS, help="Broad location groups, not a verified city-level market benchmark.")
    run_analysis = st.button("Run compensation analysis  →", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown('<div class="sidebar-section">Model information</div>', unsafe_allow_html=True)
    if metadata:
        st.caption(f"Engine · {metadata.get('selected_model', 'Trained model')}")
        if metadata.get("training_rows"):
            st.caption(f"Training records · {int(metadata['training_rows']):,}")
    else:
        st.caption("Model metadata will appear here after training.")
    st.caption("Decision support only. Do not use the estimate as the sole basis for hiring or compensation decisions.")

MODEL_PATH = BASE_DIR / "models" / "salary_model.pkl"
if not MODEL_PATH.exists():
    st.error("The trained model was not found at `models/salary_model.pkl`.")
    st.markdown("Run all cells in `pipeline.ipynb` using your salary dataset to train and save the model, then restart this app.")
    st.stop()
try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the salary model: {exc}")
    st.stop()

if run_analysis:
    required = [degree_group, degree, specialization, sector, role, company_size, city_tier]
    if any(value == PLACEHOLDER for value in required):
        st.session_state.pop("salary_analysis", None)
        st.warning("Complete each candidate input in the sidebar before running the analysis.")
    else:
        st.session_state["salary_analysis"] = {
            "Experience (Years)": float(experience),
            "Education Degree": degree,
            "Specialization": specialization,
            "Education Level": EDU_LEVEL[degree],
            "Job Sector": sector,
            "Job Role": role,
            "Company Size": company_size,
            "City Tier": city_tier,
        }
profile = st.session_state.get("salary_analysis")

if not profile:
    overview_tab, scenarios_tab, notes_tab = st.tabs(["Overview", "Scenario analysis", "Model notes"])
    with overview_tab:
        st.markdown('''<div class="empty-state"><div class="empty-icon">⌁</div><div class="empty-title">Your compensation workspace is ready</div><div class="empty-copy">Configure a candidate profile in the sidebar. The dashboard will return a model-based annual salary estimate, monthly equivalent, and scenario charts for experience, company size, and work location.</div></div>''', unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="section-heading">Analysis toolkit</div><div class="section-caption">Designed to support structured compensation conversations.</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            st.markdown('<div class="insight-card"><div class="insight-icon">₹</div><div class="insight-title">Salary estimation</div><p class="insight-text">Generate an annual estimate and monthly equivalent for the selected role profile.</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="insight-card"><div class="insight-icon">↗</div><div class="insight-title">Career progression</div><p class="insight-text">Review how the fitted model responds as years of experience change.</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="insight-card"><div class="insight-icon">▤</div><div class="insight-title">What-if scenarios</div><p class="insight-text">Compare alternative employer-size and location-tier assumptions side by side.</p></div>', unsafe_allow_html=True)
    with scenarios_tab:
        st.info("Run a compensation analysis from the sidebar to populate scenario charts.")
    with notes_tab:
        st.markdown('<div class="panel"><div class="panel-label">Methodology</div><div class="panel-title">Model-based estimates, not market survey statistics</div><p class="panel-description">Scenario charts change one input at a time and keep other attributes fixed. Model quality depends on the coverage, freshness, and performance of its training data.</p></div>', unsafe_allow_html=True)
else:
    base = pd.DataFrame([profile])
    base_features = add_engineered_features(base)
    prediction = max(0.0, float(model.predict(base_features)[0]))
    annual_lpa = prediction / 100000.0
    monthly = prediction / 12.0
    career_stage = str(base_features.iloc[0]["Career Stage"])

    # Batch predictions together rather than repeatedly invoking the model for each chart.
    experience_values = sorted(set([0, 1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30, float(profile["Experience (Years)"])]))
    exp_rows = []
    for years in experience_values:
        row = dict(profile)
        row["Experience (Years)"] = float(years)
        exp_rows.append(row)
    company_rows, city_rows = [], []
    for size in COMPANY_SIZES:
        row = dict(profile); row["Company Size"] = size; company_rows.append(row)
    for tier in CITY_TIERS:
        row = dict(profile); row["City Tier"] = tier; city_rows.append(row)
    scenario_features = add_engineered_features(pd.DataFrame(exp_rows + company_rows + city_rows))
    scenario_predictions = np.maximum(0.0, np.asarray(model.predict(scenario_features), dtype=float)) / 100000.0
    n_exp, n_company = len(exp_rows), len(company_rows)

    exp_chart = pd.DataFrame({"Experience (years)": experience_values, "Modeled salary (₹ LPA)": scenario_predictions[:n_exp]}).sort_values("Experience (years)")
    company_chart = pd.DataFrame({"Company size": COMPANY_SIZES, "Modeled salary (₹ LPA)": scenario_predictions[n_exp:n_exp+n_company]})
    company_chart["Scenario"] = np.where(company_chart["Company size"] == profile["Company Size"], "Selected profile", "Alternative")
    city_chart = pd.DataFrame({"Work location tier": CITY_TIERS, "Modeled salary (₹ LPA)": scenario_predictions[n_exp+n_company:]})
    city_chart["Scenario"] = np.where(city_chart["Work location tier"] == profile["City Tier"], "Selected profile", "Alternative")
    company_chart["Difference vs profile (₹ LPA)"] = company_chart["Modeled salary (₹ LPA)"] - annual_lpa
    city_chart["Difference vs profile (₹ LPA)"] = city_chart["Modeled salary (₹ LPA)"] - annual_lpa

    st.markdown(f'''<div class="panel" style="padding:.8rem 1rem;margin-bottom:.85rem;"><div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap;"><div><div class="panel-label">Active candidate scenario</div><div style="color:#1d3557;font-weight:800;font-size:1rem;margin-top:.25rem;">{html.escape(profile['Job Role'])}</div></div><div>{profile_chips(profile)}</div></div></div>''', unsafe_allow_html=True)
    overview_tab, scenarios_tab, candidate_tab, notes_tab = st.tabs(["Overview", "Scenario analysis", "Candidate profile", "Model notes"])

    with overview_tab:
        st.markdown('<div class="section-heading">Compensation snapshot</div><div class="section-caption">Estimated from the selected profile and the currently loaded model.</div>', unsafe_allow_html=True)
        st.markdown(f'''<div class="salary-banner"><div><div class="salary-banner-label">Model-estimated annual compensation</div><div class="salary-banner-value">₹ {annual_lpa:,.2f} LPA</div><div class="salary-banner-note">Monthly equivalent ₹ {monthly:,.0f} · Not a confidence interval or guaranteed offer.</div></div><div class="salary-banner-icon">₹</div></div>''', unsafe_allow_html=True)
        st.markdown(f'''<div class="kpi-grid">{make_kpi('Annual salary', f'₹{annual_lpa:,.2f}L', 'Model estimate in INR', '₹')}{make_kpi('Monthly equivalent', f'₹{monthly/1000:,.1f}K', 'Annual estimate divided by 12', '↗')}{make_kpi('Experience', f"{profile['Experience (Years)']:g} yrs", 'Work experience supplied', '◷')}{make_kpi('Career stage', career_stage, 'Engineered profile segment', '◎')}</div>''', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Salary outlook</div><div class="section-caption">Model response to changing experience while other profile attributes stay constant.</div>', unsafe_allow_html=True)
        fig_exp = px.line(exp_chart, x="Experience (years)", y="Modeled salary (₹ LPA)", markers=True)
        fig_exp.update_traces(line=dict(color="#2f68d2", width=3), marker=dict(size=7, color="#2f68d2", line=dict(color="#ffffff", width=2)), hovertemplate="Experience: %{x} years<br>Modeled salary: ₹%{y:.2f} LPA<extra></extra>")
        fig_exp.add_vline(x=float(profile["Experience (Years)"]), line_width=1.5, line_dash="dash", line_color="#7d8ca3")
        fig_exp.add_annotation(x=float(profile["Experience (Years)"]), y=1, yref="paper", text="Selected experience", showarrow=False, yshift=12, font=dict(size=10, color="#728198"), bgcolor="rgba(255,255,255,.9)")
        fig_exp.update_yaxes(title=None, tickprefix="₹", ticksuffix="L", rangemode="tozero")
        fig_exp.update_xaxes(title="Years of experience", dtick=5)
        make_plot_layout(fig_exp, 360)
        st.plotly_chart(fig_exp, use_container_width=True, config={"displaylogo": False, "responsive": True})

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown('<div class="section-heading">Employer-size scenario</div><div class="section-caption">Current employer type is highlighted.</div>', unsafe_allow_html=True)
            fig = px.bar(company_chart, x="Modeled salary (₹ LPA)", y="Company size", orientation="h", color="Scenario", color_discrete_map={"Selected profile":"#285fca", "Alternative":"#bac9e6"}, text="Modeled salary (₹ LPA)")
            fig.update_traces(texttemplate="₹%{text:.2f}L", textposition="outside", cliponaxis=False, hovertemplate="%{y}<br>Modeled salary: ₹%{x:.2f} LPA<extra></extra>")
            fig.update_xaxes(title=None, tickprefix="₹", ticksuffix="L", rangemode="tozero"); fig.update_yaxes(title=None, autorange="reversed"); fig.update_layout(showlegend=False)
            make_plot_layout(fig, 285); st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False,"responsive":True})
        with right:
            st.markdown('<div class="section-heading">Location-tier scenario</div><div class="section-caption">Current work location tier is highlighted.</div>', unsafe_allow_html=True)
            fig = px.bar(city_chart, x="Modeled salary (₹ LPA)", y="Work location tier", orientation="h", color="Scenario", color_discrete_map={"Selected profile":"#285fca", "Alternative":"#bac9e6"}, text="Modeled salary (₹ LPA)")
            fig.update_traces(texttemplate="₹%{text:.2f}L", textposition="outside", cliponaxis=False, hovertemplate="%{y}<br>Modeled salary: ₹%{x:.2f} LPA<extra></extra>")
            fig.update_xaxes(title=None, tickprefix="₹", ticksuffix="L", rangemode="tozero"); fig.update_yaxes(title=None, autorange="reversed"); fig.update_layout(showlegend=False)
            make_plot_layout(fig, 285); st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False,"responsive":True})

    with scenarios_tab:
        st.markdown('<div class="section-heading">Compensation scenario lab</div><div class="section-caption">Change one profile variable at a time; all other attributes remain fixed.</div>', unsafe_allow_html=True)
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Salary trend by experience**")
            fig = px.line(exp_chart, x="Experience (years)", y="Modeled salary (₹ LPA)", markers=True)
            fig.update_traces(line=dict(color="#285fca", width=3), marker=dict(size=7), hovertemplate="%{x} years<br>₹%{y:.2f} LPA<extra></extra>")
            fig.add_hline(y=annual_lpa, line_dash="dot", line_color="#94a3b8", annotation_text="Current profile", annotation_position="top left")
            fig.update_xaxes(title="Years of experience", dtick=5); fig.update_yaxes(title="Modeled salary (₹ LPA)", rangemode="tozero")
            make_plot_layout(fig, 350); st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False,"responsive":True})
        with right:
            st.markdown("**Salary by company size**")
            fig = px.bar(company_chart, x="Company size", y="Modeled salary (₹ LPA)", color="Scenario", color_discrete_map={"Selected profile":"#285fca", "Alternative":"#bac9e6"}, text="Modeled salary (₹ LPA)")
            fig.update_traces(texttemplate="₹%{text:.2f}L", textposition="outside", cliponaxis=False, hovertemplate="%{x}<br>₹%{y:.2f} LPA<extra></extra>")
            fig.update_xaxes(title=None, tickangle=-10); fig.update_yaxes(title="Modeled salary (₹ LPA)", rangemode="tozero"); fig.update_layout(showlegend=False)
            make_plot_layout(fig, 350); st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False,"responsive":True})
        st.markdown('<div class="section-heading">Scenario comparison table</div><div class="section-caption">Differences are relative to the current profile, not an external market benchmark.</div>', unsafe_allow_html=True)
        scenario_table = pd.concat([
            company_chart[["Company size", "Modeled salary (₹ LPA)", "Difference vs profile (₹ LPA)"]].rename(columns={"Company size":"Scenario"}),
            city_chart[["Work location tier", "Modeled salary (₹ LPA)", "Difference vs profile (₹ LPA)"]].rename(columns={"Work location tier":"Scenario"}),
        ], ignore_index=True)
        scenario_table["Modeled salary (₹ LPA)"] = scenario_table["Modeled salary (₹ LPA)"].round(2)
        scenario_table["Difference vs profile (₹ LPA)"] = scenario_table["Difference vs profile (₹ LPA)"].round(2)
        st.dataframe(scenario_table.style.format({"Modeled salary (₹ LPA)":"₹ {:.2f} LPA", "Difference vs profile (₹ LPA)":"{:+.2f} LPA"}), use_container_width=True, hide_index=True)
        st.download_button("Download scenario comparison (CSV)", data=scenario_table.to_csv(index=False).encode("utf-8"), file_name="compensation_scenarios.csv", mime="text/csv")

    with candidate_tab:
        st.markdown('<div class="section-heading">Candidate profile</div><div class="section-caption">Attributes used by the current prediction.</div>', unsafe_allow_html=True)
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown('<div class="panel"><div class="panel-label">Role & education</div><div class="panel-title">Candidate attributes</div></div>', unsafe_allow_html=True)
            st.write("**Target role**", profile["Job Role"]); st.write("**Job sector**", profile["Job Sector"]); st.write("**Education**", profile["Education Degree"]); st.write("**Specialization**", profile["Specialization"]); st.write("**Education level**", profile["Education Level"])
        with right:
            st.markdown('<div class="panel"><div class="panel-label">Employer & market</div><div class="panel-title">Work context</div></div>', unsafe_allow_html=True)
            st.write("**Experience**", f"{profile['Experience (Years)']:g} years"); st.write("**Company size**", profile["Company Size"]); st.write("**Location tier**", profile["City Tier"]); st.write("**Estimated annual salary**", f"₹{annual_lpa:,.2f} LPA"); st.write("**Monthly equivalent**", f"₹{monthly:,.0f}")
        st.info("Change profile variables in the sidebar and run the analysis again to produce a new scenario.")

    with notes_tab:
        st.markdown('<div class="section-heading">Model transparency & use</div><div class="section-caption">Context to review before interpreting results for HR decisions.</div>', unsafe_allow_html=True)
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown('<div class="panel"><div class="panel-label">Prediction method</div><div class="panel-title">Structured regression pipeline</div><p class="panel-description">The app loads a pre-trained pipeline and runs inference only. Engineered features include experience squared, career stage, education–sector interaction, sector–role interaction, and company–market interaction.</p></div>', unsafe_allow_html=True)
        with right:
            engine = html.escape(str(metadata.get("selected_model", "Metadata not available")))
            rows = int(metadata.get("training_rows", 0) or 0)
            mae = metadata.get("MAE_INR"); r2 = metadata.get("R2")
            mae_label = f"₹{float(mae):,.0f}" if mae is not None else "Not available"
            r2_label = f"{float(r2):.3f}" if r2 is not None else "Not available"
            st.markdown(f'<div class="panel"><div class="panel-label">Model card</div><div class="panel-title">{engine}</div><p class="panel-description">Training records: {rows:,}<br>Test MAE: {mae_label}<br>Test R²: {r2_label}</p></div>', unsafe_allow_html=True)
        st.warning("Estimates are sensitive to training-data quality, coverage, time period, and unobserved factors such as skills, employer brand, performance, and negotiation. Review error and fairness across relevant groups before operational HR use.")
        st.caption("Scenario charts are model-generated estimates. They are not observed market salaries, percentile benchmarks, or confidence intervals.")

    st.markdown("---")
    st.caption("Compensation Intelligence · HR analytics decision support. Validate model performance and fairness before using estimates in compensation decisions.")
