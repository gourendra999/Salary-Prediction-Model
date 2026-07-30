import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="💼",
    layout="wide",
)


st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.main-title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 700;
    color: white;
}

.sub-text {
    text-align: center;
    color: #dcdcdc;
    margin-bottom: 20px;
}

.card {
    background: rgba(255, 255, 255, 0.08);
    padding: 20px;
    border-radius: 16px;
    backdrop-filter: blur(10px);
    margin-bottom: 15px;
}

.result-box {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    color: white;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.3);
}

.result-box h1 {
    font-size: 3rem;
    margin: 0;
}

button[kind="primary"] {
    background: linear-gradient(135deg, #ff7e5f, #feb47b);
    border: none;
    font-weight: bold;
}

button[kind="primary"]:hover {
    transform: scale(1.03);
}
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
for d in EDUCATION_DEGREES["Bachelor's Degrees"]: EDU_LEVEL[d] = "Bachelor"
for d in EDUCATION_DEGREES["Master's Degrees"]:   EDU_LEVEL[d] = "Master"
for d in EDUCATION_DEGREES["PhD / Doctoral"]:     EDU_LEVEL[d] = "Phd"

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
CITY_TIERS    = ["Tier 1 (Metro)", "Tier 2 (Mid-Sized)", "Tier 3 (Smaller Cities)"]

@st.cache_resource(show_spinner="Training model on 11,000 records...")
def load_model():
    return joblib.load("models/salary_model.pkl")

model = load_model()

st.markdown('<div class="main-title">💼 Salary Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">AI-powered Indian Salary Estimator</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("📊 Input Details")

PLACEHOLDER = "-- Select --"

edu_group = st.sidebar.selectbox("Degree Type", [PLACEHOLDER] + list(EDUCATION_DEGREES.keys()))

if edu_group != PLACEHOLDER:
    edu_degree = st.sidebar.selectbox("Education Degree", [PLACEHOLDER] + EDUCATION_DEGREES[edu_group])
else:
    edu_degree = PLACEHOLDER

if edu_degree != PLACEHOLDER:
    edu_level = EDU_LEVEL[edu_degree]
    spec_options = SPECIALIZATIONS.get(edu_degree, ["General"])
    specialization = st.sidebar.selectbox("Specialization", [PLACEHOLDER] + spec_options)
else:
    specialization = PLACEHOLDER

experience = st.sidebar.slider("Years of Experience", 0.0, 30.0, 0.0, 0.5)

sector = st.sidebar.selectbox("Job Sector", [PLACEHOLDER] + list(SECTORS_ROLES.keys()))

if sector != PLACEHOLDER:
    job_role = st.sidebar.selectbox("Job Role", [PLACEHOLDER] + SECTORS_ROLES[sector])
else:
    job_role = PLACEHOLDER

company_size = st.sidebar.selectbox("Company Size", [PLACEHOLDER] + COMPANY_SIZES)
city_tier = st.sidebar.selectbox("City Tier", [PLACEHOLDER] + CITY_TIERS,help="Tier 1: Bengaluru, Mumbai, Delhi, Hyderabad, Chennai, Pune\nTier 2: Noida, Gurgaon, Ahmedabad, Kolkata\nTier 3: All other cities"
)

# ---------------- VALIDATION ----------------
required = [edu_group, edu_degree, specialization, sector, job_role, company_size, city_tier]
all_filled = all(v != PLACEHOLDER for v in required)

# ---------------- PREDICT BUTTON ----------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Predict Salary", use_container_width=True):

    if not all_filled:
        st.warning("⚠️ Please fill all required fields.")
    else:
        input_df = pd.DataFrame({
            "Experience (Years)": [experience],
            "Education Degree": [edu_degree],
            "Specialization": [specialization],
            "Education Level": [EDU_LEVEL[edu_degree].title()],
            "Job Sector": [sector],
            "Job Role": [job_role],
            "Company Size": [company_size],
            "City Tier": [city_tier],
        })

        prediction = model.predict(input_df)[0]
        low, high = prediction * 0.93, prediction * 1.07

        # ---------------- RESULT DISPLAY ----------------
        st.markdown(f"""
        <div class="result-box">
            <p>Predicted Annual Salary</p>
            <h1>₹ {prediction/1e5:.2f} LPA</h1>
            <p>Range: ₹{low/1e5:.2f}L - ₹{high/1e5:.2f}L</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        col1.metric("💰 Monthly Salary", f"₹ {prediction/12/1e3:.1f}K")
        col2.metric("📈 Experience Level", f"{experience} yrs")