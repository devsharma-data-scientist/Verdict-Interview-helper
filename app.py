import streamlit as st
import sys
import os

from huggingface_hub import hf_hub_download
import os

os.makedirs("models", exist_ok=True)

MODEL_FILES = [
    "best_ensemble.pth",
    "best_model.pth",
    "best_model_b3.pth",
    "best_model_b4.pth"
]

for model_file in MODEL_FILES:
    hf_hub_download(
        repo_id="devsharma0601/verdict-models",
        filename=model_file,
        local_dir="models"
    )

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.db_manager import init_db
from src.utils.model_utils import load_ensemble

st.set_page_config(
    page_title="Interview Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; background-color: #0D1117 !important; color: #E2E8F0 !important; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0D1117 0%, #141923 100%) !important; border-right: 1px solid #1E2630 !important; }
section[data-testid="stSidebar"] * { color: #CBD5E0 !important; }
div[data-testid="stButton"] > button { background: linear-gradient(135deg, #4ECDC4, #2C9E97) !important; color: #0D1117 !important; font-weight: 700 !important; border: none !important; border-radius: 8px !important; transition: all 0.2s ease !important; }
div[data-testid="stButton"] > button:hover { transform: translateY(-1px) !important; box-shadow: 0 4px 15px rgba(78,205,196,0.4) !important; }
div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input { background: #1A202C !important; border: 1px solid #2D3748 !important; color: #E2E8F0 !important; border-radius: 8px !important; }
hr { border-color: #1E2630 !important; }
details { background: #141923 !important; border: 1px solid #2D3748 !important; border-radius: 10px !important; }
::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #0D1117; } ::-webkit-scrollbar-thumb { background: #2D3748; border-radius: 3px; }
div[data-testid="stTabs"] button { background: transparent !important; color: #718096 !important; font-weight: 600 !important; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: #4ECDC4 !important; border-bottom: 2px solid #4ECDC4 !important; }
</style>
""", unsafe_allow_html=True)

init_db()

@st.cache_resource(show_spinner=" Loading AI Ensemble Models...")
def get_model():
    return load_ensemble("models")

if 'model' not in st.session_state:
    st.session_state.model = get_model()

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.2rem 0 0.8rem;">
        <div style="font-size:2.5rem"></div>
        <div style="font-size:1.1rem;font-weight:800;color:#4ECDC4">Verdict</div>
        <div style="font-size:0.7rem;color:#4A5568;text-transform:uppercase;letter-spacing:2px">Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#1E2630;margin:0.5rem 0 1rem">', unsafe_allow_html=True)

    model_ok = st.session_state.model is not None
    st.markdown(f"""
    <div style="background:#0D1117;border:1px solid #1E2630;border-radius:8px;padding:0.6rem 0.8rem;margin-bottom:1rem;">
        <div style="font-size:0.65rem;color:#4A5568;text-transform:uppercase;letter-spacing:1px">Model Status</div>
        <div style="font-size:0.85rem;font-weight:700;color:{'#2ECC71' if model_ok else '#E74C3C'}">
            {' Ensemble Loaded' if model_ok else ' Models Not Found'}
        </div>
        <div style="font-size:0.65rem;color:#4A5568">ResNet50 + EfficientNet B3/B4</div>
    </div>
    """, unsafe_allow_html=True)

    if 'current_page' not in st.session_state:
        st.session_state.current_page = "home"

    nav_items = [
    ("🏠", "Home",                    "home"),
    ("🎥", "Live Analysis",           "live"),
    ("📊", "Analytics & Reports",     "analytics"),
    ("👤", "Candidate Dashboard",     "candidates"),
    ("🎯", "Interviewer Dashboard",   "interviewer"),
    ]
    st.markdown('<div style="font-size:0.65rem;color:#4A5568;text-transform:uppercase;letter-spacing:2px;margin-bottom:0.5rem">Navigation</div>', unsafe_allow_html=True)
    for icon, label, key in nav_items:
        is_active = st.session_state.current_page == key
        bg = "rgba(78,205,196,0.15)" if is_active else "transparent"
        border = "#4ECDC4" if is_active else "transparent"
        if st.button(f"{icon}  {label}", use_container_width=True, key=f"nav_{key}"):
            st.session_state.current_page = key
            st.rerun()

    st.markdown('<hr style="border-color:#1E2630;margin:1.5rem 0 0.5rem">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.65rem;color:#2D3748;text-align:center">9 Emotions · SQLite · PDF Export</div>', unsafe_allow_html=True)

page = st.session_state.current_page

if page == "home":
    st.markdown("""
    <div style="text-align:center;padding:3rem 2rem 2rem;">
        <div style="font-size:4rem"></div>
        <h1 style="font-size:2.4rem;font-weight:800;color:#4ECDC4;margin:0.5rem 0">Interview Analytics Platform</h1>
        <p style="font-size:1rem;color:#718096;max-width:580px;margin:0 auto 2rem">
            Real-time behavioral analysis powered by ensemble deep learning.
            Track emotions, measure stress, and generate professional interview reports.
        </p>
    </div>
    """, unsafe_allow_html=True)

    features = [
    ("🎥", "Live Detection",     "Webcam, IP cam, or video upload with real-time overlay"),
    ("📊", "Deep Analytics",     "Stress score, confidence score, timeline charts"),
    ("👤", "Candidate Profiles", "Track candidates across multiple sessions"),
    ("🎯", "Interviewer View",   "Rankings, radar charts, side-by-side comparison"),
    ("🤖", "AI Insights",        "Auto behavioral observations from emotion patterns"),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        cols[i % 3].markdown(f"""
        <div style="background:linear-gradient(135deg,#1A202C,#0D1117);border:1px solid #2D3748;
                    border-radius:14px;padding:1.4rem;margin-bottom:1rem;text-align:center">
            <div style="font-size:2rem;margin-bottom:0.5rem">{icon}</div>
            <div style="font-size:0.95rem;font-weight:700;color:#E2E8F0;margin-bottom:0.3rem">{title}</div>
            <div style="font-size:0.8rem;color:#718096;line-height:1.5">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#1C3829,#0D1A14);border:1px solid #2D5A3D;
                border-radius:14px;padding:1.5rem 2rem;margin-top:0.5rem">
        <h3 style="color:#2ECC71;margin:0 0 0.8rem">⚡ Quick Start</h3>
        <ol style="color:#CBD5E0;line-height:2.2;font-size:0.88rem;margin:0">
            <li>Put model files (<code>best_ensemble.pth, b3.pth, b4.pth, best_model.pth</code>) in <code>models/</code> folder</li>
            <li>Click <strong style="color:#4ECDC4"> Live Analysis</strong> → select candidate → Start Analysis</li>
            <li>Run interview, then click <strong style="color:#FF6B6B">⏹ Stop & Save</strong></li>
            <li>Go to <strong style="color:#4ECDC4"> Analytics</strong> → view charts → Download PDF Report</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

elif page == "live":
    from src.pages.live_analysis import run_live_analysis
    run_live_analysis()

elif page == "analytics":
    from src.pages.analytics import run_analytics
    run_analytics()

elif page == "candidates":
    from src.pages.candidate_dashboard import run_candidate_dashboard
    run_candidate_dashboard()

elif page == "interviewer":
    from src.pages.interviewer_dashboard import run_interviewer_dashboard
    run_interviewer_dashboard()
