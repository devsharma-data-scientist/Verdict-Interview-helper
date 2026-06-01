import streamlit as st
import pandas as pd

from database.db_manager import (
    get_all_candidates, get_candidate_sessions,
    get_session_logs
)
from src.utils.model_utils import (
    CLASS_NAMES, compute_stress_score, compute_confidence_score,
    get_stress_label, get_confidence_label, generate_ai_insights, EMOTION_COLORS, EMOJI_MAP
)
from src.utils.charts import (
    emotion_pie_chart, emotion_bar_chart, emotion_timeline_chart,
    stress_timeline_chart, confidence_timeline_chart, gauge_chart
)



def run_analytics():
    st.markdown("""
    <style>
    .page-title { font-size: 1.6rem; font-weight: 700; color: #4ECDC4; }
    .section-title { color: #CBD5E0; font-size: 0.85rem; font-weight: 600;
                     text-transform: uppercase; letter-spacing: 1.5px; margin: 1rem 0 0.4rem; }
    .score-big { font-size: 2.8rem; font-weight: 800; }
    .insight-card {
        background: linear-gradient(135deg, #1A202C, #0D1117);
        border-left: 3px solid #4ECDC4; border-radius: 8px;
        padding: 0.7rem 1rem; margin: 0.4rem 0; font-size: 0.9rem; color: #CBD5E0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="page-title">📊 Session Analytics & Report</p>', unsafe_allow_html=True)

    # ── Select Session ────────────────────────────────────
    candidates_df = get_all_candidates()
    if candidates_df.empty:
        st.info("No candidates found. Run a live analysis session first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        candidate_name = st.selectbox("Select Candidate", candidates_df['name'].tolist())
    
    cid = int(candidates_df[candidates_df['name'] == candidate_name].iloc[0]['id'])
    sessions_df = get_candidate_sessions(cid)

    if sessions_df.empty:
        st.info(f"No sessions found for {candidate_name}.")
        return

    with col2:
        session_options = {
            f"{row['session_name']} | {row['start_time'][:16]}": row['id']
            for _, row in sessions_df.iterrows()
        }
        selected_session_label = st.selectbox("Select Session", list(session_options.keys()))

    session_id = session_options[selected_session_label]
    session_row = sessions_df[sessions_df['id'] == session_id].iloc[0]

    # ── Load Logs ─────────────────────────────────────────
    logs_df = get_session_logs(session_id)
    if logs_df.empty:
        st.warning("No emotion data recorded in this session.")
        return

    # ── Compute metrics ───────────────────────────────────
    emotion_counts = {e: int((logs_df['emotion'] == e).sum()) for e in CLASS_NAMES}
    total = len(logs_df)
    stress    = float(session_row.get('stress_score') or compute_stress_score(emotion_counts, total))
    confidence = float(session_row.get('confidence_score') or compute_confidence_score(emotion_counts, total))
    dominant  = max(emotion_counts, key=emotion_counts.get)
    duration  = int(session_row.get('duration_seconds') or logs_df['timestamp_sec'].max() or 0)
    insights  = generate_ai_insights(emotion_counts, stress, confidence, duration, logs_df)

    s_lbl, s_col = get_stress_label(stress)
    c_lbl, c_col = get_confidence_label(confidence)

    # ── Score Cards ───────────────────────────────────────
    st.markdown("---")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.markdown(f"""
    <div style="background:linear-gradient(135deg,#1A202C,#0D1117);border:1px solid #2D3748;
                border-radius:12px;padding:1rem;text-align:center;">
        <div class="score-big" style="color:{s_col}">{stress}</div>
        <div style="color:#718096;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px">Stress Score</div>
        <div style="color:{s_col};font-size:0.85rem;margin-top:4px">{s_lbl}</div>
    </div>""", unsafe_allow_html=True)

    mc2.markdown(f"""
    <div style="background:linear-gradient(135deg,#1A202C,#0D1117);border:1px solid #2D3748;
                border-radius:12px;padding:1rem;text-align:center;">
        <div class="score-big" style="color:{c_col}">{confidence}</div>
        <div style="color:#718096;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px">Confidence</div>
        <div style="color:{c_col};font-size:0.85rem;margin-top:4px">{c_lbl}</div>
    </div>""", unsafe_allow_html=True)

    mc3.markdown(f"""
    <div style="background:linear-gradient(135deg,#1A202C,#0D1117);border:1px solid #2D3748;
                border-radius:12px;padding:1rem;text-align:center;">
        <div class="score-big" style="font-size:2rem">{EMOJI_MAP.get(dominant,'😐')}</div>
        <div style="color:#4ECDC4;font-size:1rem;font-weight:700">{dominant.upper()}</div>
        <div style="color:#718096;font-size:0.75rem;text-transform:uppercase">Dominant</div>
    </div>""", unsafe_allow_html=True)

    mins, secs = duration // 60, duration % 60
    mc4.markdown(f"""
    <div style="background:linear-gradient(135deg,#1A202C,#0D1117);border:1px solid #2D3748;
                border-radius:12px;padding:1rem;text-align:center;">
        <div class="score-big" style="color:#F1C40F;font-size:2rem">{mins:02d}:{secs:02d}</div>
        <div style="color:#718096;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px">Duration</div>
        <div style="color:#F1C40F;font-size:0.85rem;margin-top:4px">{total} frames</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts Row 1: Pie + Bar ───────────────────────────
    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(emotion_pie_chart(emotion_counts), use_container_width=True)
    with ch2:
        st.plotly_chart(emotion_bar_chart(emotion_counts), use_container_width=True)

    # ── Timeline ──────────────────────────────────────────
    st.plotly_chart(emotion_timeline_chart(logs_df), use_container_width=True)

    # ── Stress + Confidence timelines ─────────────────────
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(stress_timeline_chart(logs_df), use_container_width=True)
    with g2:
        st.plotly_chart(confidence_timeline_chart(logs_df), use_container_width=True)

    # ── Gauge charts ──────────────────────────────────────
    gg1, gg2 = st.columns(2)
    with gg1:
        st.plotly_chart(gauge_chart(stress, "Stress Score"), use_container_width=True)
    with gg2:
        st.plotly_chart(gauge_chart(confidence, "Confidence Score"), use_container_width=True)

    # ── AI Insights ───────────────────────────────────────
    st.markdown('<p class="section-title">🤖 AI Behavioral Insights</p>', unsafe_allow_html=True)
    for insight in insights:
        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

    # ── Emotion Log Table ─────────────────────────────────
    with st.expander("📋 Full Emotion Log"):
        display_df = logs_df[['timestamp_sec', 'emotion', 'confidence']].copy()
        display_df['timestamp'] = display_df['timestamp_sec'].apply(
            lambda s: f"{int(s//60):02d}:{int(s%60):02d}"
        )
        display_df['emoji'] = display_df['emotion'].map(EMOJI_MAP)
        display_df['confidence'] = display_df['confidence'].round(1)
        st.dataframe(
            display_df[['timestamp', 'emoji', 'emotion', 'confidence']],
            use_container_width=True, height=300
        )

   
