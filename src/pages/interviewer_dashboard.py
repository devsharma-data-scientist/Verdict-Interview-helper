import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from database.db_manager import get_all_sessions_with_candidates, get_session_logs
from src.utils.model_utils import CLASS_NAMES, EMOTION_COLORS, EMOJI_MAP
from src.utils.charts import candidate_comparison_chart


def run_interviewer_dashboard():
    st.markdown("""
    <style>
    .page-title { font-size: 1.6rem; font-weight: 700; color: #4ECDC4; }
    .rank-card {
        background: linear-gradient(135deg, #1A202C, #0D1117);
        border: 1px solid #2D3748; border-radius: 12px;
        padding: 0.8rem 1rem; margin-bottom: 0.5rem;
        display: flex; align-items: center; gap: 12px;
    }
    .rank-num { font-size: 1.4rem; font-weight: 800; color: #F1C40F; min-width: 32px; }
    .rank-name { font-size: 1rem; font-weight: 600; color: #E2E8F0; }
    .rank-pos { font-size: 0.8rem; color: #718096; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="page-title">🎯 Interviewer Dashboard</p>', unsafe_allow_html=True)
    st.markdown("Compare candidates side-by-side, view rankings, and make informed decisions.")

    all_sessions = get_all_sessions_with_candidates()

    if all_sessions.empty:
        st.info("No completed sessions found. Run live analysis first.")
        return

    # ── Summary Metrics ───────────────────────────────────
    total_candidates = all_sessions['candidate_name'].nunique()
    total_sessions   = len(all_sessions)
    avg_stress       = all_sessions['stress_score'].mean()
    avg_conf         = all_sessions['confidence_score'].mean()

    m1, m2, m3, m4 = st.columns(4)
    for col, val, label, color in [
        (m1, total_candidates, "Candidates", "#4ECDC4"),
        (m2, total_sessions,   "Sessions",   "#F1C40F"),
        (m3, f"{avg_stress:.0f}" if pd.notna(avg_stress) else "N/A", "Avg Stress", "#FF6B6B"),
        (m4, f"{avg_conf:.0f}" if pd.notna(avg_conf) else "N/A", "Avg Confidence", "#2ECC71"),
    ]:
        col.markdown(f"""
        <div style="background:linear-gradient(135deg,#1A202C,#0D1117);border:1px solid #2D3748;
                    border-radius:12px;padding:1rem;text-align:center;">
            <div style="font-size:2.2rem;font-weight:800;color:{color}">{val}</div>
            <div style="font-size:0.75rem;color:#718096;text-transform:uppercase;letter-spacing:1px">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Best per candidate (last session) ─────────────────
    latest = all_sessions.sort_values('start_time').groupby('candidate_name').last().reset_index()
    latest = latest.dropna(subset=['confidence_score', 'stress_score'])

    tab1, tab2, tab3 = st.tabs(["🏆 Rankings", "📊 Comparison Charts", "📋 All Sessions"])

    with tab1:
        col_conf, col_stress = st.columns(2)

        with col_conf:
            st.markdown("#### 🌟 Confidence Ranking")
            ranked = latest.sort_values('confidence_score', ascending=False).reset_index(drop=True)
            medals = ["🥇", "🥈", "🥉"] + ["🎖️"] * 20
            for i, row in ranked.iterrows():
                c_col = '#2ECC71' if row['confidence_score'] >= 70 else \
                        '#F39C12' if row['confidence_score'] >= 40 else '#E74C3C'
                dom_emoji = EMOJI_MAP.get(row.get('dominant_emotion',''), '😐')
                st.markdown(f"""
                <div class="rank-card">
                    <div class="rank-num">{medals[i]}</div>
                    <div style="flex:1">
                        <div class="rank-name">{row['candidate_name']}</div>
                        <div class="rank-pos">{row.get('position','') or 'N/A'} &nbsp;|&nbsp; {dom_emoji} {row.get('dominant_emotion','')}</div>
                    </div>
                    <div style="font-size:1.5rem;font-weight:800;color:{c_col}">{row['confidence_score']:.0f}</div>
                </div>""", unsafe_allow_html=True)

        with col_stress:
            st.markdown("#### 😌 Stress Ranking (Lower = Better)")
            ranked_s = latest.sort_values('stress_score', ascending=True).reset_index(drop=True)
            for i, row in ranked_s.iterrows():
                s_col = '#2ECC71' if row['stress_score'] <= 30 else \
                        '#F39C12' if row['stress_score'] <= 60 else '#E74C3C'
                st.markdown(f"""
                <div class="rank-card">
                    <div class="rank-num">{medals[i]}</div>
                    <div style="flex:1">
                        <div class="rank-name">{row['candidate_name']}</div>
                        <div class="rank-pos">{row.get('position','') or 'N/A'}</div>
                    </div>
                    <div style="font-size:1.5rem;font-weight:800;color:{s_col}">{row['stress_score']:.0f}</div>
                </div>""", unsafe_allow_html=True)

    with tab2:
        if len(latest) >= 2:
            st.plotly_chart(candidate_comparison_chart(latest), use_container_width=True)

            # Spider / Radar chart
            st.markdown("#### 🕸️ Multi-Dimension Candidate Comparison")
            top_n = min(5, len(latest))
            top_cands = latest.sort_values('confidence_score', ascending=False).head(top_n)

            categories = ['Confidence', 'Calmness', 'Engagement', 'Consistency', 'Positivity']
            fig = go.Figure()

            for _, row in top_cands.iterrows():
                # Derived scores
                conf   = float(row.get('confidence_score', 50))
                stress = float(row.get('stress_score', 50))
                calmness = max(0, 100 - stress)
                logs = get_session_logs(int(row['id']))
                engagement = 60.0
                consistency = 60.0
                positivity  = 60.0
                if not logs.empty:
                    total = len(logs)
                    happy_pct   = (logs['emotion'] == 'happy').sum() / total * 100
                    positivity  = min(100, happy_pct * 2 + conf * 0.3)
                    consistency = min(100, 100 - logs['confidence'].std())
                    engagement  = min(100, (1 - (logs['emotion'] == 'natural').mean()) * 100 * 0.5 + conf * 0.5)

                values = [conf, calmness, engagement, consistency, positivity]
                values += [values[0]]  # close the loop

                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=row['candidate_name'],
                    opacity=0.7,
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0,100],
                                    gridcolor='#2D3748', color='#718096'),
                    angularaxis=dict(gridcolor='#2D3748', color='#E2E8F0')
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E2E8F0'),
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                height=450,
                margin=dict(t=30, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 candidates with completed sessions for comparison.")

    with tab3:
        display = all_sessions[[
            'candidate_name', 'position', 'session_name',
            'start_time', 'duration_seconds',
            'stress_score', 'confidence_score', 'dominant_emotion'
        ]].copy()
        display['start_time'] = display['start_time'].str[:16]
        display.columns = [
            'Candidate', 'Position', 'Session', 'Date',
            'Duration(s)', 'Stress', 'Confidence', 'Dominant Emotion'
        ]
        st.dataframe(display, use_container_width=True, height=400)
