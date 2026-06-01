import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from database.db_manager import get_all_candidates, get_candidate_sessions, get_session_logs, add_candidate
from src.utils.model_utils import CLASS_NAMES, EMOTION_COLORS, EMOJI_MAP


def run_candidate_dashboard():
    st.markdown("""
    <style>
    .page-title { font-size: 1.6rem; font-weight: 700; color: #4ECDC4; }
    .cand-card {
        background: linear-gradient(135deg, #1A202C, #0D1117);
        border: 1px solid #2D3748; border-radius: 14px; padding: 1.2rem;
        margin-bottom: 0.8rem; cursor: pointer;
    }
    .cand-name { font-size: 1.1rem; font-weight: 700; color: #E2E8F0; }
    .cand-meta { font-size: 0.8rem; color: #718096; margin-top: 3px; }
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 0.75rem; font-weight: 600; margin-right: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="page-title">👤 Candidate Dashboard</p>', unsafe_allow_html=True)

    # ── Add New Candidate ──────────────────────────────────
    with st.expander("➕ Add New Candidate"):
        c1, c2, c3 = st.columns(3)
        with c1: nm = st.text_input("Full Name", key="add_name")
        with c2: pos = st.text_input("Position", key="add_pos")
        with c3: em = st.text_input("Email", key="add_email")
        if st.button("Add Candidate", type="primary"):
            if nm:
                add_candidate(nm, em, pos)
                st.success(f"✅ {nm} added!")
                st.rerun()
            else:
                st.warning("Name required.")

    st.markdown("---")

    candidates_df = get_all_candidates()
    if candidates_df.empty:
        st.info("No candidates yet. Add one above or run a Live Analysis session.")
        return

    # ── Candidate List with Stats ─────────────────────────
    st.markdown(f"**{len(candidates_df)} Candidates Registered**")

    for _, cand in candidates_df.iterrows():
        sessions = get_candidate_sessions(int(cand['id']))
        n_sessions = len(sessions)

        avg_stress = sessions['stress_score'].mean() if n_sessions > 0 else None
        avg_conf   = sessions['confidence_score'].mean() if n_sessions > 0 else None

        with st.container():
            col_info, col_metrics, col_btn = st.columns([3, 4, 1])

            with col_info:
                st.markdown(f"""
                <div>
                    <div class="cand-name">👤 {cand['name']}</div>
                    <div class="cand-meta">
                        {'📌 ' + cand['position'] if cand.get('position') else ''}
                        {'&nbsp;|&nbsp; ✉️ ' + cand['email'] if cand.get('email') else ''}
                    </div>
                    <div style="margin-top:6px">
                        <span class="badge" style="background:#1C3829;color:#2ECC71">
                            🎯 {n_sessions} Sessions
                        </span>
                        <span style="color:#4A5568;font-size:0.75rem">
                            {cand.get('created_at','')[:10]}
                        </span>
                    </div>
                </div>""", unsafe_allow_html=True)

            with col_metrics:
                if n_sessions > 0 and avg_stress is not None:
                    m1, m2 = st.columns(2)
                    s_col = '#2ECC71' if avg_stress <= 30 else '#F39C12' if avg_stress <= 60 else '#E74C3C'
                    c_col = '#2ECC71' if avg_conf >= 70 else '#F39C12' if avg_conf >= 40 else '#E74C3C'
                    m1.markdown(f"""
                    <div style="text-align:center;padding:6px;background:#0D1117;border-radius:8px">
                        <div style="font-size:1.4rem;font-weight:800;color:{s_col}">{avg_stress:.0f}</div>
                        <div style="font-size:0.7rem;color:#718096">Avg Stress</div>
                    </div>""", unsafe_allow_html=True)
                    m2.markdown(f"""
                    <div style="text-align:center;padding:6px;background:#0D1117;border-radius:8px">
                        <div style="font-size:1.4rem;font-weight:800;color:{c_col}">{avg_conf:.0f}</div>
                        <div style="font-size:0.7rem;color:#718096">Avg Conf</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:#4A5568;font-size:0.85rem'>No sessions yet</span>",
                                unsafe_allow_html=True)

            with col_btn:
                if st.button("📊", key=f"view_{cand['id']}", help="View History"):
                    st.session_state[f"show_history_{cand['id']}"] = \
                        not st.session_state.get(f"show_history_{cand['id']}", False)

        # ── Expandable History ────────────────────────────
        if st.session_state.get(f"show_history_{cand['id']}", False) and n_sessions > 0:
            with st.container():
                st.markdown(f"#### 📋 Session History — {cand['name']}")
                display_cols = ['session_name', 'start_time', 'duration_seconds',
                                'stress_score', 'confidence_score', 'dominant_emotion']
                display_cols = [c for c in display_cols if c in sessions.columns]
                st.dataframe(sessions[display_cols], use_container_width=True)

                # Mini trend chart
                if len(sessions) > 1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(len(sessions))),
                        y=sessions['confidence_score'].tolist(),
                        mode='lines+markers', name='Confidence',
                        line=dict(color='#4ECDC4', width=2),
                        marker=dict(size=6)
                    ))
                    fig.add_trace(go.Scatter(
                        x=list(range(len(sessions))),
                        y=sessions['stress_score'].tolist(),
                        mode='lines+markers', name='Stress',
                        line=dict(color='#FF6B6B', width=2),
                        marker=dict(size=6)
                    ))
                    fig.update_layout(
                        title=f"Progress Over Sessions",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(13,17,23,0.6)',
                        font=dict(color='#E2E8F0'),
                        xaxis=dict(gridcolor='#2D3748', title='Session #'),
                        yaxis=dict(gridcolor='#2D3748', range=[0,100]),
                        height=280,
                        margin=dict(t=40,b=30,l=40,r=20),
                        legend=dict(bgcolor='rgba(0,0,0,0)')
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown('<hr style="border-color:#1A202C;margin:0.5rem 0">', unsafe_allow_html=True)
