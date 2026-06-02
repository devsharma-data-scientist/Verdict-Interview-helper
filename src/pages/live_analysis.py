import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime
import pandas as pd

from src.utils.model_utils import (
    predict, CLASS_NAMES, EMOTION_COLORS, EMOJI_MAP,
    compute_stress_score, compute_confidence_score,
    get_stress_label, get_confidence_label, generate_ai_insights
)
from database.db_manager import (
    get_all_candidates, add_candidate, create_session,
    close_session, log_emotion
)


def draw_overlay(frame, faces, emotion, confidence, all_probs):
    """Draw bounding box and emotion overlay on frame."""
    for (x, y, w, h) in faces:
        color_hex = EMOTION_COLORS.get(emotion, '#FFFFFF')
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        color = (b, g, r)  # BGR

        # Box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        # Corner accents
        corner_len = 15
        cv2.line(frame, (x, y), (x+corner_len, y), color, 3)
        cv2.line(frame, (x, y), (x, y+corner_len), color, 3)
        cv2.line(frame, (x+w, y), (x+w-corner_len, y), color, 3)
        cv2.line(frame, (x+w, y), (x+w, y+corner_len), color, 3)
        cv2.line(frame, (x, y+h), (x+corner_len, y+h), color, 3)
        cv2.line(frame, (x, y+h), (x, y+h-corner_len), color, 3)
        cv2.line(frame, (x+w, y+h), (x+w-corner_len, y+h), color, 3)
        cv2.line(frame, (x+w, y+h), (x+w, y+h-corner_len), color, 3)

        # Label background
        label = f"{emotion.upper()}  {confidence:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        cv2.rectangle(frame, (x, y-30), (x+tw+10, y), color, -1)
        cv2.putText(frame, label, (x+5, y-8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 1)
    return frame


def run_live_analysis():
    st.markdown("""
    <style>
    .live-header { font-size: 1.6rem; font-weight: 700; color: #4ECDC4; margin-bottom: 0.3rem; }
    .metric-card {
        background: linear-gradient(135deg, #1A202C, #0D1117);
        border: 1px solid #2D3748; border-radius: 12px;
        padding: 1rem 1.2rem; text-align: center; margin-bottom: 0.6rem;
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #4ECDC4; }
    .metric-label { font-size: 0.75rem; color: #718096; text-transform: uppercase; letter-spacing: 1px; }
    .emotion-badge {
        display: inline-block; padding: 6px 14px;
        border-radius: 20px; font-weight: 700; font-size: 1rem;
        margin: 2px;
    }
    .section-title { color: #CBD5E0; font-size: 0.9rem; font-weight: 600;
                     text-transform: uppercase; letter-spacing: 1.5px; margin: 0.8rem 0 0.4rem; }
    </style>
    """, unsafe_allow_html=True)

    # ── Setup Section ─────────────────────────────────────
    st.markdown('<p class="live-header">🎥 Live Interview Analysis</p>', unsafe_allow_html=True)

    if 'model' not in st.session_state or st.session_state.model is None:
        st.error("⚠️ Model not loaded. Please check that model files are in the `models/` folder.")
        st.info("Required: `best_ensemble.pth`, `b3.pth`, `b4.pth`, `best_model.pth`")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    col_setup, col_candidate = st.columns([1, 1])

    with col_setup:
        st.markdown('<p class="section-title">📷 Camera Source</p>', unsafe_allow_html=True)
        source_type = "Upload Vedio"
        uploaded = st.file_uploader("Upload Video", type=['mp4','avi','mov','mkv'])
        if uploaded:
            import tempfile, os
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded.read())
            video_source = tfile.name
        else:
            video_source = None

    with col_candidate:
        st.markdown('<p class="section-title">👤 Candidate Setup</p>', unsafe_allow_html=True)
        candidates_df = get_all_candidates()
        options = ["➕ New Candidate"] + list(candidates_df['name'].values if not candidates_df.empty else [])
        selected = st.selectbox("Select Candidate", options)

        if selected == "➕ New Candidate":
            new_name = st.text_input("Candidate Name", placeholder="Enter full name")
            new_pos  = st.text_input("Position Applied", placeholder="e.g. ML Engineer")
            new_email = st.text_input("Email*", placeholder="email@example.com")
        else:
            row = candidates_df[candidates_df['name'] == selected].iloc[0]
            new_name = selected
            new_pos  = row.get('position', '')
            new_email = row.get('email', '')

        session_name = st.text_input("Session Name", value=f"Interview {datetime.now().strftime('%d%b %H:%M')}")

    # ── Controls ──────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
    with ctrl1:
        start_btn = st.button("▶ Start Analysis", use_container_width=True, type="primary")
    with ctrl2:
        stop_btn = st.button("⏹ Stop & Save", use_container_width=True)

    st.divider()

    # ── Live Feed + Metrics ────────────────────────────────
    col_feed, col_stats = st.columns([3, 2])

    with col_feed:
        feed_placeholder = st.empty()

    with col_stats:
        st.markdown('<p class="section-title">📊 Live Stats</p>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        emotion_metric    = m1.empty()
        confidence_metric = m2.empty()
        m3, m4 = st.columns(2)
        stress_metric     = m3.empty()
        frames_metric     = m4.empty()
        st.markdown('<p class="section-title">📈 Emotion Counts</p>', unsafe_allow_html=True)
        counts_placeholder = st.empty()
        st.markdown('<p class="section-title">🕐 Recent Timeline</p>', unsafe_allow_html=True)
        timeline_placeholder = st.empty()

    # ── State init ────────────────────────────────────────
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'emotion_log' not in st.session_state:
        st.session_state.emotion_log = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'session_start' not in st.session_state:
        st.session_state.session_start = None
    if 'candidate_id' not in st.session_state:
        st.session_state.candidate_id = None

    # ── Start ─────────────────────────────────────────────
    if start_btn:
        if not new_name:
            st.warning("Please enter a candidate name.")
            return
        
        if not new_email:
            st.warning("Please enter email.")
            return

        # Create/find candidate
        if selected == "➕ New Candidate":
            cid = add_candidate(new_name, new_email, new_pos)
        else:
            cid = int(candidates_df[candidates_df['name'] == selected].iloc[0]['id'])

        sid = create_session(cid, session_name)
        st.session_state.recording = True
        st.session_state.emotion_log = []
        st.session_state.session_id = sid
        st.session_state.session_start = datetime.now().isoformat()
        st.session_state.candidate_id = cid
        st.session_state.analysis_started = True
        st.success(f"✅ Session started for **{new_name}** — ID: {sid}")

    # ── Stop ──────────────────────────────────────────────
    if stop_btn and st.session_state.recording:
        st.session_state.recording = False
        log = st.session_state.emotion_log
        if log and st.session_state.session_id:
            emo_counts = {e: sum(1 for l in log if l['emotion'] == e) for e in CLASS_NAMES}
            total = len(log)
            stress = compute_stress_score(emo_counts, total)
            conf   = compute_confidence_score(emo_counts, total)
            dominant = max(emo_counts, key=emo_counts.get)
            close_session(
                st.session_state.session_id, stress, conf,
                dominant, total, st.session_state.session_start
            )
            st.success(f"✅ Session saved! Stress: {stress} | Confidence: {conf}")
            st.balloons()

    # ── Main recording loop ───────────────────────────────
    if st.session_state.recording and video_source is not None:
        cap = cv2.VideoCapture(video_source)

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30

        frame_interval = int(fps * 0.5)

        frame_count = 0

        if not cap.isOpened():
            st.error("❌ Cannot open video source. Check camera/URL.")
            st.session_state.recording = False
            return

        start_ts = time.time()
        emotion_counts_live = {e: 0 for e in CLASS_NAMES}

        while st.session_state.recording:
            ret, frame = cap.read()
            if not ret:
                if source_type == "Upload Video":
                    st.info("Video analysis complete.")
                break

            frame_count += 1

            if frame_count % frame_interval != 0:
                continue

            elapsed = time.time() - start_ts
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(60,60))

            emotion, confidence = 'natural', 0.0
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                try:
                    emotion, confidence, all_probs = predict(st.session_state.model, face_rgb)
                except Exception:
                    all_probs = {}

                frame = draw_overlay(frame, [faces[0]], emotion, confidence, all_probs)

                # Log every 3rd frame to reduce DB writes
                if len(st.session_state.emotion_log) % 3 == 0:
                    log_emotion(st.session_state.session_id, round(elapsed, 2), emotion, confidence)

                st.session_state.emotion_log.append({
                    'timestamp_sec': round(elapsed, 2),
                    'emotion': emotion,
                    'confidence': round(confidence, 2)
                })
                emotion_counts_live[emotion] += 1

            # Timestamp on frame
            cv2.putText(frame, f"⏱ {int(elapsed//60):02d}:{int(elapsed%60):02d}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (78, 205, 196), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feed_placeholder.image(frame_rgb, channels='RGB', use_container_width=True)

            # Update metrics every 10 frames
            if len(st.session_state.emotion_log) % 10 == 0:
                total = max(sum(emotion_counts_live.values()), 1)
                stress = compute_stress_score(emotion_counts_live, total)
                conf   = compute_confidence_score(emotion_counts_live, total)
                s_lbl, s_col = get_stress_label(stress)
                c_lbl, c_col = get_confidence_label(conf)
                emoji = EMOJI_MAP.get(emotion, '😐')

                emotion_metric.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:2.5rem">{emoji}</div>
                    <div class="metric-value" style="color:{EMOTION_COLORS.get(emotion,'#4ECDC4')};font-size:1.2rem">{emotion.upper()}</div>
                    <div class="metric-label">Current Emotion</div>
                </div>""", unsafe_allow_html=True)

                confidence_metric.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{confidence:.0f}%</div>
                    <div class="metric-label">Detection Confidence</div>
                </div>""", unsafe_allow_html=True)

                stress_metric.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{s_col};font-size:1.5rem">{stress}</div>
                    <div class="metric-label">Stress Score</div>
                </div>""", unsafe_allow_html=True)

                frames_metric.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size:1.5rem">{total}</div>
                    <div class="metric-label">Frames Analyzed</div>
                </div>""", unsafe_allow_html=True)

                # Live counts
                counts_html = ""
                for em, cnt in sorted(emotion_counts_live.items(), key=lambda x: -x[1]):
                    if cnt > 0:
                        col = EMOTION_COLORS.get(em, '#888')
                        pct = cnt / total * 100
                        counts_html += f"""
                        <div style="display:flex;justify-content:space-between;align-items:center;
                                    padding:4px 0;border-bottom:1px solid #2D3748;">
                            <span style="color:{col};font-weight:600">{EMOJI_MAP.get(em,'')} {em}</span>
                            <span style="color:#CBD5E0">{pct:.0f}%</span>
                        </div>"""
                counts_placeholder.markdown(counts_html, unsafe_allow_html=True)

                # Timeline last 8
                recent = st.session_state.emotion_log[-8:]
                tl_html = ""
                for entry in reversed(recent):
                    ts = entry['timestamp_sec']
                    em = entry['emotion']
                    conf_v = entry['confidence']
                    col = EMOTION_COLORS.get(em, '#888')
                    m, s = int(ts // 60), int(ts % 60)
                    tl_html += f"""
                    <div style="display:flex;gap:8px;align-items:center;padding:3px 0;
                                border-bottom:1px solid #1A202C;font-size:0.85rem;">
                        <span style="color:#718096;min-width:40px">{m:02d}:{s:02d}</span>
                        <span style="color:{col};font-weight:600">{EMOJI_MAP.get(em,'')} {em}</span>
                        <span style="color:#4A5568;margin-left:auto">{conf_v:.0f}%</span>
                    </div>"""
                timeline_placeholder.markdown(tl_html, unsafe_allow_html=True)

        cap.release()
