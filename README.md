#  AI Interview Analytics Platform

Real-time behavioral analysis for interview preparation and evaluation, powered by a weighted ensemble of ResNet50 + EfficientNet B3/B4.

---

##  Project Structure

```
interview_platform/
├── app.py                          # Main Streamlit entry point
├── requirements.txt
├── models/                         # ← PUT YOUR .pth FILES HERE
│   ├── best_ensemble.pth
│   ├── best_model.pth
│   ├── b3.pth
│   └── b4.pth
├── database/
│   ├── db_manager.py               # SQLite: candidates, sessions, emotion_logs
│   └── interview_platform.db       # Auto-created on first run
├── reports/                        # PDF reports saved here
└── src/
    ├── utils/
    │   ├── model_utils.py          # Model loading, predict(), score calculations
    │   ├── charts.py               # All Plotly charts
    │   └── pdf_generator.py        # ReportLab PDF generator
    └── pages/
        ├── live_analysis.py        #  Live camera feed + recording
        ├── analytics.py            #  Charts, scores, PDF export
        ├── candidate_dashboard.py  #  Candidate profiles & history
        └── interviewer_dashboard.py #  Rankings, comparison, radar chart
```

---


##  How to Use

### Interview Recording
1. Go to **🎥 Live Analysis**
2. Choose camera source (Webcam / IP URL / Video file)
3. Add/select a candidate
4. Click **▶ Start Analysis**
5. Conduct the interview
6. Click **⏹ Stop & Save** when done

### View Analytics
1. Go to **📊 Analytics & Reports**
2. Select candidate and session
3. View charts:
   - Emotion Distribution (Pie + Bar)
   - Emotion Timeline (confidence over time)
   - Stress Timeline
   - Confidence Timeline
   - Gauge charts
4. Read AI-generated behavioral insights
5. Click **📥 Generate PDF Report** → Download

### Interviewer Dashboard
- **Rankings**: Candidates ranked by confidence & stress
- **Comparison Charts**: Bar + Radar (spider) charts
- **All Sessions**: Full table view

---

##  Detected Emotions (9 Classes)
| Emotion | Emoji | Role in Scoring |
|---------|-------|----------------|
| happy | 😊 | Positive (confidence) |
| natural | 😐 | Positive (confidence) |
| surprised | 😲 | Slight positive |
| fear | 😨 | Stress + negative |
| angry | 😠 | Stress + negative |
| sad | 😢 | Stress + negative |
| disgust | 🤢 | Stress |
| contempt | 😒 | Neutral |
| sleepy | 😴 | Neutral/negative |

---

##  Scoring System

### Stress Score (0–100)
```
Fear   × 0.35
Angry  × 0.30
Sad    × 0.20
Disgust× 0.15
```
- 0–30: 🟢 Low Stress
- 31–60: 🟡 Moderate Stress
- 61–100: 🔴 High Stress

### Confidence Score (0–100)
```
Positive: Happy (0.45), Natural (0.35), Surprised (0.20)
Negative: Fear (0.40), Angry (0.35), Sad (0.25)
Base: 50
```

---

## 🛠️ Tech Stack
- **Streamlit** — Web UI
- **PyTorch + timm** — Ensemble model inference
- **OpenCV** — Face detection (Haar Cascade)
- **Plotly** — Interactive charts
- **SQLite** — Session storage
- **ReportLab** — PDF generation

---

##  IP Webcam Setup (Android)
1. Install **IP Webcam** app from Play Store
2. Start server in the app
3. Note the URL shown (e.g., `http://10.194.72.xx:4747/video`)
4. Paste in the "IP Webcam URL" field in Live Analysis

---

*Built for interview coaching, HR analytics, and behavioral research.*
