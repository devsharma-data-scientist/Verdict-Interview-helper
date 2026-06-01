import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from torchvision import transforms
import numpy as np
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 9
IMG_SIZE = 224

CLASS_NAMES = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'natural', 'sad', 'sleepy', 'surprised']

EMOTION_COLORS = {
    'angry':     '#FF4B4B',
    'contempt':  '#FF8C00',
    'disgust':   '#9ACD32',
    'fear':      '#9B59B6',
    'happy':     '#2ECC71',
    'natural':   '#3498DB',
    'sad':       '#5DADE2',
    'sleepy':    '#AAB7B8',
    'surprised': '#F1C40F',
}

EMOJI_MAP = {
    'angry': '😠', 'contempt': '😒', 'disgust': '🤢',
    'fear': '😨', 'happy': '😊', 'natural': '😐',
    'sad': '😢', 'sleepy': '😴', 'surprised': '😲'
}


STRESS_WEIGHTS = {'fear': 0.35, 'angry': 0.30, 'sad': 0.20, 'disgust': 0.15}

CONFIDENCE_POSITIVE = {'happy': 0.45, 'natural': 0.35, 'surprised': 0.20}
CONFIDENCE_NEGATIVE = {'fear': 0.40, 'angry': 0.35, 'sad': 0.25}


class EnsembleModel(nn.Module):
    def __init__(self, w1=0.16, w2=0.54, w3=0.30):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES)
        )
        self.b3 = timm.create_model('efficientnet_b3', pretrained=False, num_classes=NUM_CLASSES)
        self.b4 = timm.create_model('efficientnet_b4', pretrained=False, num_classes=NUM_CLASSES)
        self.w1, self.w2, self.w3 = w1, w2, w3

    def forward(self, x):
        p1 = F.softmax(self.resnet(x), dim=1)
        p2 = F.softmax(self.b3(x),    dim=1)
        p3 = F.softmax(self.b4(x),    dim=1)
        return self.w1 * p1 + self.w2 * p2 + self.w3 * p3


def load_ensemble(model_dir="models"):
    """Load ensemble model — returns model or None if files missing."""
    pth = os.path.join(model_dir, "best_ensemble.pth")
    if not os.path.exists(pth):
        return None
    model = EnsembleModel()
    model.load_state_dict(torch.load(pth, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    return model


def load_single_model(model_name, model_dir="models"):
    """Load individual models for comparison."""
    pth = os.path.join(model_dir, f"{model_name}.pth")
    if not os.path.exists(pth):
        return None

    if "resnet" in model_name or "best_model" in model_name:
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES)
        )
    elif "b3" in model_name:
        m = timm.create_model('efficientnet_b3', pretrained=False, num_classes=NUM_CLASSES)
    elif "b4" in model_name:
        m = timm.create_model('efficientnet_b4', pretrained=False, num_classes=NUM_CLASSES)
    else:
        return None

    m.load_state_dict(torch.load(pth, map_location=DEVICE))
    m = m.to(DEVICE).eval()
    return m


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@torch.no_grad()
def predict(model, face_rgb_np):
    """Returns (emotion_str, confidence_float, all_probs_dict)"""
    img = transform(face_rgb_np).unsqueeze(0).to(DEVICE)
    out = model(img)
    probs = out[0].cpu().numpy()
    idx = probs.argmax()
    emotion = CLASS_NAMES[idx]
    confidence = float(probs[idx]) * 100
    all_probs = {CLASS_NAMES[i]: float(probs[i]) * 100 for i in range(NUM_CLASSES)}
    return emotion, confidence, all_probs


def compute_stress_score(emotion_counts: dict, total: int) -> float:
    if total == 0:
        return 0.0
    score = 0.0
    for em, w in STRESS_WEIGHTS.items():
        score += w * (emotion_counts.get(em, 0) / total)
    return round(score * 100, 1)


def compute_confidence_score(emotion_counts: dict, total: int) -> float:
    if total == 0:
        return 50.0
    pos = sum(CONFIDENCE_POSITIVE.get(em, 0) * (emotion_counts.get(em, 0) / total)
              for em in CONFIDENCE_POSITIVE)
    neg = sum(CONFIDENCE_NEGATIVE.get(em, 0) * (emotion_counts.get(em, 0) / total)
              for em in CONFIDENCE_NEGATIVE)
    score = 50 + (pos - neg) * 100
    return round(min(max(score, 0), 100), 1)


def get_stress_label(score):
    if score <= 30:
        return "🟢 Low Stress", "#2ECC71"
    elif score <= 60:
        return "🟡 Moderate Stress", "#F39C12"
    else:
        return "🔴 High Stress", "#E74C3C"


def get_confidence_label(score):
    if score >= 70:
        return " High Confidence", "#2ECC71"
    elif score >= 40:
        return " Moderate Confidence", "#F39C12"
    else:
        return " Low Confidence", "#E74C3C"


def generate_ai_insights(emotion_dist: dict, stress: float, confidence: float,
                         duration_sec: int, timeline_df=None) -> list:
    insights = []
    total = sum(emotion_dist.values())
    if total == 0:
        return ["No data available for analysis."]

    happy_pct = emotion_dist.get('happy', 0) / total * 100
    fear_pct  = emotion_dist.get('fear', 0) / total * 100
    natural_pct = emotion_dist.get('natural', 0) / total * 100
    sleepy_pct = emotion_dist.get('sleepy', 0) / total * 100
    calm_pct = happy_pct + natural_pct

   
    if calm_pct >= 70:
        insights.append(f" Candidate remained calm and composed for {calm_pct:.0f}% of the interview.")
    elif calm_pct >= 50:
        insights.append(f" Candidate was mostly composed ({calm_pct:.0f}%), with occasional stress signs.")
    else:
        insights.append(f" Candidate showed stress or discomfort for majority of the interview.")

   
    if happy_pct >= 40:
        insights.append(f" Strong positive engagement — happiness detected {happy_pct:.0f}% of the time.")

  
    if fear_pct >= 20:
        insights.append(f" Noticeable anxiety detected ({fear_pct:.0f}%) — candidate may need confidence coaching.")

    
    if sleepy_pct >= 15:
        insights.append(f" Signs of fatigue or disengagement observed ({sleepy_pct:.0f}%).")

  
    if stress >= 61:
        insights.append(f" High stress score ({stress}) — candidate experienced significant pressure.")
    elif stress >= 31:
        insights.append(f" Moderate stress ({stress}) — manageable but noticeable under pressure.")
    else:
        insights.append(f" Low stress score ({stress}) — candidate handled the interview well.")

   
    if confidence >= 70:
        insights.append(f" Excellent confidence score ({confidence}) — strong overall performance.")
    elif confidence >= 40:
        insights.append(f" Average confidence ({confidence}) — room for improvement in assertiveness.")
    else:
        insights.append(f" Low confidence ({confidence}) — candidate may need more interview practice.")

   
    mins = duration_sec // 60
    if mins > 0:
        insights.append(f"⏱️ Interview lasted {mins} minute(s) — sufficient data for behavioral analysis.")

    
    if timeline_df is not None and len(timeline_df) > 10:
        mid = len(timeline_df) // 2
        first_half = timeline_df.iloc[:mid]
        second_half = timeline_df.iloc[mid:]
        f_stress = first_half['emotion'].isin(['fear', 'angry', 'sad']).mean()
        s_stress = second_half['emotion'].isin(['fear', 'angry', 'sad']).mean()
        if s_stress > f_stress + 0.15:
            insights.append(" Stress increased in the second half — likely due to harder questions.")
        elif f_stress > s_stress + 0.15:
            insights.append(" Candidate became more relaxed as the interview progressed — good adaptability.")

    return insights
