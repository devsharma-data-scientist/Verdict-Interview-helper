import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from src.utils.model_utils import EMOTION_COLORS, CLASS_NAMES


def emotion_pie_chart(emotion_counts: dict):
    labels = [k for k, v in emotion_counts.items() if v > 0]
    values = [v for v in emotion_counts.values() if v > 0]
    colors = [EMOTION_COLORS.get(k, '#888') for k in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors, line=dict(color='#0D1117', width=2)),
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Frames: %{value}<br>%{percent}<extra></extra>',
        hole=0.4
    ))
    fig.update_layout(
        title=dict(text="Emotion Distribution", font=dict(size=18, color='#E2E8F0')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E2E8F0'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#E2E8F0')),
        margin=dict(t=50, b=20, l=20, r=20),
        height=380,
    )
    return fig


def emotion_bar_chart(emotion_counts: dict):
    emotions = list(emotion_counts.keys())
    values = list(emotion_counts.values())
    colors = [EMOTION_COLORS.get(e, '#888') for e in emotions]

    fig = go.Figure(go.Bar(
        x=emotions, y=values,
        marker=dict(color=colors, line=dict(color='#0D1117', width=1)),
        text=[f"{v}" for v in values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text="Emotion Frequency", font=dict(size=18, color='#E2E8F0')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,17,23,0.6)',
        font=dict(color='#E2E8F0'),
        xaxis=dict(gridcolor='#2D3748', title='Emotion'),
        yaxis=dict(gridcolor='#2D3748', title='Frame Count'),
        margin=dict(t=50, b=40, l=40, r=20),
        height=350,
    )
    return fig


def emotion_timeline_chart(df: pd.DataFrame):
    """Line chart: confidence of each emotion over time."""
    if df.empty:
        return go.Figure()

    fig = go.Figure()
    for emotion in CLASS_NAMES:
        sub = df[df['emotion'] == emotion]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub['timestamp_sec'],
            y=sub['confidence'],
            mode='lines+markers',
            name=emotion,
            line=dict(color=EMOTION_COLORS.get(emotion, '#888'), width=2),
            marker=dict(size=4),
            hovertemplate=f'<b>{emotion}</b><br>Time: %{{x:.1f}}s<br>Confidence: %{{y:.1f}}%<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text="Emotion Confidence Timeline", font=dict(size=18, color='#E2E8F0')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,17,23,0.6)',
        font=dict(color='#E2E8F0'),
        xaxis=dict(gridcolor='#2D3748', title='Time (seconds)'),
        yaxis=dict(gridcolor='#2D3748', title='Confidence (%)', range=[0, 105]),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#E2E8F0')),
        hovermode='x unified',
        margin=dict(t=50, b=40, l=50, r=20),
        height=400,
    )
    return fig


def stress_timeline_chart(df: pd.DataFrame):
    """Rolling stress score over time."""
    if df.empty:
        return go.Figure()

    STRESS_EM = {'fear': 0.35, 'angry': 0.30, 'sad': 0.20, 'disgust': 0.15}
    df = df.copy().sort_values('timestamp_sec')
    df['stress_val'] = df['emotion'].map(lambda e: STRESS_EM.get(e, 0)) * df['confidence'] / 100 * 100

    window = max(5, len(df) // 20)
    df['rolling_stress'] = df['stress_val'].rolling(window=window, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp_sec'], y=df['rolling_stress'],
        mode='lines', fill='tozeroy',
        line=dict(color='#FF6B6B', width=2),
        fillcolor='rgba(255,107,107,0.15)',
        name='Stress',
        hovertemplate='Time: %{x:.1f}s<br>Stress: %{y:.1f}<extra></extra>'
    ))
    # Zone lines
    for y_val, color, label in [(30, '#2ECC71', 'Low'), (60, '#F39C12', 'Moderate')]:
        fig.add_hline(y=y_val, line_dash='dash', line_color=color,
                      annotation_text=label, annotation_font_color=color)

    fig.update_layout(
        title=dict(text="Stress Score Over Time", font=dict(size=18, color='#E2E8F0')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,17,23,0.6)',
        font=dict(color='#E2E8F0'),
        xaxis=dict(gridcolor='#2D3748', title='Time (seconds)'),
        yaxis=dict(gridcolor='#2D3748', title='Stress Level', range=[0, 105]),
        margin=dict(t=50, b=40, l=50, r=20),
        height=350,
    )
    return fig


def confidence_timeline_chart(df: pd.DataFrame):
    """Rolling interview confidence score over time."""
    if df.empty:
        return go.Figure()

    POS = {'happy': 0.45, 'natural': 0.35, 'surprised': 0.20}
    NEG = {'fear': 0.40, 'angry': 0.35, 'sad': 0.25}
    df = df.copy().sort_values('timestamp_sec')

    def conf_val(row):
        w = row['confidence'] / 100
        if row['emotion'] in POS:
            return 50 + POS[row['emotion']] * w * 100
        elif row['emotion'] in NEG:
            return 50 - NEG[row['emotion']] * w * 100
        return 50

    df['conf_val'] = df.apply(conf_val, axis=1)
    window = max(5, len(df) // 20)
    df['rolling_conf'] = df['conf_val'].rolling(window=window, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp_sec'], y=df['rolling_conf'],
        mode='lines', fill='tozeroy',
        line=dict(color='#4ECDC4', width=2),
        fillcolor='rgba(78,205,196,0.15)',
        name='Confidence',
        hovertemplate='Time: %{x:.1f}s<br>Confidence: %{y:.1f}<extra></extra>'
    ))
    fig.add_hline(y=50, line_dash='dot', line_color='#888',
                  annotation_text='Baseline', annotation_font_color='#888')

    fig.update_layout(
        title=dict(text="Interview Confidence Over Time", font=dict(size=18, color='#E2E8F0')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,17,23,0.6)',
        font=dict(color='#E2E8F0'),
        xaxis=dict(gridcolor='#2D3748', title='Time (seconds)'),
        yaxis=dict(gridcolor='#2D3748', title='Confidence Score', range=[0, 105]),
        margin=dict(t=50, b=40, l=50, r=20),
        height=350,
    )
    return fig


def gauge_chart(value, title, low_color='#2ECC71', high_color='#E74C3C'):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16, 'color': '#E2E8F0'}},
        number={'font': {'size': 36, 'color': '#E2E8F0'}},
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor='#E2E8F0',
                      tickfont=dict(color='#E2E8F0')),
            bar=dict(color='#4ECDC4'),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='#2D3748',
            steps=[
                dict(range=[0, 30], color='rgba(46,204,113,0.25)'),
                dict(range=[30, 60], color='rgba(243,156,18,0.25)'),
                dict(range=[60, 100], color='rgba(231,76,60,0.25)'),
            ],
            threshold=dict(
                line=dict(color='white', width=3),
                thickness=0.8,
                value=value
            )
        )
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E2E8F0'),
        margin=dict(t=40, b=10, l=20, r=20),
        height=250,
    )
    return fig


def candidate_comparison_chart(sessions_df: pd.DataFrame):
    """Bar chart comparing multiple candidates."""
    if sessions_df.empty:
        return go.Figure()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Confidence Score', 'Stress Score'])

    fig.add_trace(go.Bar(
        x=sessions_df['candidate_name'],
        y=sessions_df['confidence_score'],
        marker_color='#4ECDC4',
        name='Confidence',
        hovertemplate='%{x}<br>Confidence: %{y:.1f}<extra></extra>'
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=sessions_df['candidate_name'],
        y=sessions_df['stress_score'],
        marker_color='#FF6B6B',
        name='Stress',
        hovertemplate='%{x}<br>Stress: %{y:.1f}<extra></extra>'
    ), row=1, col=2)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,17,23,0.6)',
        font=dict(color='#E2E8F0'),
        showlegend=False,
        margin=dict(t=50, b=40),
        height=380,
    )
    for i in [1, 2]:
        fig.update_xaxes(gridcolor='#2D3748', row=1, col=i)
        fig.update_yaxes(gridcolor='#2D3748', row=1, col=i, range=[0, 100])

    return fig
