import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "interview_platform.db")

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT,
            position TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id INTEGER,
            session_name TEXT,
            start_time TEXT,
            end_time TEXT,
            duration_seconds INTEGER,
            stress_score REAL,
            confidence_score REAL,
            dominant_emotion TEXT,
            total_frames INTEGER,
            FOREIGN KEY(candidate_id) REFERENCES candidates(id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS emotion_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            timestamp_sec REAL,
            emotion TEXT,
            confidence REAL,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
    ''')

    conn.commit()
    conn.close()

def add_candidate(name, email="", position=""):
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO candidates (name, email, position) VALUES (?, ?, ?)", (name, email, position))
    cid = c.lastrowid
    conn.commit()
    conn.close()
    return cid

def get_all_candidates():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM candidates ORDER BY created_at DESC", conn)
    conn.close()
    return df

def create_session(candidate_id, session_name):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO sessions (candidate_id, session_name, start_time) VALUES (?, ?, ?)",
        (candidate_id, session_name, datetime.now().isoformat())
    )
    sid = c.lastrowid
    conn.commit()
    conn.close()
    return sid

def close_session(session_id, stress, confidence, dominant, total_frames, start_time):
    end_time = datetime.now()
    start_dt = datetime.fromisoformat(start_time)
    duration = int((end_time - start_dt).total_seconds())
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        UPDATE sessions SET end_time=?, duration_seconds=?, stress_score=?,
        confidence_score=?, dominant_emotion=?, total_frames=? WHERE id=?
    ''', (end_time.isoformat(), duration, stress, confidence, dominant, total_frames, session_id))
    conn.commit()
    conn.close()

def log_emotion(session_id, timestamp_sec, emotion, confidence):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO emotion_logs (session_id, timestamp_sec, emotion, confidence) VALUES (?, ?, ?, ?)",
        (session_id, timestamp_sec, emotion, confidence)
    )
    conn.commit()
    conn.close()

def get_session_logs(session_id):
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM emotion_logs WHERE session_id=? ORDER BY timestamp_sec",
        conn, params=(session_id,)
    )
    conn.close()
    return df

def get_candidate_sessions(candidate_id):
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM sessions WHERE candidate_id=? ORDER BY start_time DESC",
        conn, params=(candidate_id,)
    )
    conn.close()
    return df

def get_all_sessions_with_candidates():
    conn = get_connection()
    df = pd.read_sql_query('''
        SELECT s.*, c.name as candidate_name, c.position
        FROM sessions s JOIN candidates c ON s.candidate_id = c.id
        ORDER BY s.start_time DESC
    ''', conn)
    conn.close()
    return df

def reset_all_data():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("DELETE FROM emotion_logs")
    cur.execute("DELETE FROM sessions")
    cur.execute("DELETE FROM candidates")

    conn.commit()
    conn.close()
