import os
import re
import json
import time
import sqlite3
import uuid
import hashlib
import textwrap
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    import cv2
except Exception:
    cv2 = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

st.set_page_config(
    page_title="Neural Search Relevance Optimization Engine",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Constants and synthetic domain configuration
# -----------------------------------------------------------------------------

SEED = 20261106
NOW = datetime(2026, 11, 6, 12, 0, 0, tzinfo=timezone.utc)

ARTIFACT_DIR = os.path.join(tempfile.gettempdir(), "neural_search_relevance_engine")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

DB_PATH = os.path.join(ARTIFACT_DIR, "nsro_demo.sqlite")
VIDEO_MP4 = os.path.join(ARTIFACT_DIR, "nsro_live_demo.mp4")
VIDEO_AVI = os.path.join(ARTIFACT_DIR, "nsro_live_demo.avi")

FEATURES = [
    "tfidf_cosine",
    "title_overlap",
    "body_overlap",
    "exact_title",
    "category_match",
    "brand_match",
    "freshness",
    "popularity",
    "legacy_ctr",
]

CATEGORIES = [
    "Laptops",
    "Monitors",
    "Keyboards",
    "Mice",
    "Headsets",
    "Software",
    "Support",
]

BRANDS = ["Northwind", "Contoso", "Fabrikam", "Litware", "Adventure"]

CATALOG = {
    "Laptops": [
        "Ultrabook 14",
        "Gaming Laptop 15",
        "Business Notebook 13",
        "Creator Laptop 16",
    ],
    "Monitors": [
        "4K Monitor 27",
        "Ultrawide Monitor 34",
        "Office Monitor 24",
        "Design Monitor 32",
    ],
    "Keyboards": [
        "Mechanical Keyboard",
        "Wireless Keyboard",
        "Split Ergonomic Keyboard",
        "Quiet Office Keyboard",
    ],
    "Mice": [
        "Vertical Mouse",
        "Gaming Mouse",
        "Silent Mouse",
        "Travel Mouse",
    ],
    "Headsets": [
        "Noise Cancelling Headset",
        "USB Meeting Headset",
        "Open Back Gaming Headset",
        "Wireless Office Headset",
    ],
    "Software": [
        "Project Tracker",
        "CRM Suite",
        "Endpoint Security",
        "Knowledge Base",
    ],
    "Support": [
        "Password Reset Guide",
        "VPN Troubleshooting",
        "Onboarding Documentation",
        "Return Policy Article",
    ],
}

ADJECTIVES = {
    "Laptops": [
        "lightweight",
        "business",
        "gaming",
        "creator",
        "travel",
        "14 inch",
        "15 inch",
        "rgb",
    ],
    "Monitors": [
        "4k",
        "ultrawide",
        "office",
        "design",
        "usb-c",
        "hdmi",
        "high refresh",
        "budget",
    ],
    "Keyboards": [
        "mechanical",
        "quiet",
        "wireless",
        "split",
        "ergonomic",
        "bluetooth",
        "rgb",
        "office",
    ],
    "Mice": [
        "vertical",
        "gaming",
        "silent",
        "ergonomic",
        "bluetooth",
        "low latency",
        "travel",
        "office",
    ],
    "Headsets": [
        "noise cancelling",
        "usb",
        "open back",
        "meetings",
        "gaming",
        "wireless",
        "mic",
        "office",
    ],
    "Software": [
        "project management",
        "crm",
        "cybersecurity",
        "endpoint",
        "license",
        "cloud",
        "small business",
        "knowledge base",
    ],
    "Support": [
        "password",
        "vpn",
        "troubleshooting",
        "onboarding",
        "return policy",
        "battery",
        "no signal",
        "pairing",
        "cursor",
        "remote work",
    ],
}

CATEGORY_KEYWORDS = {
    "Laptops": ["laptop", "notebook", "ultrabook", "creator", "gaming"],
    "Monitors": ["monitor", "display", "ultrawide", "4k", "office"],
    "Keyboards": ["keyboard", "mechanical", "split", "quiet", "wireless"],
    "Mice": ["mouse", "vertical", "gaming", "silent", "ergonomic"],
    "Headsets": ["headset", "headphone", "mic", "noise", "meeting"],
    "Software": ["software", "crm", "tracker", "security", "license", "cloud"],
    "Support": [
        "how",
        "guide",
        "troubleshooting",
        "policy",
        "reset",
        "vpn",
        "onboarding",
        "battery",
        "signal",
        "pairing",
        "cursor",
        "remote",
    ],
}

QUERY_SPECS = [
    ("lightweight business laptop", "Laptops", ["lightweight", "business", "laptop"]),
    ("gaming laptop with rgb", "Laptops", ["gaming", "laptop", "rgb"]),
    ("14 inch laptop for travel", "Laptops", ["14", "inch", "laptop", "travel"]),
    ("creator laptop 16 inch", "Laptops", ["creator", "laptop", "16"]),
    ("4k monitor for design", "Monitors", ["4k", "monitor", "design"]),
    ("ultrawide monitor productivity", "Monitors", ["ultrawide", "monitor", "productivity"]),
    ("budget office monitor", "Monitors", ["budget", "office", "monitor"]),
    ("usb-c monitor for laptop", "Monitors", ["usb-c", "monitor", "laptop"]),
    ("quiet mechanical keyboard", "Keyboards", ["quiet", "mechanical", "keyboard"]),
    ("wireless keyboard for laptop", "Keyboards", ["wireless", "keyboard", "laptop"]),
    ("ergonomic split keyboard", "Keyboards", ["ergonomic", "split", "keyboard"]),
    ("bluetooth keyboard office", "Keyboards", ["bluetooth", "keyboard", "office"]),
    ("vertical mouse wrist pain", "Mice", ["vertical", "mouse", "wrist"]),
    ("gaming mouse low latency", "Mice", ["gaming", "mouse", "latency"]),
    ("silent mouse office", "Mice", ["silent", "mouse", "office"]),
    ("travel mouse bluetooth", "Mice", ["travel", "mouse", "bluetooth"]),
    ("noise cancelling headset calls", "Headsets", ["noise", "cancelling", "headset", "calls"]),
    ("usb headset for meetings", "Headsets", ["usb", "headset", "meetings"]),
    ("open back headset gaming", "Headsets", ["open", "back", "headset", "gaming"]),
    ("wireless office headset mic", "Headsets", ["wireless", "office", "headset", "mic"]),
    ("project management software", "Software", ["project", "management", "software"]),
    ("crm for small business", "Software", ["crm", "small", "business", "software"]),
    ("endpoint cybersecurity software", "Software", ["endpoint", "cybersecurity", "software"]),
    ("knowledge base software license", "Software", ["knowledge", "base", "software", "license"]),
    ("how to reset password", "Support", ["how", "reset", "password"]),
    ("vpn troubleshooting guide", "Support", ["vpn", "troubleshooting", "guide"]),
    ("employee onboarding documentation", "Support", ["onboarding", "documentation"]),
    ("return policy support article", "Support", ["return", "policy", "support"]),
    ("laptop battery drain fix", "Support", ["laptop", "battery", "drain", "fix"]),
    ("monitor no signal troubleshooting", "Support", ["monitor", "no", "signal", "troubleshooting"]),
    ("keyboard pairing bluetooth issue", "Support", ["keyboard", "pairing", "bluetooth"]),
    ("mouse cursor drifting fix", "Support", ["mouse", "cursor", "drifting"]),
    ("software license transfer help", "Support", ["software", "license", "transfer"]),
    ("remote work equipment policy", "Support", ["remote", "work", "equipment", "policy"]),
]

FEEDBACK_COLS = [
    "id",
    "ts",
    "request_id",
    "query_id",
    "query_text",
    "doc_id",
    "label",
    "note",
    "actor",
]

AUDIT_COLS = ["id", "ts", "action", "details", "actor", "correlation_id"]


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-np.clip(float(x), -50, 50))))


def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def infer_category(query):
    qtokens = set(tokenize(query))
    scores = {c: 0 for c in CATEGORIES}
    for cat, words in CATEGORY_KEYWORDS.items():
        for w in words:
            if w in qtokens:
                scores[cat] += 1
    for cat in CATEGORIES:
        if cat.lower() in qtokens:
            scores[cat] += 2
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def infer_brand(query):
    ql = str(query).lower()
    for b in BRANDS:
        if b.lower() in ql:
            return b
    return None


def empty_df(cols):
    return pd.DataFrame(columns=cols)


def trim_session_list(key, max_len):
    if key in st.session_state and isinstance(st.session_state[key], list):
        if len(st.session_state[key]) > max_len:
            st.session_state[key] = st.session_state[key][-max_len:]


# -----------------------------------------------------------------------------
# Synthetic data generation
# -----------------------------------------------------------------------------


def make_docs(rng):
    rows = []
    for i in range(120):
        cat = CATEGORIES[i % len(CATEGORIES)]
        subtype = str(rng.choice(CATALOG[cat]))
        adj_list = ADJECTIVES[cat]
        adj1, adj2 = [str(x) for x in rng.choice(adj_list, size=2, replace=False)]
        brand = str(rng.choice(BRANDS))

        title = f"{brand} {subtype} {adj1}"
        feature = str(rng.choice(["Fast setup", "Enterprise ready", "Remote friendly", "Budget option", "Premium build"]))
        support_channel = str(rng.choice(["chat", "email", "24x7", "community"]))
        body = (
            f"{title}. {subtype} for {cat.lower()} with {adj1}, {adj2}. "
            f"{feature}. Support: {support_channel}. "
            f"Category {cat}. Brand {brand}."
        )

        published = NOW - timedelta(days=int(rng.integers(120, 900)))
        updated = published + timedelta(days=int(rng.integers(0, 90)))
        if updated > NOW:
            updated = NOW
        age_days = max(0, int((NOW - updated).days))

        quality = float(rng.beta(2.2, 4.5))
        legacy_ctr = float(np.clip(0.015 + 0.12 * quality + rng.normal(0, 0.006), 0.002, 0.22))
        popularity = int(rng.integers(0, 100))

        rows.append(
            {
                "doc_id": f"D{i:04d}",
                "title": title,
                "body": body,
                "category": cat,
                "brand": brand,
                "published_at": published.isoformat(),
                "updated_at": updated.isoformat(),
                "age_days": age_days,
                "popularity": popularity,
                "legacy_ctr": legacy_ctr,
                "source": "synthetic_fictional_catalog",
                "consent": "synthetic_no_pii",
                "retention_days": 90,
                "schema_version": "1.0.0",
            }
        )
    return pd.DataFrame(rows)


def make_queries(docs, rng):
    rows = []
    gold_rows = []

    indices = list(range(len(QUERY_SPECS)))
    rng.shuffle(indices)
    n_train = int(0.7 * len(indices))
    split_map = {idx: ("train" if pos < n_train else "test") for pos, idx in enumerate(indices)}

    doc_token_map = {row.doc_id: set(tokenize(f"{row.title} {row.body}")) for row in docs.itertuples()}
    title_token_map = {row.doc_id: set(tokenize(row.title)) for row in docs.itertuples()}

    for idx, (text, cat, tokens) in enumerate(QUERY_SPECS):
        qid = f"Q{idx:03d}"
        split = split_map[idx]
        rows.append(
            {
                "query_id": qid,
                "query_text": text,
                "category": cat,
                "query_split": split,
                "tokens": " ".join(tokens),
                "schema_version": "1.0.0",
            }
        )

        token_set = set(tokens)
        scored = []
        for d in docs.itertuples():
            overlap = len(token_set & doc_token_map[d.doc_id])
            cat_bonus = 2 if d.category == cat else 0
            title_bonus = 1 if any(t in title_token_map[d.doc_id] for t in tokens) else 0
            score = overlap + cat_bonus + title_bonus
            if score >= 4:
                rel = 2
            elif score >= 2:
                rel = 1
            else:
                continue
            scored.append((d.doc_id, rel, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        existing = {s[0] for s in scored}

        if len(scored) < 3:
            for d in docs[docs.category == cat].itertuples():
                if len(scored) >= 3:
                    break
                if d.doc_id not in existing:
                    scored.append((d.doc_id, 1, 1))
                    existing.add(d.doc_id)
            scored.sort(key=lambda x: x[2], reverse=True)

        for doc_id, rel, score in scored[:8]:
            gold_rows.append(
                {
                    "query_id": qid,
                    "doc_id": doc_id,
                    "relevance": int(rel),
                    "gold_source": "synthetic_rule_based",
                    "schema_version": "1.0.0",
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(gold_rows, columns=["query_id", "doc_id", "relevance", "gold_source", "schema_version"])


def make_events(docs, queries, gold, rng):
    doc_by_id = docs.set_index("doc_id")
    doc_token_map = {row.doc_id: set(tokenize(f"{row.title} {row.body}")) for row in docs.itertuples()}
    rows = []
    eid = 0

    for q in queries.itertuples():
        gold_map = gold[gold.query_id == q.query_id].set_index("doc_id")["relevance"].to_dict()
        n_sessions = 25 if q.query_split == "train" else 10
        qtokens = set(tokenize(q.query_text))

        cand_scores = []
        for d in docs.itertuples():
            overlap = len(qtokens & doc_token_map[d.doc_id])
            catb = 1 if d.category == q.category else 0
            cand_scores.append((d.doc_id, overlap + catb + rng.random() * 0.4))
        cand_scores.sort(key=lambda x: x[1], reverse=True)
        base_shown = [x[0] for x in cand_scores[:12]]

        for _ in range(n_sessions):
            if q.query_split == "train":
                ts = NOW - timedelta(days=int(rng.integers(8, 36)), hours=int(rng.integers(0, 24)))
            else:
                ts = NOW - timedelta(days=int(rng.integers(0, 7)), hours=int(rng.integers(0, 24)))

            shown = base_shown.copy()
            rng.shuffle(shown)

            gold_ids = list(gold_map.keys())
            if gold_ids and rng.random() < 0.75:
                chosen = str(rng.choice(gold_ids))
                if chosen in shown:
                    shown.remove(chosen)
                shown[0] = chosen

            for pos, doc_id in enumerate(shown[:10], start=1):
                rel = int(gold_map.get(doc_id, 0))
                legacy = float(doc_by_id.at[doc_id, "legacy_ctr"])
                z = 1.45 * rel - 0.42 * pos + 9.0 * (legacy - 0.06) + float(rng.normal(0, 0.45))
                click_p = sigmoid(z)
                clicked = int(rng.random() < click_p)
                converted = int(clicked == 1 and rel == 2 and rng.random() < 0.35)
                dwell = int(np.clip(rng.normal(12 + 8 * rel - 1.5 * pos, 4), 1, 90)) if clicked else 0

                feedback = ""
                if rng.random() < 0.07:
                    if rel > 0 and rng.random() < 0.75:
                        feedback = "relevant"
                    elif rng.random() < 0.65:
                        feedback = "irrelevant"
                    else:
                        feedback = "uncertain"

                label = int(clicked == 1 or feedback == "relevant")

                rows.append(
                    {
                        "event_id": f"E{eid:06d}",
                        "timestamp": ts.isoformat(),
                        "query_id": q.query_id,
                        "query_text": q.query_text,
                        "query_split": q.query_split,
                        "doc_id": doc_id,
                        "position": pos,
                        "clicked": clicked,
                        "converted": converted,
                        "dwell_seconds": dwell,
                        "feedback_label": feedback,
                        "label": label,
                        "source": "synthetic_behavior_log",
                        "schema_version": "1.0.0",
                    }
                )
                eid += 1

    return pd.DataFrame(rows)


def data_quality(docs, queries, events, gold):
    issues = []

    def add(severity, check, detail):
        issues.append({"severity": severity, "check": check, "detail": str(detail)})

    expected = {
        "docs": ["doc_id", "title", "body", "category", "brand", "age_days", "popularity", "legacy_ctr"],
        "queries": ["query_id", "query_text", "category", "query_split"],
        "events": ["event_id", "timestamp", "query_id", "doc_id", "position", "clicked", "label", "query_split"],
        "gold": ["query_id", "doc_id", "relevance"],
    }

    for name, df in {"docs": docs, "queries": queries, "events": events, "gold": gold}.items():
        missing = [c for c in expected[name] if c not in df.columns]
        if missing:
            add("critical", f"{name}_schema", f"Missing columns: {missing}")
        if df.empty:
            add("critical", f"{name}_empty", "DataFrame is empty")

    if not docs.empty and docs.doc_id.duplicated().any():
        add("critical", "duplicate_doc_id", "Duplicate document identifiers found")
    if not queries.empty and queries.query_id.duplicated().any():
        add("critical", "duplicate_query_id", "Duplicate query identifiers found")
    if not events.empty and events.event_id.duplicated().any():
        add("critical", "duplicate_event_id", "Duplicate event identifiers found")

    if not gold.empty and not gold.relevance.between(0, 2).all():
        add("critical", "invalid_gold_range", "Gold relevance must be 0, 1, or 2")
    if not events.empty and not events.clicked.isin([0, 1]).all():
        add("critical", "invalid_clicked_range", "clicked must be binary")
    if not events.empty and not events.label.isin([0, 1]).all():
        add("critical", "invalid_label_range", "label must be binary")
    if not events.empty and not (events.position > 0).all():
        add("warning", "invalid_position", "Position must be positive")

    train_q = set(queries[queries.query_split == "train"].query_id)
    test_q = set(queries[queries.query_split == "test"].query_id)
    overlap = train_q & test_q
    if overlap:
        add("critical", "query_leakage", f"Query IDs appear in both train and test: {list(overlap)[:5]}")

    if not events.empty:
        event_times = pd.to_datetime(events.timestamp, utc=True).dt.tz_localize(None)
        train_times = event_times[events.query_split == "train"]
        test_times = event_times[events.query_split == "test"]
        if len(train_times) and len(test_times):
            if train_times.max() > test_times.min():
                add("warning", "temporal_leakage_risk", "Some train events are newer than test events")

    if not docs.empty:
        skew_val = float(pd.Series(docs.popularity).skew())
        if abs(skew_val) > 2.0:
            add("info", "numeric_skew", f"Popularity skew={skew_val:.2f}; inspect distribution")

    if not issues:
        add("info", "quality_summary", "No critical or warning issues detected in synthetic data")

    return pd.DataFrame(issues, columns=["severity", "check", "detail"])


@st.cache_data(show_spinner=False)
def load_dataset():
    rng = np.random.default_rng(SEED)
    docs = make_docs(rng)
    queries, gold = make_queries(docs, rng)
    events = make_events(docs, queries, gold, rng)
    quality = data_quality(docs, queries, events, gold)
    return docs, queries, events, gold, quality


# -----------------------------------------------------------------------------
# Persistence: SQLite for feedback, audit, model registry metadata
# -----------------------------------------------------------------------------


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            request_id TEXT UNIQUE NOT NULL,
            query_id TEXT NOT NULL,
            query_text TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            label TEXT NOT NULL,
            note TEXT,
            actor TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT,
            actor TEXT NOT NULL,
            correlation_id TEXT
        );

        CREATE TABLE IF NOT EXISTS model_versions (
            version TEXT PRIMARY KEY,
            created_ts TEXT NOT NULL,
            train_rows INTEGER,
            champion INTEGER DEFAULT 0,
            metrics_json TEXT
        );
        """
    )
    cutoff = (NOW - timedelta(days=90)).isoformat()
    conn.execute("DELETE FROM feedback WHERE ts < ?", (cutoff,))
    conn.commit()
    conn.close()


def load_table(name, cols):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(f"SELECT * FROM {name} ORDER BY ts DESC", conn)
        conn.close()
        return df
    except Exception:
        return empty_df(cols)


def insert_feedback(query_id, query_text, doc_id, label, note, actor, request_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM feedback WHERE request_id=?", (request_id,))
    if cur.fetchone():
        conn.close()
        return False
    ts = utc_now_iso()
    cur.execute(
        """
        INSERT INTO feedback (ts, request_id, query_id, query_text, doc_id, label, note, actor)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (ts, request_id, query_id, query_text, doc_id, label, note or "", actor),
    )
    conn.commit()
    conn.close()
    return True


def append_audit(action, details="", actor="system", correlation_id=None):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO audit_log (ts, action, details, actor, correlation_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                action,
                str(details),
                actor,
                correlation_id or uuid.uuid4().hex[:12],
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def save_model_version(version, train_rows, champion, metrics):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        metrics_clean = {}
        for k, v in dict(metrics).items():
            try:
                fv = float(v)
                metrics_clean[str(k)] = None if np.isnan(fv) else fv
            except Exception:
                metrics_clean[str(k)] = None
        cur.execute(
            """
            INSERT OR REPLACE INTO model_versions (version, created_ts, train_rows, champion, metrics_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (version, utc_now_iso(), int(train_rows), int(champion), json.dumps(metrics_clean)),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Vectorization and feature engineering
# -----------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_corpus_vectorizer():
    docs, _, _, _, _ = load_dataset()
    corpus = (docs.title.str.lower() + " " + docs.body.str.lower()).tolist()
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=8000, min_df=1)
    mat = vec.fit_transform(corpus)
    return vec, mat, docs.set_index("doc_id")


def compute_features(query_text, docs_index, vec, mat, query_category=None, query_brand=None):
    qtokens = set(tokenize(query_text))
    qvec = vec.transform([str(query_text).lower()])
    sims = cosine_similarity(qvec, mat).ravel()

    data = []
    for i, (doc_id, row) in enumerate(docs_index.iterrows()):
        ttoks = set(tokenize(row.title))
        btoks = set(tokenize(row.body))

        union_t = qtokens | ttoks
        union_b = qtokens | btoks
        title_overlap = len(qtokens & ttoks) / len(union_t) if union_t else 0.0
        body_overlap = len(qtokens & btoks) / len(union_b) if union_b else 0.0

        qt = str(query_text).lower().strip()
        exact = 1.0 if qt and qt in str(row.title).lower() else 0.0
        cat = 1.0 if query_category and row.category == query_category else 0.0
        brand = 1.0 if query_brand and row.brand == query_brand else 0.0
        fresh = float(np.exp(-max(0, int(row.age_days)) / 240.0))
        pop = float(row.popularity) / 100.0
        legacy = float(row.legacy_ctr)

        data.append(
            [
                float(sims[i]),
                float(title_overlap),
                float(body_overlap),
                float(exact),
                float(cat),
                float(brand),
                float(fresh),
                float(pop),
                float(legacy),
            ]
        )

    return pd.DataFrame(data, columns=FEATURES, index=docs_index.index)


def candidate_pool(query_text, docs_index, vec, mat, candidate_k=50, gold_ids=None):
    qc = infer_category(query_text)
    qb = infer_brand(query_text)
    feats = compute_features(query_text, docs_index, vec, mat, qc, qb)
    feats["baseline_score"] = (
        feats["tfidf_cosine"] + 0.20 * feats["title_overlap"] + 0.05 * feats["exact_title"]
    )

    pool = feats.sort_values("baseline_score", ascending=False).head(int(candidate_k)).copy()

    if gold_ids:
        missing = [g for g in gold_ids if g not in pool.index and g in feats.index]
        if missing:
            pool = pd.concat([pool, feats.loc[missing]])

    pool = pool.sort_values("baseline_score", ascending=False)
    pool["baseline_rank"] = np.arange(1, len(pool) + 1)
    return pool, feats


def rank_results(query_text, model=None, top_k=10, candidate_k=50, gold_ids=None):
    vec, mat, docs_index = get_corpus_vectorizer()
    pool, feats = candidate_pool(query_text, docs_index, vec, mat, candidate_k=candidate_k, gold_ids=gold_ids)

    use_ai = model is not None
    if use_ai:
        try:
            X = model.named_steps["scaler"].transform(pool[FEATURES])
            proba = model.named_steps["clf"].predict_proba(X)
            if proba.shape[1] < 2:
                raise ValueError("Model does not expose binary probabilities")
            pool["ai_score"] = proba[:, 1]
        except Exception:
            use_ai = False

    if not use_ai:
        pool["ai_score"] = pool["baseline_score"]

    pool = pool.sort_values(["ai_score", "baseline_score"], ascending=False).reset_index()
    if "doc_id" not in pool.columns:
        pool = pool.rename(columns={pool.columns[0]: "doc_id"})

    pool["final_rank"] = np.arange(1, len(pool) + 1)

    meta = docs_index[["title", "category", "brand", "age_days", "legacy_ctr"]].reset_index()
    out = pool.merge(meta, on="doc_id", how="left")
    return out.head(int(top_k)), feats


# -----------------------------------------------------------------------------
# Model training, evaluation, explanation
# -----------------------------------------------------------------------------


def train_relevance_model(events, docs_index, vec, mat, extra_feedback=None):
    train_events = events[events.query_split == "train"].copy()
    if extra_feedback is not None and not extra_feedback.empty:
        train_events = pd.concat([train_events, extra_feedback], ignore_index=True)

    train_events["label"] = train_events.apply(
        lambda row: int(row.get("clicked", 0) == 1 or row.get("feedback_label", "") == "relevant"),
        axis=1,
    )
    train_events = train_events.drop_duplicates(subset=["query_text", "doc_id"], keep="last")

    X = []
    y = []

    for qtext, grp in train_events.groupby("query_text"):
        feats = compute_features(
            qtext,
            docs_index,
            vec,
            mat,
            query_category=infer_category(qtext),
            query_brand=infer_brand(qtext),
        )
        for doc_id, label in zip(grp.doc_id, grp.label):
            if doc_id in feats.index:
                X.append(feats.loc[doc_id].values)
                y.append(int(label))

    if not X:
        return None, "empty-training"

    X = np.array(X)
    y = np.array(y)

    if len(np.unique(y)) < 2:
        return None, "single-class"

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=1.0,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipe.fit(X, y)

    coef_sum = float(pipe.named_steps["clf"].coef_.sum())
    version = hashlib.sha256(f"{len(y)}|{coef_sum}|{SEED}".encode()).hexdigest()[:10]
    return pipe, version


@st.cache_resource(show_spinner=False)
def get_initial_model():
    _, _, events, _, _ = load_dataset()
    vec, mat, docs_index = get_corpus_vectorizer()
    model, version = train_relevance_model(events, docs_index, vec, mat)
    if model is None:
        version = "baseline-fallback"
    return model, version


def dcg_at_k(rels, k):
    rels = list(rels)[:k]
    return sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(rels, ideal, k):
    ideal_top = sorted(ideal, reverse=True)[:k]
    idcg = dcg_at_k(ideal_top, k)
    return dcg_at_k(rels, k) / idcg if idcg > 0 else 0.0


def bootstrap_ci(values, n_boot=200, alpha=0.05):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boots.append(float(sample.mean()))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def evaluate_ranker(model, k=5):
    _, queries, _, gold, _ = load_dataset()
    test_q = queries[queries.query_split == "test"]
    rows = []

    for _, q in test_q.iterrows():
        gold_map = gold[gold.query_id == q.query_id].set_index("doc_id")["relevance"].to_dict()
        ranked, _ = rank_results(
            q.query_text,
            model=model,
            top_k=20,
            candidate_k=50,
            gold_ids=list(gold_map.keys()),
        )
        rels = [int(gold_map.get(d, 0)) for d in ranked.doc_id]
        ideal = list(gold_map.values())

        first_pos = next((i + 1 for i, r in enumerate(rels) if r > 0), np.nan)
        mrr = 0.0 if np.isnan(first_pos) else 1.0 / float(first_pos)
        retrieved_rel = sum(1 for r in rels[:k] if r > 0)
        relevant_total = sum(1 for r in ideal if r > 0)

        rows.append(
            {
                "query_id": q.query_id,
                "query_text": q.query_text,
                "category": q.category,
                f"ndcg@{k}": ndcg_at_k(rels, ideal, k),
                f"recall@{k}": retrieved_rel / relevant_total if relevant_total else 0.0,
                f"precision@{k}": retrieved_rel / float(k),
                "mrr": mrr,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.Series(dtype=float), (np.nan, np.nan)

    mean = df.mean(numeric_only=True)
    ci = bootstrap_ci(df[f"ndcg@{k}"].values)
    return df, mean, ci


def min_subgroup_ndcg(eval_df, k=5):
    col = f"ndcg@{k}"
    if eval_df is None or eval_df.empty or col not in eval_df.columns:
        return 0.0
    return float(eval_df.groupby("category")[col].mean().min())


def feedback_to_training_rows(fb_df, exclude_texts):
    if fb_df is None or fb_df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in fb_df.iterrows():
        label = str(r.get("label", "")).lower()
        if label not in ["relevant", "irrelevant"]:
            continue
        qt = str(r.get("query_text", ""))
        if qt.lower() in exclude_texts:
            continue
        rows.append(
            {
                "query_text": qt,
                "doc_id": str(r.get("doc_id", "")),
                "clicked": 1 if label == "relevant" else 0,
                "feedback_label": label,
                "query_split": "feedback",
            }
        )
    return pd.DataFrame(rows)


def train_candidate_model(fb_df):
    _, queries, events, _, _ = load_dataset()
    vec, mat, docs_index = get_corpus_vectorizer()
    exclude = set(queries[queries.query_split == "test"].query_text.str.lower())
    fb_rows = feedback_to_training_rows(fb_df, exclude)
    model, version = train_relevance_model(
        events,
        docs_index,
        vec,
        mat,
        extra_feedback=fb_rows if not fb_rows.empty else None,
    )
    return model, version, len(fb_rows)


def calibration_report(model, bins=10):
    _, _, events, _, _ = load_dataset()
    if model is None:
        return pd.DataFrame(), {}, pd.DataFrame()

    test_events = events[events.query_split == "test"].copy()
    rows = []

    for qtext, grp in test_events.groupby("query_text"):
        grp = grp.drop_duplicates(subset=["doc_id"], keep="last")
        vec, mat, docs_index = get_corpus_vectorizer()
        feats = compute_features(
            qtext,
            docs_index,
            vec,
            mat,
            query_category=infer_category(qtext),
            query_brand=infer_brand(qtext),
        )
        valid = [d for d in grp.doc_id if d in feats.index]
        if not valid:
            continue
        X = model.named_steps["scaler"].transform(feats.loc[valid, FEATURES])
        proba = model.named_steps["clf"].predict_proba(X)
        if proba.shape[1] < 2:
            continue
        scores = proba[:, 1]
        labels = grp.loc[grp.doc_id.isin(valid), "label"].values
        for doc_id, score, label in zip(valid, scores, labels):
            rows.append({"query_text": qtext, "doc_id": doc_id, "score": float(score), "label": int(label)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df, {}, pd.DataFrame()

    metrics = {}
    try:
        if df.label.nunique() >= 2:
            metrics["brier"] = float(brier_score_loss(df.label, df.score))
            metrics["log_loss"] = float(log_loss(df.label, np.clip(df.score, 1e-6, 1 - 1e-6), labels=[0, 1]))
            metrics["auc"] = float(roc_auc_score(df.label, df.score))
    except Exception:
        pass

    cal = pd.DataFrame()
    try:
        df["bin"] = pd.cut(df.score, bins=np.linspace(0, 1, bins + 1), include_lowest=True)
        cal = (
            df.groupby("bin", observed=True)
            .agg(pred=("score", "mean"), obs=("label", "mean"), n=("label", "size"))
            .reset_index()
        )
        cal["pred"] = cal["pred"].astype(float)
        cal["obs"] = cal["obs"].astype(float)
    except Exception:
        cal = pd.DataFrame()

    return df, metrics, cal


def failure_analysis(model, k=5):
    _, queries, _, gold, _ = load_dataset()
    if model is None:
        empty = pd.DataFrame(columns=["query_text", "doc_id", "title", "score", "gold_rel", "rank"])
        return empty, empty.copy(), empty.copy()

    _, _, docs_index_local = get_corpus_vectorizer()
    test_q = queries[queries.query_split == "test"]
    fp, fn, unc = [], [], []

    for _, q in test_q.iterrows():
        gold_map = gold[gold.query_id == q.query_id].set_index("doc_id")["relevance"].to_dict()
        ranked, _ = rank_results(
            q.query_text,
            model=model,
            top_k=20,
            candidate_k=50,
            gold_ids=list(gold_map.keys()),
        )
        for _, r in ranked.iterrows():
            rel = int(gold_map.get(r.doc_id, 0))
            score = float(r.ai_score)
            title = str(docs_index_local.at[r.doc_id, "title"]) if r.doc_id in docs_index_local.index else ""
            item = {
                "query_text": q.query_text,
                "doc_id": r.doc_id,
                "title": title,
                "score": score,
                "gold_rel": rel,
                "rank": int(r.final_rank),
            }
            if score >= 0.65 and rel == 0:
                fp.append(item)
            if rel > 0 and int(r.final_rank) > k:
                fn.append(item)
            if 0.40 <= score <= 0.60:
                unc.append(item)

    cols = ["query_text", "doc_id", "title", "score", "gold_rel", "rank"]
    return (
        pd.DataFrame(fp, columns=cols).head(20),
        pd.DataFrame(fn, columns=cols).head(20),
        pd.DataFrame(unc, columns=cols).head(20),
    )


def explain_doc(query_text, doc_id, model, feats):
    _, _, docs_index = get_corpus_vectorizer()
    if doc_id not in feats.index:
        return pd.Series(dtype=float), set()

    matched = set(tokenize(query_text)) & set(tokenize(f"{docs_index.at[doc_id, 'title']} {docs_index.at[doc_id, 'body']}"))

    if model is None:
        contrib = feats.loc[doc_id].copy()
        contrib.name = "baseline_feature_value"
        return contrib, matched

    X = feats.loc[[doc_id], FEATURES]
    xs = model.named_steps["scaler"].transform(X)
    coef = model.named_steps["clf"].coef_[0]
    contrib = pd.Series(xs[0] * coef, index=FEATURES, name="contribution")
    return contrib.sort_values(), matched


def benchmark_latency(model, n=20):
    _, queries, _, _, _ = load_dataset()
    qs = queries.query_text.head(5).tolist()
    times = []
    for i in range(n):
        q = qs[i % len(qs)]
        t0 = time.perf_counter()
        rank_results(q, model=model, top_k=10, candidate_k=50)
        times.append((time.perf_counter() - t0) * 1000.0)
    s = pd.Series(times)
    return pd.DataFrame(
        {
            "metric": ["p50_ms", "p95_ms", "mean_ms", "max_ms"],
            "value": [float(s.quantile(0.50)), float(s.quantile(0.95)), float(s.mean()), float(s.max())],
        }
    )


# -----------------------------------------------------------------------------
# Synthetic live demo video generation
# -----------------------------------------------------------------------------


def _put(img, text, x, y, scale=0.6, color=(240, 240, 240), thickness=1):
    if cv2 is None:
        return
    cv2.putText(img, str(text), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, float(scale), color, int(thickness), cv2.LINE_AA)


def _wrap_put(img, text, x, y, scale=0.5, color=(220, 220, 220), max_chars=72):
    if cv2 is None:
        return
    line_height = int(24 * scale + 8)
    yy = int(y)
    for line in textwrap.wrap(str(text), max_chars):
        _put(img, line, x, yy, scale, color)
        yy += line_height


def _bar(img, x, y, w, h, pct, color, label, value=""):
    if cv2 is None:
        return
    pct = float(np.clip(pct, 0, 1))
    cv2.rectangle(img, (x, y), (x + w, y + h), (55, 55, 65), -1)
    cv2.rectangle(img, (x, y), (x + int(w * pct), y + h), color, -1)
    _put(img, label, x, y - 8, 0.42, (210, 210, 210))
    if value:
        _put(img, value, x + w + 12, y + h // 2 + 5, 0.42, color)


def generate_demo_video():
    if cv2 is None:
        return None

    cached = st.session_state.get("demo_video_path")
    if cached and os.path.exists(cached) and os.path.getsize(cached) > 1000:
        return cached

    width, height, fps, frames = 960, 540, 20, 120

    baseline_labels = [
        "D001 Lightweight laptop",
        "D014 Business notebook",
        "D022 Gaming laptop",
        "D031 Creator laptop",
        "D044 Travel ultrabook",
    ]
    baseline_scores = [0.82, 0.67, 0.54, 0.41, 0.29]
    ai_labels = [
        "D001 Lightweight laptop",
        "D044 Travel ultrabook",
        "D014 Business notebook",
        "D022 Gaming laptop",
        "D031 Creator laptop",
    ]
    ai_scores = [0.91, 0.78, 0.62, 0.49, 0.35]

    def draw_frame(f):
        img = np.zeros((height, width, 3), np.uint8)

        _put(img, "Neural Search Relevance Optimization Engine", 20, 38, 0.85, (255, 255, 255), 2)
        _put(img, "Synthetic local demo - not production metrics", 20, 62, 0.45, (180, 220, 255))

        scene = min(4, f // 24)
        prog = (f % 24) / 23.0

        if scene == 0:
            _wrap_put(img, "1. Ingest synthetic queries, documents, impressions, clicks, and human corrections.", 40, 130, 0.55)
            for i in range(8):
                x = 90 + i * 95
                y = 280 + int(35 * np.sin((f + i * 12) / 7.0))
                cv2.circle(img, (x, y), 9, (80, 200, 255), -1)
            _put(img, "Event stream", 40, 380, 0.55, (180, 180, 180))

        elif scene == 1:
            _wrap_put(img, "2. Transparent lexical baseline ranks candidates by term overlap and title evidence.", 40, 110, 0.55)
            for i, (lab, sc) in enumerate(zip(baseline_labels, baseline_scores)):
                _bar(img, 80, 170 + i * 58, 560, 28, sc * prog, (255, 150, 50), lab, f"{sc * prog:.2f}")

        elif scene == 2:
            _wrap_put(img, "3. Learning-to-rank optimizer reranks using behavior, freshness, category, and lexical features.", 40, 110, 0.55)
            for i, (lab, sc) in enumerate(zip(ai_labels, ai_scores)):
                _bar(img, 80, 170 + i * 58, 560, 28, sc * prog, (80, 220, 120), lab, f"{sc * prog:.2f}")

        elif scene == 3:
            _wrap_put(img, "4. Explanation and human review: reviewers see evidence, correct mistakes, and approve changes.", 40, 100, 0.55)
            cv2.line(img, (480, 150), (480, 410), (120, 120, 120), 1)
            contribs = [("lexical", 0.72), ("category", 0.48), ("freshness", -0.22), ("legacy_ctr", 0.31), ("title", 0.55)]
            for i, (name, val) in enumerate(contribs):
                y = 175 + i * 42
                _put(img, name, 250, y + 10, 0.45, (210, 210, 210))
                x2 = 480 + int(val * 180 * prog)
                color = (80, 220, 120) if val >= 0 else (70, 90, 250)
                cv2.rectangle(img, (min(480, x2), y), (max(480, x2), y + 20), color, -1)
            _put(img, "Feedback: Like / Dislike / Uncertain", 80, 455, 0.55, (255, 220, 120))

        else:
            _wrap_put(img, "5. Controlled synthetic evaluation compares baseline and candidate before promotion.", 40, 110, 0.55)
            _bar(img, 120, 190, 520, 34, 0.62 * prog, (255, 150, 50), "Baseline NDCG@5", f"{0.62 * prog:.2f}")
            _bar(img, 120, 260, 520, 34, 0.71 * prog, (80, 220, 120), "Candidate NDCG@5", f"{0.71 * prog:.2f}")
            _wrap_put(img, "Illustrative only. Run the Evaluation tab for locally computed synthetic metrics.", 80, 360, 0.50, (180, 220, 255))

        return img

    targets = [
        (VIDEO_MP4, ["mp4v", "avc1"]),
        (VIDEO_AVI, ["XVID", "MJPG"]),
    ]

    for path, codecs in targets:
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            st.session_state["demo_video_path"] = path
            return path

        writer = None
        for cc in codecs:
            try:
                w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), fps, (width, height))
                if w.isOpened():
                    writer = w
                    break
                w.release()
            except Exception:
                continue

        if writer is None:
            continue

        try:
            for f in range(frames):
                writer.write(draw_frame(f))
        finally:
            writer.release()

        if os.path.exists(path) and os.path.getsize(path) > 1000:
            st.session_state["demo_video_path"] = path
            return path

        try:
            os.remove(path)
        except Exception:
            pass

    st.session_state["demo_video_path"] = None
    return None


# -----------------------------------------------------------------------------
# Guided replay helpers
# -----------------------------------------------------------------------------


def run_auto_simulation(container, speed=0.25):
    docs, queries, events, gold, _ = load_dataset()
    model = st.session_state.get("champion_model")
    test_q = queries[queries.query_split == "test"].head(4)

    head = container.empty()
    bl = container.empty()
    ai = container.empty()
    log = container.empty()
    met = container.empty()

    lines = []
    for tick, q in enumerate(test_q.itertuples(), start=1):
        gold_map = gold[gold.query_id == q.query_id].set_index("doc_id")["relevance"].to_dict()
        base, _ = rank_results(q.query_text, model=None, top_k=5, candidate_k=50, gold_ids=list(gold_map.keys()))
        ranked, _ = rank_results(q.query_text, model=model, top_k=5, candidate_k=50, gold_ids=list(gold_map.keys()))

        if base.empty or ranked.empty:
            continue

        head.markdown(f"**Tick {tick}** - `{q.query_text}` - category **{q.category}**")
        bl.dataframe(base[["final_rank", "doc_id", "title", "baseline_score"]], hide_index=True, height=170)
        ai.dataframe(ranked[["final_rank", "doc_id", "title", "ai_score", "baseline_score"]], hide_index=True, height=170)

        clicked_doc = ranked.iloc[1].doc_id if len(ranked) > 1 and tick % 2 == 0 else ranked.iloc[0].doc_id
        lines.append(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} click query={q.query_id} doc={clicked_doc}")
        log.code("\n".join(lines[-5:]))

        base_ndcg = ndcg_at_k([gold_map.get(d, 0) for d in base.doc_id[:5]], list(gold_map.values()), 5)
        ai_ndcg = ndcg_at_k([gold_map.get(d, 0) for d in ranked.doc_id[:5]], list(gold_map.values()), 5)
        met.metric("Synthetic NDCG@5 for this query", f"{ai_ndcg:.3f}", delta=f"{ai_ndcg - base_ndcg:+.3f}")

        time.sleep(max(0.01, float(speed)))


def show_static_simulation_summary(container):
    docs, queries, events, gold, _ = load_dataset()
    qtext = st.session_state.get("last_query", queries.iloc[0].query_text)
    qrow = queries[queries.query_text == qtext]
    gold_map = {}
    if not qrow.empty:
        gold_map = gold[gold.query_id == qrow.iloc[0].query_id].set_index("doc_id")["relevance"].to_dict()

    base, _ = rank_results(qtext, model=None, top_k=5, candidate_k=50, gold_ids=list(gold_map.keys()))
    ranked, _ = rank_results(qtext, model=st.session_state.get("champion_model"), top_k=5, candidate_k=50, gold_ids=list(gold_map.keys()))

    container.markdown(f"**Last analyzed query:** `{qtext}`")

    if base.empty or ranked.empty:
        container.info("No ranking results available for this query.")
        return

    c1, c2 = container.columns(2)
    c1.subheader("Baseline")
    c1.dataframe(base[["final_rank", "doc_id", "title", "baseline_score"]], hide_index=True)
    c2.subheader("AI-enhanced")
    c2.dataframe(ranked[["final_rank", "doc_id", "title", "ai_score", "baseline_score"]], hide_index=True)


def add_feedback_record(doc_id, label, note, query_text):
    if st.session_state.get("role") not in ["Reviewer", "Admin"]:
        st.warning("Current local role is read-only.")
        return False

    qt = query_text or st.session_state.get("last_query", "")
    qmatch = queries[queries.query_text == qt]
    qid = qmatch.iloc[0].query_id if not qmatch.empty else "U" + hashlib.sha256(qt.encode()).hexdigest()[:8]
    rid = hashlib.sha256(f"{qid}|{doc_id}|{label}|{note or ''}".encode()).hexdigest()

    ok = insert_feedback(qid, qt, doc_id, label, note, st.session_state.get("actor", "reviewer"), rid)
    if ok:
        append_audit("feedback", f"{doc_id}={label}", st.session_state.get("actor", "reviewer"), rid)
        st.session_state.feedback_df = load_table("feedback", FEEDBACK_COLS)
        st.session_state.audit_df = load_table("audit_log", AUDIT_COLS)
        st.success(f"Saved {label} feedback for {doc_id}.")
    else:
        st.info("Duplicate feedback request ignored.")
    return ok


def run_self_tests():
    results = []

    def add(name, passed, detail=""):
        results.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    try:
        docs, queries, events, gold, quality = load_dataset()
        add("Dataset loads", len(docs) > 0, f"{len(docs)} docs, {len(events)} events")
        add("No duplicate doc IDs", bool(not docs.doc_id.duplicated().any()))
        add("No duplicate query IDs", bool(not queries.query_id.duplicated().any()))

        train_q = set(queries[queries.query_split == "train"].query_id)
        test_q = set(queries[queries.query_split == "test"].query_id)
        add("Train/test query separation", len(train_q & test_q) == 0)

        base, _ = rank_results("lightweight business laptop", model=None, top_k=5)
        add("Baseline returns finite scores", bool(len(base) == 5 and np.isfinite(base.baseline_score).all()))

        model = st.session_state.get("champion_model")
        if model is not None:
            ai, _ = rank_results("lightweight business laptop", model=model, top_k=5)
            add("Model probabilities bounded", bool(((ai.ai_score >= 0) & (ai.ai_score <= 1)).all()))
        else:
            add("Model available", False, "Using baseline fallback")

        if "champion_eval_df" in st.session_state and not st.session_state.champion_eval_df.empty:
            col = "ndcg@5"
            add("NDCG in range", bool(st.session_state.champion_eval_df[col].between(0, 1).all()))

        vp = generate_demo_video()
        add("Demo video generated", vp is not None and os.path.exists(vp), vp or "unavailable")
        add("SQLite path exists", os.path.exists(DB_PATH), DB_PATH)
        add("Quality report has no critical issues", bool(not (quality.severity == "critical").any()))
        add("Live simulation state exists", "sim_history" in st.session_state)
    except Exception as e:
        add("Unexpected error", False, repr(e))

    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# Live simulation engine
# -----------------------------------------------------------------------------


def compute_simulation_reference():
    _, _, events, _, _ = load_dataset()
    champ_mean = st.session_state.get("champion_eval_mean", pd.Series(dtype=float))
    base_mean = st.session_state.get("baseline_eval_mean", pd.Series(dtype=float))

    return {
        "ndcg": float(champ_mean.get("ndcg@5", 0.66)),
        "baseline_ndcg": float(base_mean.get("ndcg@5", 0.60)),
        "ctr": float(events.clicked.mean()) if not events.empty else 0.085,
        "conversion": float(events.converted.mean()) if not events.empty else 0.031,
        "zero": 0.024,
        "p50": 29.0,
        "p95": 74.0,
        "error": 0.0016,
    }


def init_simulation_state():
    st.session_state.setdefault("sim_running", True)
    st.session_state.setdefault("sim_tick", 0)
    st.session_state.setdefault("sim_delay", 0.75)
    st.session_state.setdefault("sim_max_ticks", 18)
    st.session_state.setdefault("sim_history", [])
    st.session_state.setdefault("sim_events", [])
    st.session_state.setdefault("sim_alerts", [])
    st.session_state.setdefault("sim_incident", None)
    st.session_state.setdefault("sim_pending_incident", False)
    st.session_state.setdefault("sim_fallback_active", False)

    if "sim_reference" not in st.session_state:
        st.session_state.sim_reference = compute_simulation_reference()


def add_sim_event(event_type, severity, message, query_id=None, doc_id=None, value=None):
    st.session_state.sim_events.append(
        {
            "timestamp": utc_now_iso(),
            "type": str(event_type),
            "severity": str(severity),
            "message": str(message),
            "query_id": str(query_id) if query_id is not None else "",
            "doc_id": str(doc_id) if doc_id is not None else "",
            "value": float(value) if value is not None else None,
        }
    )
    trim_session_list("sim_events", 700)


def add_sim_alert(severity, title, detail):
    alerts = st.session_state.sim_alerts
    recent_same = [a for a in alerts[-25:] if a.get("title") == title]
    if recent_same and recent_same[-1].get("severity") == severity:
        return
    alerts.append(
        {
            "timestamp": utc_now_iso(),
            "severity": str(severity),
            "title": str(title),
            "detail": str(detail),
            "status": "active",
        }
    )
    trim_session_list("sim_alerts", 120)


def choose_incident(tick):
    rng = np.random.default_rng(SEED + tick * 31)
    types = ["latency_spike", "relevance_drift", "data_quality", "feedback_burst"]
    typ = str(rng.choice(types))
    sev = "critical" if typ in ["relevance_drift", "latency_spike"] and rng.random() < 0.55 else "warning"
    remaining = int(rng.integers(4, 9))
    return {
        "type": typ,
        "severity": sev,
        "started_tick": int(tick),
        "remaining": remaining,
        "mitigated": False,
    }


def resolve_current_incident(manual=True):
    inc = st.session_state.get("sim_incident")
    if not inc:
        return
    inc["mitigated"] = True
    inc["remaining"] = min(int(inc.get("remaining", 1)), 2)
    st.session_state.sim_incident = inc
    st.session_state.sim_fallback_active = False
    add_sim_event(
        "incident_mitigation",
        "info",
        "Manual or automatic mitigation applied. Effects reduced and fallback cleared." if manual else "Incident auto-resolved.",
    )
    append_audit("sim_mitigate", str(inc.get("type", "")), st.session_state.get("actor", "system"))


def simulate_one_tick():
    tick = int(st.session_state.get("sim_tick", 0)) + 1
    st.session_state.sim_tick = tick
    ts = utc_now_iso()
    rng = np.random.default_rng(SEED + tick * 101)
    ref = st.session_state.get("sim_reference", {})

    # Incident lifecycle
    if st.session_state.get("sim_pending_incident", False):
        st.session_state.sim_incident = choose_incident(tick)
        st.session_state.sim_pending_incident = False
        inc = st.session_state.sim_incident
        add_sim_event(
            "incident_started",
            inc["severity"],
            f"Synthetic incident started: {inc['type']} ({inc['severity']}).",
            value=float(inc["remaining"]),
        )
        add_sim_alert(
            inc["severity"],
            f"Incident: {inc['type']}",
            f"Started at tick {tick}. Expected duration {inc['remaining']} ticks.",
        )

    inc = st.session_state.get("sim_incident")
    if inc:
        inc["remaining"] = max(0, int(inc.get("remaining", 0)) - 1)
        if inc["remaining"] == 0:
            add_sim_event("incident_resolved", "info", f"Incident resolved: {inc.get('type', 'unknown')}.")
            st.session_state.sim_incident = None
            st.session_state.sim_fallback_active = False
        else:
            st.session_state.sim_incident = inc

    # Base synthetic operational metrics
    p50 = float(ref.get("p50", 29.0)) + float(rng.normal(0, 2.4))
    p95 = float(ref.get("p95", 74.0)) + float(rng.normal(0, 6.5))
    error = max(0.0, float(ref.get("error", 0.0016)) + abs(float(rng.normal(0, 0.00055))))
    ctr = float(ref.get("ctr", 0.085)) + float(rng.normal(0, 0.008))
    conv = float(ref.get("conversion", 0.031)) + float(rng.normal(0, 0.0035))
    zero = max(0.0, float(ref.get("zero", 0.024)) + abs(float(rng.normal(0, 0.0045))))
    feedback = int(max(0, rng.poisson(3)))
    qps = max(1.0, 12.0 + float(rng.normal(0, 1.4)))
    ndcg = float(ref.get("ndcg", 0.66)) + float(rng.normal(0, 0.012))

    # Live ranking sample: small, CPU-friendly, deterministic by tick
    sample_queries = queries[queries.query_split == "test"].head(3)
    if sample_queries.empty:
        sample_queries = queries.head(3)

    model_for_tick = None if st.session_state.get("sim_fallback_active", False) else st.session_state.get("champion_model")
    ndcg_scores = []

    for q in sample_queries.itertuples():
        gold_map = gold[gold.query_id == q.query_id].set_index("doc_id")["relevance"].to_dict()
        try:
            ranked, _ = rank_results(
                q.query_text,
                model=model_for_tick,
                top_k=5,
                candidate_k=20,
                gold_ids=list(gold_map.keys()),
            )
        except Exception as e:
            ranked = pd.DataFrame()
            add_sim_event("ranking_error", "warning", f"Ranking error for {q.query_id}: {e}", query_id=q.query_id)

        if ranked.empty:
            zero = min(0.5, zero + 0.01)
            continue

        rels = [int(gold_map.get(d, 0)) for d in ranked.doc_id]
        ndcg_scores.append(ndcg_at_k(rels, list(gold_map.values()), 5))

        for pos, row in enumerate(ranked.head(3).itertuples(), start=1):
            score = float(getattr(row, "ai_score", getattr(row, "baseline_score", 0.0)))
            click_p = sigmoid(2.1 * score - 0.75 + float(rng.normal(0, 0.25)) - 0.24 * (pos - 1))
            clicked = bool(rng.random() < click_p)

            add_sim_event(
                "impression",
                "info",
                f"query={q.query_id} doc={row.doc_id} pos={pos}",
                query_id=q.query_id,
                doc_id=row.doc_id,
                value=score,
            )

            if clicked:
                add_sim_event(
                    "click",
                    "info",
                    f"query={q.query_id} doc={row.doc_id} pos={pos}",
                    query_id=q.query_id,
                    doc_id=row.doc_id,
                    value=score,
                )
                if rng.random() < max(0.01, conv * 7.5):
                    add_sim_event(
                        "conversion",
                        "info",
                        f"query={q.query_id} doc={row.doc_id}",
                        query_id=q.query_id,
                        doc_id=row.doc_id,
                        value=score,
                    )

    if ndcg_scores:
        ndcg = float(np.mean(ndcg_scores))

    # Apply incident effects
    inc = st.session_state.get("sim_incident")
    if inc:
        factor = 0.35 if inc.get("mitigated", False) else 1.0
        typ = inc.get("type", "")

        if typ == "latency_spike":
            p50 *= 1.0 + 2.1 * factor
            p95 *= 1.0 + 3.2 * factor
            error += 0.018 * factor
            ndcg -= 0.015 * factor
        elif typ == "relevance_drift":
            ndcg -= 0.13 * factor
            ctr -= 0.045 * factor
            conv -= 0.012 * factor
        elif typ == "data_quality":
            zero += 0.085 * factor
            error += 0.012 * factor
            feedback += int(6 * factor)
            ndcg -= 0.035 * factor
        elif typ == "feedback_burst":
            feedback += int(18 * factor)
            ndcg -= 0.025 * factor
            ctr += 0.008 * factor

    # Clamp metrics
    p50 = max(5.0, float(p50))
    p95 = max(p50 + 5.0, float(p95))
    error = float(np.clip(error, 0.0, 0.25))
    ctr = float(np.clip(ctr, 0.0, 0.60))
    conv = float(np.clip(conv, 0.0, 0.30))
    zero = float(np.clip(zero, 0.0, 0.50))
    ndcg = float(np.clip(ndcg, 0.0, 1.0))
    qps = float(max(1.0, qps))
    feedback = int(max(0, feedback))

    # Alerting
    active_alerts = []

    if p95 > 220.0:
        sev = "critical" if p95 > 300.0 else "warning"
        title = "Latency threshold exceeded"
        detail = f"p95={p95:.1f} ms. Safe fallback may be required."
        active_alerts.append((sev, title, detail))
    if ndcg < float(ref.get("ndcg", 0.66)) - 0.08:
        sev = "critical"
        title = "Relevance degradation detected"
        detail = f"NDCG proxy={ndcg:.3f}, reference={float(ref.get('ndcg', 0.66)):.3f}."
        active_alerts.append((sev, title, detail))
    if error > 0.02:
        sev = "critical"
        title = "Error rate threshold exceeded"
        detail = f"error_rate={error:.4f}."
        active_alerts.append((sev, title, detail))
    if zero > 0.10:
        sev = "warning"
        title = "Zero-result rate elevated"
        detail = f"zero_result_rate={zero:.3f}."
        active_alerts.append((sev, title, detail))
    if feedback > 20:
        sev = "warning"
        title = "Feedback burst detected"
        detail = f"feedback_count={feedback}. Check for manipulation or incident."
        active_alerts.append((sev, title, detail))

    for sev, title, detail in active_alerts:
        add_sim_alert(sev, title, detail)

    # Automatic safe fallback for critical service/relevance issues
    if any(sev == "critical" for sev, _, _ in active_alerts) and not st.session_state.get("sim_fallback_active", False):
        st.session_state.sim_fallback_active = True
        add_sim_event("fallback_enabled", "warning", "Automatic safe fallback to lexical baseline enabled.")
        append_audit("sim_fallback_enabled", "critical alert threshold crossed", "system")

    row = {
        "tick": tick,
        "timestamp": ts,
        "qps": qps,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "error_rate": error,
        "ndcg_proxy": ndcg,
        "baseline_ndcg_proxy": float(ref.get("baseline_ndcg", ndcg - 0.03)),
        "ctr_proxy": ctr,
        "conversion_proxy": conv,
        "zero_result_rate": zero,
        "feedback_count": feedback,
        "model_version": str(st.session_state.get("champion_version", "unknown")),
        "fallback_active": bool(st.session_state.get("sim_fallback_active", False)),
        "incident": str(inc.get("type", "")) if inc else "",
        "incident_severity": str(inc.get("severity", "")) if inc else "",
        "alert_count": len(active_alerts),
    }

    st.session_state.sim_history.append(row)
    trim_session_list("sim_history", 300)

    add_sim_event("tick_complete", "info", f"Tick {tick} complete.", value=ndcg)


# -----------------------------------------------------------------------------
# Application initialization
# -----------------------------------------------------------------------------


init_db()

docs, queries, events, gold, quality = load_dataset()

if "last_query" not in st.session_state:
    first_test = queries[queries.query_split == "test"]
    st.session_state.last_query = first_test.iloc[0].query_text if not first_test.empty else queries.iloc[0].query_text

if "champion_model" not in st.session_state:
    with st.spinner("Loading synthetic corpus, training local champion, and running frozen synthetic evaluation..."):
        champion_model, champion_version = get_initial_model()
        baseline_eval_df, baseline_eval_mean, baseline_eval_ci = evaluate_ranker(None)
        champion_eval_df, champion_eval_mean, champion_eval_ci = evaluate_ranker(champion_model)

        st.session_state.champion_model = champion_model
        st.session_state.champion_version = champion_version
        st.session_state.baseline_eval_df = baseline_eval_df
        st.session_state.baseline_eval_mean = baseline_eval_mean
        st.session_state.baseline_eval_ci = baseline_eval_ci
        st.session_state.champion_eval_df = champion_eval_df
        st.session_state.champion_eval_mean = champion_eval_mean
        st.session_state.champion_eval_ci = champion_eval_ci
        st.session_state.baseline_subgroup_min = min_subgroup_ndcg(baseline_eval_df)
        st.session_state.champion_subgroup_min = min_subgroup_ndcg(champion_eval_df)
        st.session_state.model_history = [
            {
                "version": champion_version,
                "created": utc_now_iso(),
                "role": "champion",
                "ndcg@5": float(champion_eval_mean.get("ndcg@5", 0.0)),
            }
        ]

        train_rows = int(len(events[events.query_split == "train"]))
        save_model_version(champion_version, train_rows, 1, champion_eval_mean)
        append_audit("init", f"champion={champion_version}", "system")

st.session_state.feedback_df = load_table("feedback", FEEDBACK_COLS)
st.session_state.audit_df = load_table("audit_log", AUDIT_COLS)

init_simulation_state()

# Auto-start a bounded live simulation immediately on launch.
if st.session_state.get("sim_running", False) and st.session_state.get("sim_tick", 0) < st.session_state.get("sim_max_ticks", 0):
    try:
        simulate_one_tick()
        if st.session_state.sim_tick >= st.session_state.sim_max_ticks:
            st.session_state.sim_running = False
    except Exception as e:
        st.session_state.sim_running = False
        add_sim_event("simulation_error", "critical", repr(e))

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

st.sidebar.title("🎛️ Controls")
st.sidebar.caption("Local synthetic pilot. No API keys. No private FAANG data.")

st.sidebar.header("Query")
st.session_state.role = st.sidebar.selectbox("Local role", ["Viewer", "Reviewer", "Admin"], index=1)
st.session_state.actor = st.sidebar.text_input("Reviewer alias (no PII)", "reviewer")
st.session_state.top_k = st.sidebar.slider("Top K results", 3, 20, 10)
st.session_state.candidate_k = st.sidebar.slider("Candidate pool size", 20, 100, 50)

query_input = st.sidebar.text_input("Query", value=st.session_state.last_query)
if st.sidebar.button("Analyze query", use_container_width=True):
    st.session_state.last_query = query_input
    append_audit("analyze", query_input, st.session_state.actor)
    st.rerun()

st.sidebar.header("Live simulation")
sim_running_ui = st.sidebar.checkbox("Auto-run ticks", value=bool(st.session_state.get("sim_running", False)))
st.session_state.sim_running = bool(sim_running_ui)

st.session_state.sim_delay = float(
    st.sidebar.slider("Tick delay seconds", 0.2, 3.0, float(st.session_state.get("sim_delay", 0.75)), 0.05)
)
st.session_state.sim_max_ticks = int(
    st.sidebar.number_input("Max ticks", min_value=5, max_value=200, value=int(st.session_state.get("sim_max_ticks", 18)), step=1)
)

sb1, sb2 = st.sidebar.columns(2)
if sb1.button("Reset sim", use_container_width=True):
    st.session_state.sim_history = []
    st.session_state.sim_events = []
    st.session_state.sim_alerts = []
    st.session_state.sim_incident = None
    st.session_state.sim_pending_incident = False
    st.session_state.sim_fallback_active = False
    st.session_state.sim_tick = 0
    st.session_state.sim_running = True
    st.session_state.sim_reference = compute_simulation_reference()
    append_audit("reset_simulation", "", st.session_state.actor)
    st.rerun()

if sb2.button("Inject incident", use_container_width=True):
    st.session_state.sim_pending_incident = True
    if not st.session_state.get("sim_running", False):
        st.session_state.sim_running = True
    append_audit("inject_incident", "", st.session_state.actor)
    st.rerun()

st.sidebar.metric("Champion version", st.session_state.champion_version)
st.sidebar.caption("Promotion requires candidate validation and human approval.")

current_query = st.session_state.last_query
top_k = int(st.session_state.top_k)
candidate_k = int(st.session_state.candidate_k)

# -----------------------------------------------------------------------------
# Header and tabs
# -----------------------------------------------------------------------------

st.title("🧭 Neural Search Relevance Optimization Engine")
st.caption(
    "Production-oriented local prototype for AI Search, Information Retrieval & Discovery Intelligence. "
    "Synthetic data only. Not affiliated with Meta, Amazon, Apple, Netflix, or Google."
)

tab_demo, tab_live, tab_workbench, tab_explain, tab_feedback, tab_eval, tab_data, tab_ops = st.tabs(
    [
        "Live Demo",
        "Live Simulation",
        "Ranking Workbench",
        "Explainability",
        "Feedback & Governance",
        "Evaluation",
        "Data & Tests",
        "Ops & Commercial",
    ]
)

# -----------------------------------------------------------------------------
# Live Demo
# -----------------------------------------------------------------------------

with tab_demo:
    st.subheader("Generated live demo video")
    with st.spinner("Preparing synthetic live demo video..."):
        video_path = generate_demo_video()

    if video_path:
        try:
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            mime = "video/avi" if video_path.lower().endswith(".avi") else "video/mp4"
            st.video(video_bytes, format=mime)
            st.download_button(
                "Download demo video",
                data=video_bytes,
                file_name=os.path.basename(video_path),
                mime=mime,
            )
        except Exception as e:
            st.warning(f"Video file exists but could not be embedded: {e}")
    else:
        st.warning("Video generation is unavailable in this environment. The live simulation below still runs.")

    st.divider()
    st.subheader("Automatic live simulation snapshot")

    hist = pd.DataFrame(st.session_state.sim_history)
    if hist.empty:
        st.info("Waiting for first simulation tick...")
    else:
        latest = hist.iloc[-1]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Tick", int(latest.tick))
        m2.metric("NDCG proxy", f"{float(latest.ndcg_proxy):.3f}")
        m3.metric("CTR proxy", f"{float(latest.ctr_proxy):.3f}")
        m4.metric("p95 latency", f"{float(latest.p95_latency_ms):.1f} ms")
        m5.metric("Fallback", "ON" if bool(latest.fallback_active) else "OFF")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(hist, x="tick", y=["ndcg_proxy", "baseline_ndcg_proxy"], title="Live relevance proxy")
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.line(hist, x="tick", y=["p50_latency_ms", "p95_latency_ms"], title="Live latency proxy")
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Guided replay")
    sim_container = st.empty()

    if not st.session_state.get("guided_demo_done", False):
        try:
            run_auto_simulation(sim_container, max(0.05, float(st.session_state.sim_delay)))
        except Exception as e:
            st.error(f"Guided replay error: {e}")
        st.session_state.guided_demo_done = True
    else:
        show_static_simulation_summary(sim_container)

    b1, b2 = st.columns(2)
    if b1.button("Replay guided simulation", use_container_width=True):
        st.session_state.guided_demo_done = False
        append_audit("replay_guided_simulation", "", st.session_state.actor)
        st.rerun()
    if b2.button("Load sample query", use_container_width=True):
        st.session_state.last_query = "lightweight business laptop"
        append_audit("load_sample_query", st.session_state.last_query, st.session_state.actor)
        st.rerun()

# -----------------------------------------------------------------------------
# Live Simulation
# -----------------------------------------------------------------------------

with tab_live:
    st.subheader("Live simulation dashboard")
    st.caption("Synthetic operational telemetry for local demonstration only. Not production metrics.")

    hist = pd.DataFrame(st.session_state.sim_history)
    events_df = pd.DataFrame(st.session_state.sim_events)
    alerts_df = pd.DataFrame(st.session_state.sim_alerts)
    inc = st.session_state.get("sim_incident")

    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns(5)
    if ctrl1.button("Start", use_container_width=True):
        st.session_state.sim_running = True
        if st.session_state.sim_tick >= st.session_state.sim_max_ticks:
            st.session_state.sim_tick = 0
        st.rerun()
    if ctrl2.button("Pause", use_container_width=True):
        st.session_state.sim_running = False
        st.rerun()
    if ctrl3.button("Step", use_container_width=True, disabled=bool(st.session_state.sim_running)):
        simulate_one_tick()
        if st.session_state.sim_tick >= st.session_state.sim_max_ticks:
            st.session_state.sim_running = False
        st.rerun()
    if ctrl4.button("Inject incident", use_container_width=True):
        st.session_state.sim_pending_incident = True
        st.session_state.sim_running = True
        st.rerun()
    if ctrl5.button("Reset", use_container_width=True):
        st.session_state.sim_history = []
        st.session_state.sim_events = []
        st.session_state.sim_alerts = []
        st.session_state.sim_incident = None
        st.session_state.sim_pending_incident = False
        st.session_state.sim_fallback_active = False
        st.session_state.sim_tick = 0
        st.session_state.sim_running = True
        st.session_state.sim_reference = compute_simulation_reference()
        st.rerun()

    status1, status2, status3 = st.columns(3)
    status1.metric("Simulation state", "RUNNING" if st.session_state.sim_running else "PAUSED")
    status2.metric("Tick", f"{st.session_state.sim_tick}/{st.session_state.sim_max_ticks}")
    status3.metric("Fallback", "ENABLED" if st.session_state.sim_fallback_active else "DISABLED")

    if inc:
        if inc.get("severity") == "critical":
            st.error(
                f"Active incident: **{inc.get('type')}** | severity **{inc.get('severity')}** | "
                f"remaining **{inc.get('remaining')}** ticks | mitigated **{inc.get('mitigated')}**"
            )
        else:
            st.warning(
                f"Active incident: **{inc.get('type')}** | severity **{inc.get('severity')}** | "
                f"remaining **{inc.get('remaining')}** ticks | mitigated **{inc.get('mitigated')}**"
            )

        ib1, ib2 = st.columns(2)
        if ib1.button("Mitigate incident", use_container_width=True):
            resolve_current_incident(manual=True)
            st.rerun()
        if ib2.button("Enable lexical fallback", use_container_width=True):
            st.session_state.sim_fallback_active = True
            add_sim_event("fallback_enabled", "warning", "Manual fallback to lexical baseline enabled.")
            append_audit("manual_fallback_enabled", "", st.session_state.actor)
            st.rerun()
    else:
        st.success("No active incident.")
        if st.session_state.sim_fallback_active:
            if st.button("Disable fallback and resume champion model", use_container_width=True):
                st.session_state.sim_fallback_active = False
                add_sim_event("fallback_disabled", "info", "Manual fallback disabled. Champion model resumed.")
                append_audit("manual_fallback_disabled", "", st.session_state.actor)
                st.rerun()

    if hist.empty:
        st.info("No simulation history yet. Press Start or Step.")
    else:
        latest = hist.iloc[-1]

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("QPS", f"{float(latest.qps):.1f}")
        k2.metric("NDCG proxy", f"{float(latest.ndcg_proxy):.3f}", delta=f"{float(latest.ndcg_proxy) - float(latest.baseline_ndcg_proxy):+.3f}")
        k3.metric("CTR proxy", f"{float(latest.ctr_proxy):.3f}")
        k4.metric("Conversion", f"{float(latest.conversion_proxy):.3f}")
        k5.metric("p95 latency", f"{float(latest.p95_latency_ms):.1f} ms")
        k6.metric("Error rate", f"{float(latest.error_rate):.4f}")

        chart1, chart2 = st.columns(2)
        with chart1:
            fig = px.line(
                hist,
                x="tick",
                y=["ndcg_proxy", "baseline_ndcg_proxy", "ctr_proxy"],
                title="Relevance and engagement over time",
            )
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with chart2:
            fig = px.line(
                hist,
                x="tick",
                y=["p50_latency_ms", "p95_latency_ms"],
                title="Latency over time",
            )
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        chart3, chart4 = st.columns(2)
        with chart3:
            fig = px.area(hist, x="tick", y="error_rate", title="Error rate")
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with chart4:
            fig = px.bar(hist, x="tick", y="feedback_count", title="Feedback volume")
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Current query live ranking")
    qrow = queries[queries.query_text == current_query]
    gold_ids = []
    if not qrow.empty:
        gold_ids = gold[gold.query_id == qrow.iloc[0].query_id].doc_id.tolist()

    live_model = None if st.session_state.sim_fallback_active else st.session_state.champion_model
    try:
        live_ranked, live_feats = rank_results(
            current_query,
            model=live_model,
            top_k=min(5, top_k),
            candidate_k=min(20, candidate_k),
            gold_ids=gold_ids,
        )
    except Exception as e:
        live_ranked = pd.DataFrame()
        live_feats = pd.DataFrame()
        st.error(f"Live ranking failed: {e}")

    if live_ranked.empty:
        st.info("No live ranking results available.")
    else:
        st.dataframe(
            live_ranked[["final_rank", "doc_id", "title", "category", "ai_score", "baseline_score"]],
            hide_index=True,
        )

        if st.session_state.role in ["Reviewer", "Admin"]:
            st.markdown("**Inject reviewer feedback into simulation context**")
            for idx, row in live_ranked.head(5).iterrows():
                fc1, fc2, fc3 = st.columns([5, 1, 1])
                fc1.write(f"{int(row.final_rank)}. `{row.doc_id}` {row.title}")
                if fc2.button("👍", key=f"live_up_{row.doc_id}_{idx}"):
                    add_feedback_record(row.doc_id, "relevant", "live simulation", current_query)
                    add_sim_event("reviewer_feedback", "info", f"Reviewer marked {row.doc_id} relevant.", doc_id=row.doc_id)
                if fc3.button("👎", key=f"live_down_{row.doc_id}_{idx}"):
                    add_feedback_record(row.doc_id, "irrelevant", "live simulation", current_query)
                    add_sim_event("reviewer_feedback", "info", f"Reviewer marked {row.doc_id} irrelevant.", doc_id=row.doc_id)

    st.subheader("Recent simulation events")
    if events_df.empty:
        st.info("No events yet.")
    else:
        st.dataframe(events_df.tail(80), hide_index=True, height=280)

    st.subheader("Alerts")
    if alerts_df.empty:
        st.success("No alerts.")
    else:
        st.dataframe(alerts_df.tail(30), hide_index=True, height=220)

# -----------------------------------------------------------------------------
# Ranking Workbench
# -----------------------------------------------------------------------------

with tab_workbench:
    st.subheader("Baseline vs AI-enhanced ranking")
    st.write(f"Query: `{current_query}`")

    qrow = queries[queries.query_text == current_query]
    gold_ids = []
    if not qrow.empty:
        gold_ids = gold[gold.query_id == qrow.iloc[0].query_id].doc_id.tolist()

    try:
        base_ranked, feats_base = rank_results(
            current_query,
            model=None,
            top_k=top_k,
            candidate_k=candidate_k,
            gold_ids=gold_ids,
        )
        ai_ranked, feats_ai = rank_results(
            current_query,
            model=st.session_state.champion_model,
            top_k=top_k,
            candidate_k=candidate_k,
            gold_ids=gold_ids,
        )
    except Exception as e:
        st.error(f"Ranking failed: {e}")
        base_ranked = pd.DataFrame()
        ai_ranked = pd.DataFrame()
        feats_ai = pd.DataFrame()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Transparent lexical baseline**")
        if not base_ranked.empty:
            st.dataframe(base_ranked[["final_rank", "doc_id", "title", "category", "baseline_score"]], hide_index=True)
        else:
            st.info("No baseline results.")
    with c2:
        st.markdown("**Neural/ML relevance optimizer**")
        if not ai_ranked.empty:
            st.dataframe(ai_ranked[["final_rank", "doc_id", "title", "category", "ai_score", "baseline_score"]], hide_index=True)
        else:
            st.info("No AI results.")

    if not ai_ranked.empty:
        st.markdown("**Rank movement: baseline rank vs AI rank**")
        mov = ai_ranked[["doc_id", "title", "baseline_rank", "final_rank", "ai_score"]].copy()
        fig = px.scatter(
            mov,
            x="baseline_rank",
            y="final_rank",
            text="title",
            hover_data=["doc_id", "ai_score"],
            title="Lower is better. Points below diagonal were promoted by the optimizer.",
        )
        max_rank = max(mov.baseline_rank.max(), mov.final_rank.max())
        fig.add_trace(go.Scatter(x=[1, max_rank], y=[1, max_rank], mode="lines", name="no change"))
        fig.update_layout(height=420, xaxis_title="Baseline rank", yaxis_title="AI rank")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Explainability
# -----------------------------------------------------------------------------

with tab_explain:
    st.subheader("Evidence and feature contribution")
    if "ai_ranked" not in locals() or ai_ranked.empty:
        st.info("Run a query in the Ranking Workbench first.")
    else:
        title_map = dict(zip(ai_ranked.doc_id, ai_ranked.title))
        doc_choice = st.selectbox(
            "Select result to explain",
            ai_ranked.doc_id.tolist(),
            format_func=lambda x: f"{x} - {title_map.get(x, '')}",
        )
        contrib, matched = explain_doc(current_query, doc_choice, st.session_state.champion_model, feats_ai)

        e1, e2 = st.columns([2, 1])
        with e1:
            if not contrib.empty:
                st.bar_chart(contrib)
            else:
                st.info("No contribution available.")
        with e2:
            st.markdown("**Matched terms**")
            st.write(", ".join(sorted(matched)) if matched else "None")
            st.markdown("**Document evidence**")
            if doc_choice in feats_ai.index:
                st.json(json.loads(feats_ai.loc[[doc_choice]].to_json(orient="records")))

# -----------------------------------------------------------------------------
# Feedback & Governance
# -----------------------------------------------------------------------------

with tab_feedback:
    st.subheader("Human review, correction, and model governance")
    can_write = st.session_state.role in ["Reviewer", "Admin"]
    if not can_write:
        st.warning("Viewer role is read-only. Switch to Reviewer or Admin in the sidebar for local demo corrections.")

    if "ai_ranked" in locals() and not ai_ranked.empty:
        st.markdown("**Quick feedback on current top 5**")
        for idx, row in ai_ranked.head(5).iterrows():
            c1, c2, c3 = st.columns([5, 1, 1])
            c1.write(f"{int(row.final_rank)}. `{row.doc_id}` {row.title}")
            if can_write:
                if c2.button("👍", key=f"up_{row.doc_id}_{idx}"):
                    add_feedback_record(row.doc_id, "relevant", "", current_query)
                if c3.button("👎", key=f"down_{row.doc_id}_{idx}"):
                    add_feedback_record(row.doc_id, "irrelevant", "", current_query)

    if can_write and "ai_ranked" in locals() and not ai_ranked.empty:
        with st.form("feedback_form"):
            st.markdown("**Detailed correction**")
            title_map = dict(zip(ai_ranked.doc_id, ai_ranked.title))
            doc = st.selectbox(
                "Document",
                ai_ranked.doc_id.tolist(),
                format_func=lambda x: f"{x} - {title_map.get(x, '')}",
                key="form_doc",
            )
            label = st.selectbox("Correction", ["relevant", "irrelevant", "uncertain"], key="form_label")
            note = st.text_area("Reviewer note", key="form_note")
            submitted = st.form_submit_button("Submit correction")
            if submitted:
                qmatch = queries[queries.query_text == current_query]
                qid = qmatch.iloc[0].query_id if not qmatch.empty else "U" + hashlib.sha256(current_query.encode()).hexdigest()[:8]
                rid = hashlib.sha256(f"{qid}|{doc}|{label}|{note or ''}".encode()).hexdigest()
                ok = insert_feedback(qid, current_query, doc, label, note, st.session_state.actor, rid)
                if ok:
                    append_audit("feedback_form", f"{doc}={label}", st.session_state.actor, rid)
                    st.session_state.feedback_df = load_table("feedback", FEEDBACK_COLS)
                    st.session_state.audit_df = load_table("audit_log", AUDIT_COLS)
                    st.success("Correction saved.")
                else:
                    st.info("Duplicate correction request ignored.")

    st.subheader("Feedback history")
    st.dataframe(st.session_state.feedback_df, hide_index=True, height=220)

    st.subheader("Audit log")
    st.dataframe(st.session_state.audit_df, hide_index=True, height=180)

    st.subheader("Candidate retraining and promotion gate")
    st.caption("Candidate training excludes frozen test-query feedback to preserve evaluation integrity.")

    if can_write:
        cb1, cb2 = st.columns(2)
        if cb1.button("Train candidate model from approved feedback", use_container_width=True):
            with st.spinner("Training candidate and evaluating on frozen synthetic test set..."):
                cand_model, cand_version, n_fb_used = train_candidate_model(st.session_state.feedback_df)
                cand_df, cand_mean, cand_ci = evaluate_ranker(cand_model)
                st.session_state.candidate = {
                    "model": cand_model,
                    "version": cand_version,
                    "df": cand_df,
                    "mean": cand_mean,
                    "ci": cand_ci,
                    "feedback_rows_used": n_fb_used,
                }
                append_audit("train_candidate", f"version={cand_version}, feedback_rows={n_fb_used}", st.session_state.actor)

        if "candidate" in st.session_state:
            cand = st.session_state.candidate
            champ_mean = st.session_state.champion_eval_mean
            cand_mean = cand["mean"]
            ndcg_col = "ndcg@5"

            improvement = float(cand_mean.get(ndcg_col, 0.0) - champ_mean.get(ndcg_col, 0.0))
            cand_sub_min = min_subgroup_ndcg(cand["df"])
            gate = (
                cand["model"] is not None
                and improvement >= 0.005
                and cand_sub_min >= st.session_state.champion_subgroup_min - 0.02
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Candidate version", cand["version"])
            m2.metric("NDCG@5 delta vs champion", f"{improvement:+.4f}")
            m3.metric("Feedback rows used", cand["feedback_rows_used"])

            compare = pd.DataFrame(
                {
                    "champion": champ_mean,
                    "candidate": cand_mean,
                }
            )
            st.dataframe(compare, hide_index=True)

            st.markdown(f"**Promotion gate:** {'PASS' if gate else 'FAIL'}")
            st.caption("Gate: +0.005 NDCG@5 and no subgroup drop > 0.02 versus champion.")

            approve = st.checkbox("I reviewed candidate metrics and approve promotion", disabled=not gate)
            pb1, pb2 = st.columns(2)
            if pb1.button("Promote candidate to champion", disabled=not (gate and approve), use_container_width=True):
                st.session_state.previous_model = st.session_state.champion_model
                st.session_state.previous_version = st.session_state.champion_version
                st.session_state.previous_eval_df = st.session_state.champion_eval_df
                st.session_state.previous_eval_mean = st.session_state.champion_eval_mean
                st.session_state.previous_subgroup_min = st.session_state.champion_subgroup_min

                st.session_state.champion_model = cand["model"]
                st.session_state.champion_version = cand["version"]
                st.session_state.champion_eval_df = cand["df"]
                st.session_state.champion_eval_mean = cand["mean"]
                st.session_state.champion_eval_ci = cand["ci"]
                st.session_state.champion_subgroup_min = cand_sub_min

                st.session_state.model_history.append(
                    {
                        "version": cand["version"],
                        "created": utc_now_iso(),
                        "role": "champion",
                        "ndcg@5": float(cand["mean"].get("ndcg@5", 0.0)),
                    }
                )
                save_model_version(cand["version"], 0, 1, cand["mean"])
                append_audit("promote_model", f"version={cand['version']}", st.session_state.actor)
                st.success("Candidate promoted to champion.")
                st.rerun()

            if pb2.button("Rollback to previous champion", disabled="previous_model" not in st.session_state, use_container_width=True):
                st.session_state.champion_model = st.session_state.previous_model
                st.session_state.champion_version = st.session_state.previous_version
                st.session_state.champion_eval_df = st.session_state.previous_eval_df
                st.session_state.champion_eval_mean = st.session_state.previous_eval_mean
                st.session_state.champion_subgroup_min = st.session_state.previous_subgroup_min
                append_audit("rollback_model", f"version={st.session_state.champion_version}", st.session_state.actor)
                st.success("Rolled back to previous champion.")
                st.rerun()
    else:
        st.info("Only Reviewer or Admin roles can train or promote candidate models in this local demo.")

    st.subheader("Model history")
    st.dataframe(pd.DataFrame(st.session_state.model_history), hide_index=True)

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

with tab_eval:
    st.subheader("Frozen synthetic offline evaluation")
    st.caption("These are locally computed synthetic metrics. They are not production claims.")

    compare = pd.DataFrame(
        {
            "baseline": st.session_state.baseline_eval_mean,
            "champion": st.session_state.champion_eval_mean,
        }
    )
    st.dataframe(compare, hide_index=True)

    ndcg_col = "ndcg@5"
    base_ndcg = float(st.session_state.baseline_eval_mean.get(ndcg_col, 0.0))
    champ_ndcg = float(st.session_state.champion_eval_mean.get(ndcg_col, 0.0))
    delta = champ_ndcg - base_ndcg

    e1, e2, e3 = st.columns(3)
    e1.metric("Baseline NDCG@5", f"{base_ndcg:.3f}")
    e2.metric("Champion NDCG@5", f"{champ_ndcg:.3f}", delta=f"{delta:+.3f}")
    e3.metric(
        "Champion bootstrap CI",
        f"{st.session_state.champion_eval_ci[0]:.3f}-{st.session_state.champion_eval_ci[1]:.3f}",
    )

    st.markdown("**Subgroup performance by query category**")
    if (
        not st.session_state.baseline_eval_df.empty
        and not st.session_state.champion_eval_df.empty
        and "category" in st.session_state.baseline_eval_df.columns
        and "category" in st.session_state.champion_eval_df.columns
    ):
        sub_base = st.session_state.baseline_eval_df.groupby("category")[ndcg_col].mean().rename("baseline")
        sub_champ = st.session_state.champion_eval_df.groupby("category")[ndcg_col].mean().rename("champion")
        sub = pd.concat([sub_base, sub_champ], axis=1).reset_index()
        sub["delta"] = sub["champion"] - sub["baseline"]
        st.dataframe(sub, hide_index=True)
    else:
        st.info("No test queries available for subgroup evaluation.")

    st.markdown("**Calibration and discrimination on synthetic test impressions**")
    cal_df, cal_metrics, cal_curve = calibration_report(st.session_state.champion_model)
    if cal_metrics:
        c1, c2, c3 = st.columns(3)
        c1.metric("Brier score", f"{cal_metrics.get('brier', float('nan')):.3f}")
        c2.metric("Log loss", f"{cal_metrics.get('log_loss', float('nan')):.3f}")
        c3.metric("AUC", f"{cal_metrics.get('auc', float('nan')):.3f}")
    else:
        st.info("Calibration metrics unavailable, likely due to single-class test impressions or fallback model.")

    if not cal_curve.empty:
        fig = px.line(cal_curve, x="pred", y="obs", markers=True, title="Calibration curve (predicted vs observed)")
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect"))
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Single-user local latency benchmark**")
    st.caption("Not a distributed load test. Use this only as a local CPU sanity check.")
    bench = benchmark_latency(st.session_state.champion_model, n=20)
    st.dataframe(bench, hide_index=True)

    st.markdown("**Failure analysis**")
    fp, fn, unc = failure_analysis(st.session_state.champion_model, k=5)
    f1, f2, f3 = st.tabs(["False positives", "False negatives", "Uncertain cases"])
    with f1:
        st.dataframe(fp, hide_index=True)
    with f2:
        st.dataframe(fn, hide_index=True)
    with f3:
        st.dataframe(unc, hide_index=True)

# -----------------------------------------------------------------------------
# Data & Tests
# -----------------------------------------------------------------------------

with tab_data:
    st.subheader("Synthetic data provenance")
    st.markdown(
        """
        - All documents, queries, events, and gold labels are synthetic and fictional.
        - No real people, private data, proprietary APIs, or FAANG internal data are used.
        - Train/test split is by `query_id`.
        - Training events are temporally older than test events.
        - Feedback on frozen test queries is excluded from candidate training.
        """
    )

    st.markdown("**Example document record**")
    st.json(json.loads(docs.head(1).to_json(orient="records")))

    st.markdown("**Example event record**")
    st.json(json.loads(events.head(1).to_json(orient="records")))

    st.markdown("**Example gold label record**")
    st.json(json.loads(gold.head(1).to_json(orient="records")))

    st.subheader("Dataset previews")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Documents**")
        st.dataframe(docs.head(20), hide_index=True, height=300)
    with d2:
        st.markdown("**Queries**")
        st.dataframe(queries.head(20), hide_index=True, height=300)

    st.markdown("**Events**")
    st.dataframe(events.head(30), hide_index=True, height=300)

    st.subheader("Data quality report")
    st.dataframe(quality, hide_index=True)

    st.subheader("Automated smoke tests")
    tests = run_self_tests()
    st.dataframe(tests, hide_index=True)
    if bool(tests.passed.all()):
        st.success("All local smoke tests passed.")
    else:
        st.error("Some smoke tests failed. Inspect the table above.")

# -----------------------------------------------------------------------------
# Ops & Commercial
# -----------------------------------------------------------------------------

with tab_ops:
    st.subheader("Deployment, operations, and commercial readiness")

    st.markdown(
        """
        ### Environments
        - **Local demo:** Streamlit, SQLite, synthetic data, CPU-only.
        - **Staging:** containerized FastAPI service, PostgreSQL, warehouse snapshots, CI tests.
        - **Production:** HA search stack, model registry, canary deployment, rollback, SSO/RBAC, audit storage.

        ### SLO proposal
        - Ranking API p95 latency: `< 120 ms` for top-50 candidate rerank on CPU.
        - Availability: `99.9%` for decision service.
        - Data freshness: feedback visible in review UI `< 1 minute`.
        - Model drift alert: NDCG proxy drop `> 3%` over rolling window.
        - Human review threshold: any promotion requires validation gate and approver.

        ### Observability
        - Service: latency, error rate, queue depth, correlation IDs.
        - Data: missing fields, duplicate events, schema drift, delayed events.
        - Model: score distribution, calibration, subgroup NDCG, feedback volume.
        - Governance: audit log, model version, promotion/rollback events.

        ### Security and privacy
        - Minimize data: no PII in synthetic demo.
        - Local SQLite file is disposable.
        - Role gating: Viewer, Reviewer, Admin.
        - Audit sensitive actions: feedback, analyze, train, promote, rollback.
        - Deletion/correction: feedback table supports retention cleanup and reviewer notes.

        ### Planned API contract
        - `POST /v1/rank` - query + candidate IDs -> ranked results + explanation payload.
        - `POST /v1/feedback` - idempotent correction endpoint.
        - `POST /v1/models/{version}/validate` - frozen evaluation report.
        - `POST /v1/models/{version}/promote` - requires approval token.
        - `GET /health` - liveness/readiness.

        ### Commercial validation
        - **Ideal customer:** companies with search-driven revenue or support deflection.
        - **Urgent pain:** poor discovery, high zero-result rate, manual relevance tuning.
        - **Competing alternatives:** native search tools, custom Elasticsearch tuning, vendor black boxes.
        - **Differentiators:** explainability, governance, evaluation gates, local-first prototype.
        - **Pricing hypothesis:** subscription by indexed documents + monthly queries; pilot first.
        - **ROI calculator:** value = incremental conversions or deflected tickets - platform cost.
        - **Renewal signal:** sustained NDCG/CTR lift, reduced review time, model governance adoption.

        ### 90-day roadmap
        1. Days 0-15: harden synthetic evaluation, tests, docs.
        2. Days 16-30: FastAPI service, PostgreSQL, containerization.
        3. Days 31-60: customer pilot with anonymized logs, A/B framework.
        4. Days 61-90: production HA, model registry, canary, security review.

        ### Go/no-go checklist
        - [ ] Reproducible setup from clean Windows environment.
        - [ ] Smoke tests pass.
        - [ ] Frozen synthetic evaluation documented.
        - [ ] No automatic promotion without human approval.
        - [ ] Privacy/security checklist completed.
        - [ ] Customer validation interview scheduled.
        """
    )

    st.info(
        "This prototype is independent and not endorsed by or affiliated with Meta, Amazon, Apple, Netflix, or Google."
    )

# -----------------------------------------------------------------------------
# Bottom scheduling for live simulation
# -----------------------------------------------------------------------------

if (
    st.session_state.get("sim_running", False)
    and st.session_state.get("sim_tick", 0) < st.session_state.get("sim_max_ticks", 0)
):
    time.sleep(max(0.05, float(st.session_state.get("sim_delay", 0.75))))
    st.rerun()