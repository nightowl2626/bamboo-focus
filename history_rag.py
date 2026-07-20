"""Privacy-safe local BM25 history retrieval for Bamboo Focus.

This mirrors Qwen-Agent's lightweight RAG shape: build text chunks from local
documents, score them with sparse keyword matching, and return grounded snippets.
The corpus is limited to derived records. It never indexes raw video or frames.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edge_privacy import read_json, tail_jsonl


TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass
class RagDocument:
    id: str
    source: str
    title: str
    text: str
    created_at: str | None = None
    metadata: dict[str, Any] | None = None


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def in_lookback(created_at: str | None, lookback_days: float | None) -> bool:
    if lookback_days is None or lookback_days <= 0:
        return True
    parsed = parse_time(created_at)
    if parsed is None:
        return True
    age_seconds = datetime.now(timezone.utc).timestamp() - parsed.timestamp()
    return age_seconds <= lookback_days * 86400


def compact_json(value: Any, max_chars: int = 1200) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def add_doc(
    docs: list[RagDocument],
    *,
    source: str,
    title: str,
    text: str,
    created_at: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return
    docs.append(
        RagDocument(
            id=f"{source}:{len(docs) + 1}",
            source=source,
            title=title,
            text=normalized,
            created_at=created_at,
            metadata=metadata or {},
        )
    )


def session_summary_docs(paths: Any) -> list[RagDocument]:
    docs: list[RagDocument] = []
    for item in tail_jsonl(paths.agent_data_dir / "session_summaries.jsonl", 200):
        stats = item.get("stats") if isinstance(item.get("stats"), dict) else {}
        settings = item.get("settings") if isinstance(item.get("settings"), dict) else {}
        session = item.get("session") if isinstance(item.get("session"), dict) else {}
        text = (
            f"Session summary. Intent: {settings.get('intent', '')}. "
            f"Tone: {settings.get('tone', '')}. Focus areas: {settings.get('focus_areas', [])}. "
            f"Stats: {compact_json(stats)}. Paragraph: {item.get('paragraph', '')}"
        )
        add_doc(
            docs,
            source="session_summary",
            title=f"Session summary {session.get('started_at') or item.get('created_at') or ''}".strip(),
            text=text,
            created_at=item.get("created_at") or session.get("ended_at"),
            metadata={"session": session, "stats": stats},
        )
    return docs


def nudge_decision_docs(paths: Any) -> list[RagDocument]:
    docs: list[RagDocument] = []
    for item in tail_jsonl(paths.decisions_path, 500):
        decision = item.get("decision") if isinstance(item.get("decision"), dict) else item
        retrieval = item.get("retrieval") if isinstance(item.get("retrieval"), dict) else {}
        text = (
            f"Nudge decision. Should nudge: {decision.get('should_nudge')}. "
            f"Category: {decision.get('category')}. Urgency: {decision.get('urgency')}. "
            f"Focus: {decision.get('recommended_focus')}. Rationale: {decision.get('rationale')}. "
            f"Signals: {decision.get('supporting_signals', [])}. Suppress reason: {decision.get('suppress_reason')}. "
            f"Cooldown key: {decision.get('cooldown_key')}. Retrieval: {compact_json(retrieval, 800)}."
        )
        add_doc(
            docs,
            source="nudge_decision",
            title=f"Nudge decision {decision.get('category') or 'none'}",
            text=text,
            created_at=item.get("created_at"),
            metadata={"decision": decision},
        )
    return docs


def decision_trace_docs(paths: Any) -> list[RagDocument]:
    docs: list[RagDocument] = []
    for item in tail_jsonl(paths.agent_data_dir / "decision_traces.jsonl", 500):
        decision = item.get("decision") if isinstance(item.get("decision"), dict) else {}
        why = item.get("why") if isinstance(item.get("why"), dict) else {}
        memory = item.get("memory_influence") if isinstance(item.get("memory_influence"), dict) else {}
        edge = item.get("edge_evidence") if isinstance(item.get("edge_evidence"), dict) else {}
        text = (
            f"Decision trace. Category: {decision.get('category')}. Should nudge: {decision.get('should_nudge')}. "
            f"Why: {why.get('rationale') or why.get('suppress_reason')}. Signals: {why.get('supporting_signals', [])}. "
            f"Edge evidence: {compact_json(edge)}. Memory influence: {compact_json(memory)}."
        )
        add_doc(
            docs,
            source="decision_trace",
            title=f"Decision trace {decision.get('category') or 'none'}",
            text=text,
            created_at=item.get("created_at") or item.get("decision_created_at"),
            metadata={"decision": decision, "memory_influence": memory},
        )
    return docs


def posture_analysis_docs(paths: Any) -> list[RagDocument]:
    docs: list[RagDocument] = []
    for item in tail_jsonl(paths.long_analyses_path, 500):
        analysis = item.get("analysis") if isinstance(item.get("analysis"), dict) else {}
        text = (
            f"Posture analysis. Judgement: {analysis.get('judgement')}. "
            f"Posture: {analysis.get('posture')}. Behaviour: {analysis.get('behaviour')}. "
            f"Stillness or restlessness: {analysis.get('stillness_or_restlessness')}. "
            f"Changes: {analysis.get('significant_changes', [])}. Observations: {analysis.get('observations', [])}. "
            f"Confidence: {analysis.get('confidence')}."
        )
        add_doc(
            docs,
            source="posture_analysis",
            title=f"Posture analysis {analysis.get('posture') or 'unknown'}",
            text=text,
            created_at=item.get("created_at"),
            metadata={"analysis": analysis},
        )
    return docs


def object_snapshot_docs(paths: Any) -> list[RagDocument]:
    docs: list[RagDocument] = []
    for item in tail_jsonl(paths.object_events_path, 500):
        objects = item.get("monitorable_objects") if isinstance(item.get("monitorable_objects"), list) else []
        labels = [
            {
                "label": obj.get("label"),
                "score": obj.get("score"),
            }
            for obj in objects
            if isinstance(obj, dict)
        ]
        text = (
            f"Object snapshot. Monitorable count: {item.get('monitorable_count')}. "
            f"Monitorable objects: {compact_json(labels)}. "
            f"Whitelist labels: {item.get('whitelist_labels', [])}. Monitor labels: {item.get('monitor_labels', [])}."
        )
        add_doc(
            docs,
            source="object_snapshot",
            title="Object snapshot",
            text=text,
            created_at=item.get("received_at"),
            metadata={"monitorable_objects": labels},
        )
    return docs


def baseline_policy_docs(paths: Any) -> list[RagDocument]:
    docs: list[RagDocument] = []
    baseline = read_json(paths.baseline) or {}
    policy = baseline.get("object_policy") if isinstance(baseline.get("object_policy"), dict) else {}
    calibration = baseline.get("calibration") if isinstance(baseline.get("calibration"), dict) else {}
    text = (
        f"Baseline policy. Created at: {baseline.get('created_at')}. "
        f"Whitelisted objects: {policy.get('whitelisted_objects', [])}. "
        f"Monitorable objects: {policy.get('monitorable_objects', [])}. "
        f"Uncertain objects: {policy.get('uncertain_objects', [])}. Notes: {policy.get('notes', '')}. "
        f"Calibration event count: {len(calibration.get('events', [])) if isinstance(calibration.get('events'), list) else 0}."
    )
    add_doc(
        docs,
        source="baseline_policy",
        title="Baseline object policy",
        text=text,
        created_at=baseline.get("created_at"),
        metadata={"object_policy": policy},
    )
    return docs


def build_history_documents(paths: Any) -> list[RagDocument]:
    docs: list[RagDocument] = []
    for builder in (
        session_summary_docs,
        nudge_decision_docs,
        decision_trace_docs,
        posture_analysis_docs,
        object_snapshot_docs,
        baseline_policy_docs,
    ):
        docs.extend(builder(paths))
    return docs


def bm25_scores(query_tokens: list[str], documents: list[list[str]]) -> list[float]:
    if not documents or not query_tokens:
        return [0.0 for _ in documents]
    k1 = 1.5
    b = 0.75
    avgdl = sum(len(doc) for doc in documents) / len(documents)
    df: dict[str, int] = {}
    for doc in documents:
        for token in set(doc):
            df[token] = df.get(token, 0) + 1
    scores = []
    for doc in documents:
        doc_len = len(doc) or 1
        tf: dict[str, int] = {}
        for token in doc:
            tf[token] = tf.get(token, 0) + 1
        score = 0.0
        for token in set(query_tokens):
            freq = tf.get(token, 0)
            if not freq:
                continue
            idf = math.log(1 + (len(documents) - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len / (avgdl or 1))
            score += idf * (freq * (k1 + 1) / denom)
        scores.append(score)
    return scores


def search_history(
    paths: Any,
    query: str,
    limit: int = 6,
    lookback_days: float | None = 30,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    limit = max(1, min(20, int(limit)))
    docs = build_history_documents(paths)
    allowed_sources = {str(item) for item in sources or [] if str(item)}
    filtered = [
        doc
        for doc in docs
        if (not allowed_sources or doc.source in allowed_sources)
        and in_lookback(doc.created_at, lookback_days)
    ]
    query_tokens = tokenize(query)
    tokenized_docs = [tokenize(f"{doc.title} {doc.text}") for doc in filtered]
    scores = bm25_scores(query_tokens, tokenized_docs)
    ranked = sorted(zip(filtered, scores), key=lambda pair: pair[1], reverse=True)
    matches = []
    for doc, score in ranked[:limit]:
        if score <= 0 and query_tokens:
            continue
        snippet = doc.text[:900].rstrip()
        matches.append(
            {
                "id": doc.id,
                "source": doc.source,
                "title": doc.title,
                "created_at": doc.created_at,
                "score": round(score, 4),
                "snippet": snippet,
                "metadata": doc.metadata or {},
            }
        )
    return {
        "tool": "search_history_rag",
        "query": query,
        "lookback_days": lookback_days,
        "indexed_documents": len(docs),
        "searched_documents": len(filtered),
        "matches": matches,
        "privacy": {
            "raw_video_indexed": False,
            "camera_frames_indexed": False,
            "indexed_data": "derived JSON summaries, decisions, traces, object labels, and baseline policy only",
        },
    }
