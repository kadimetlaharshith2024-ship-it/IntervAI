#!/usr/bin/env python3
"""
Evaluation script to analyze interview_sessions, retrieval_traces, questions,
and database PII / junk-topic rates across the IntervAI platform.
"""

import sys
import os
import re
import json
from collections import defaultdict, Counter
from typing import Dict, Any, List

# Ensure backend app is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
backend_dir = os.path.join(project_root, "backend")

if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

os.chdir(backend_dir)
from dotenv import load_dotenv
load_dotenv(os.path.join(backend_dir, ".env"))

try:
    from app.core.database import SessionLocal
    from app.models.interview import InterviewSession, InterviewQuestion, InterviewAnswer
    from app.models.rag import RetrievalTrace, ResumeClaim, ResumeChunk
    from app.models.relationship import Relationship
    from app.models.resume import Resume
except ImportError as e:
    print(f"Error importing app modules: {e}")
    sys.exit(1)


# Comprehensive regex patterns for PII scanning
PII_PATTERNS = {
    "phone_labeled": re.compile(r"(?i)\b(?:t|tel|ph|phone|mob|mobile|call|contact)[:\s]*(\+?[\d\s().-]{7,20})\b"),
    "phone_10digit": re.compile(r"\b(?:\+?91[-.\s]?)?[6-9]\d{4}[-.\s]?\d{5}\b"),
    "phone_us": re.compile(r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s][2-9]\d{2}[-.\s]\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+\s*(?:@|\[at\]|\(at\))\s*[A-Za-z0-9.-]+\s*(?:\.|\(dot\)|\[dot\])\s*[A-Za-z]{2,}\b", re.IGNORECASE),
    "linkedin_url": re.compile(r"(?:https?:\/\/)?(?:www\.)?linkedin\.com\/(?:in|pub|profile)\/[A-Za-z0-9_\-\/]+", re.IGNORECASE),
    "linkedin_handle": re.compile(r"(?i)\blinkedin[:\s]+@?([a-zA-Z0-9_\-\.]{3,30})\b"),
    "github_url": re.compile(r"(?:https?:\/\/)?(?:www\.)?github\.com\/[A-Za-z0-9_\-\/]+", re.IGNORECASE),
    "github_handle": re.compile(r"(?i)\bgithub[:\s]+@?([a-zA-Z0-9_\-\.]{3,39})\b"),
}

# Generic words / Non-assessable topic patterns
JUNK_TOPIC_BLOCKLIST = {
    "candidate", "institute", "college", "university", "school", "department",
    "semester", "cgpa", "gpa", "b.tech", "m.tech", "btech", "mtech", "b.e", "b.s",
    "bachelor", "master", "phd", "degree", "education", "experience", "projects",
    "academic", "achievements", "activities", "leadership", "extracurricular",
    "volleyball", "cricket", "sports", "annual", "fest", "captain", "captained",
    "winner", "won", "runner", "rank", "ranking", "jee", "mains", "gate",
    "telugu", "hindi", "english", "tamil", "kannada", "french", "german", "spanish",
    "native", "fluent", "intermediate", "the abo", "accuracy", "timestamp-ordered operation"
}


def is_junk_topic(topic_name: str) -> bool:
    t_lower = topic_name.strip().lower()
    if t_lower in JUNK_TOPIC_BLOCKLIST:
        return True
    if len(t_lower) > 35 or len(t_lower.split()) > 4:
        return True
    if any(b in t_lower for b in ["institute", "university", "college", "b.tech", "m.tech", "jee mains", "fest", "captained", "extracurricular", "all india rank"]):
        return True
    if any(l in t_lower for l in ["telugu", "hindi", "tamil", "fluent in"]):
        return True
    return False


def run_evaluation() -> Dict[str, Any]:
    db = SessionLocal()
    try:
        # 1. Latency Metrics
        traces = db.query(RetrievalTrace).all()
        latencies = []
        for t in traces:
            if t.latencies_ms and isinstance(t.latencies_ms, dict):
                tot = t.latencies_ms.get("total_ms")
                if tot is not None and isinstance(tot, (int, float)) and tot > 0:
                    latencies.append(float(tot))

        latencies.sort()
        n_lats = len(latencies)
        def percentile(p):
            if not latencies:
                return 0.0
            idx = int(round((p / 100.0) * (n_lats - 1)))
            return latencies[min(idx, n_lats - 1)]

        p50 = percentile(50)
        p90 = percentile(90)
        p95 = percentile(95)
        max_lat = max(latencies) if latencies else 0.0
        over_10s_count = sum(1 for l in latencies if l > 10000.0)
        share_over_10s = (over_10s_count / n_lats * 100.0) if n_lats else 0.0

        # 2. Strategy Distribution
        # Inferred from questions and session_state
        strategy_counts = Counter()
        questions = db.query(InterviewQuestion).all()
        for q in questions:
            strat = "UNCOVERED_TOPIC"
            val_details = q.validation_details or {}
            rat = (q.rationale or "").lower()

            if q.parent_question_id is not None or "[follow-up" in rat:
                if "contradiction" in rat or "discrepancy" in rat:
                    strat = "CONTRADICTION_CLARIFICATION"
                elif "partial" in rat or "omitted" in rat:
                    strat = "PARTIAL_CLARIFICATION"
                elif "gap" in rat or "knowledge gap" in rat or "foundational" in rat:
                    strat = "GAP_FOLLOW_UP"
                elif "escalat" in rat or "deepen" in rat or "failure" in rat:
                    strat = "DIFFICULTY_ESCALATION"
                else:
                    strat = "GAP_FOLLOW_UP"
            elif q.category == "Claim-Verification" or q.target_claim_id:
                strat = "CLAIM_VERIFICATION"
            elif q.target_relationship:
                strat = "RELATIONSHIP_VERIFICATION"
            elif q.category == "Project-Depth" or (q.target_project and not q.target_entity):
                strat = "PROJECT_DEPTH"
            elif "coverage" in rat or "baseline" in rat:
                strat = "COVERAGE_EXPANSION"
            strategy_counts[strat] += 1

        # 3. Requested vs Assessed Difficulty Matrix
        diff_matrix = defaultdict(lambda: defaultdict(int))
        for q in questions:
            req_d = q.difficulty or 3
            ass_d = q.assessed_difficulty or req_d
            diff_matrix[req_d][ass_d] += 1

        # 4. Junk Topic Rate
        sessions = db.query(InterviewSession).all()
        total_topics_set = set()
        junk_topics_set = set()
        coverage_percentages = []

        for s in sessions:
            if not s.session_state or not isinstance(s.session_state, dict):
                continue
            k_state = s.session_state.get("knowledge_state", {})
            topics = k_state.get("topics", {})
            for t_name in topics.keys():
                total_topics_set.add(t_name)
                if is_junk_topic(t_name):
                    junk_topics_set.add(t_name)

            cov = k_state.get("coverage", {})
            if "coverage_percentage" in cov and cov["coverage_percentage"] is not None:
                coverage_percentages.append(float(cov["coverage_percentage"]))

        junk_topic_rate = (len(junk_topics_set) / len(total_topics_set) * 100.0) if total_topics_set else 0.0
        avg_coverage = (sum(coverage_percentages) / len(coverage_percentages)) if coverage_percentages else 0.0

        # 5. PII Scan across database tables
        pii_violations = {
            "resume_chunks": 0,
            "resume_claims": 0,
            "relationships": 0,
            "interview_questions": 0
        }

        # Check resume_chunks
        for c in db.query(ResumeChunk).all():
            txt = c.chunk_text or ""
            # Exclude standardized redaction markers like [REDACTED_PHONE]
            clean_txt = re.sub(r"\[REDACTED_[A-Z]+\]", "", txt)
            if any(p.search(clean_txt) for p in PII_PATTERNS.values()):
                pii_violations["resume_chunks"] += 1

        # Check resume_claims
        for cl in db.query(ResumeClaim).all():
            txt = f"{cl.claim_text or ''} {cl.evidence_text or ''}"
            clean_txt = re.sub(r"\[REDACTED_[A-Z]+\]", "", txt)
            if any(p.search(clean_txt) for p in PII_PATTERNS.values()):
                pii_violations["resume_claims"] += 1

        # Check relationships
        for r in db.query(Relationship).all():
            txt = f"{r.evidence_text or ''} {r.source_entity or ''} {r.target_entity or ''}"
            clean_txt = re.sub(r"\[REDACTED_[A-Z]+\]", "", txt)
            if any(p.search(clean_txt) for p in PII_PATTERNS.values()):
                pii_violations["relationships"] += 1

        # Check interview_questions
        for q in db.query(InterviewQuestion).all():
            txt = f"{q.evidence_text or ''} {q.question_text or ''}"
            clean_txt = re.sub(r"\[REDACTED_[A-Z]+\]", "", txt)
            if any(p.search(clean_txt) for p in PII_PATTERNS.values()):
                pii_violations["interview_questions"] += 1

        return {
            "total_sessions": len(sessions),
            "total_questions": len(questions),
            "total_traces": len(traces),
            "latency": {
                "count": n_lats,
                "p50_ms": round(p50, 2),
                "p90_ms": round(p90, 2),
                "p95_ms": round(p95, 2),
                "max_ms": round(max_lat, 2),
                "share_over_10s_pct": round(share_over_10s, 2),
            },
            "strategy_distribution": dict(strategy_counts),
            "difficulty_matrix": {f"Req {r}": {f"Ass {a}": diff_matrix[r][a] for a in range(1, 6)} for r in range(1, 6)},
            "topics": {
                "unique_topics_count": len(total_topics_set),
                "junk_topics_count": len(junk_topics_set),
                "junk_topic_rate_pct": round(junk_topic_rate, 2),
                "sample_junk_topics": sorted(list(junk_topics_set))[:10],
                "avg_coverage_pct": round(avg_coverage, 2),
            },
            "pii_scan": pii_violations
        }
    finally:
        db.close()


def print_report(data: Dict[str, Any], title: str = "INTERVAI SESSION & TRACE STATE EVALUATION"):
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print(f"Total Sessions:  {data['total_sessions']}")
    print(f"Total Questions: {data['total_questions']}")
    print(f"Total Traces:    {data['total_traces']}")
    print("-" * 80)

    # Latencies
    lat = data["latency"]
    print("1. LATENCY DISTRIBUTION (from retrieval_traces)")
    print(f"   Traces Measured:     {lat['count']}")
    print(f"   p50 (Median):        {lat['p50_ms']:>10.2f} ms ({lat['p50_ms']/1000:.2f}s)")
    print(f"   p90:                 {lat['p90_ms']:>10.2f} ms ({lat['p90_ms']/1000:.2f}s)")
    print(f"   p95:                 {lat['p95_ms']:>10.2f} ms ({lat['p95_ms']/1000:.2f}s)")
    print(f"   Max:                 {lat['max_ms']:>10.2f} ms ({lat['max_ms']/1000:.2f}s)")
    print(f"   Share > 10s:         {lat['share_over_10s_pct']:>10.2f} %")
    print("-" * 80)

    # Strategy Distribution
    print("2. STRATEGY DISTRIBUTION")
    strat = data["strategy_distribution"]
    tot_strat = sum(strat.values()) or 1
    for s_name, count in sorted(strat.items(), key=lambda x: x[1], reverse=True):
        pct = (count / tot_strat) * 100
        print(f"   - {s_name:<30} {count:>5} ({pct:>5.1f}%)")
    print("-" * 80)

    # Difficulty Matrix
    print("3. REQUESTED VS ASSESSED DIFFICULTY MATRIX")
    header = "Req \\ Ass | " + " | ".join(f"Level {i}" for i in range(1, 6))
    print("   " + header)
    print("   " + "-" * len(header))
    matrix = data["difficulty_matrix"]
    for req_d in range(1, 6):
        row_str = f"   Req L{req_d}  | "
        row_vals = [f"{matrix.get(f'Req {req_d}', {}).get(f'Ass {a}', 0):>7}" for a in range(1, 6)]
        print(row_str + " | ".join(row_vals))
    print("-" * 80)

    # Junk Topic Rate
    top = data["topics"]
    print("4. TOPIC EXTRACTION & COVERAGE METRICS")
    print(f"   Unique Topics:       {top['unique_topics_count']}")
    print(f"   Junk Topics:         {top['junk_topics_count']}")
    print(f"   Junk Topic Rate:     {top['junk_topic_rate_pct']:>6.2f} %")
    print(f"   Average Coverage:    {top['avg_coverage_pct']:>6.2f} %")
    if top["sample_junk_topics"]:
        print(f"   Sample Junk Topics:  {top['sample_junk_topics'][:6]}")
    print("-" * 80)

    # PII Scan
    pii = data["pii_scan"]
    print("5. UNREDACTED PII SCAN")
    print(f"   resume_chunks with PII:        {pii['resume_chunks']}")
    print(f"   resume_claims with PII:         {pii['resume_claims']}")
    print(f"   relationships with PII:         {pii['relationships']}")
    print(f"   interview_questions with PII:   {pii['interview_questions']}")
    tot_pii = sum(pii.values())
    print(f"   Total Unredacted Violations:    {tot_pii}")
    print("=" * 80)


if __name__ == "__main__":
    results = run_evaluation()
    print_report(results)
    # Save JSON snapshot for comparison
    output_path = os.path.join(project_root, "scripts", "latest_eval_snapshot.json")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Could not save snapshot: {e}")
