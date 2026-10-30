import sys
sys.path.insert(0, '.')
import re
from typing import Dict, List, Any

from app.core.canonicalizer import (
    clean_project_title,
    is_tech_stack_or_tool_list,
    is_patent_or_metadata,
    canonicalize_entity
)
from app.services.resume_service import detect_sections, segment_resume_sections
from app.services.relationship_service import (
    parse_certifications_section,
    parse_achievements_section,
    parse_linguistic_sentence,
    parse_tech_stack_line
)

BENCHMARK_CASES = [
    # 1. Combined Headings (Leadership, Achievements & Certifications)
    {
        "id": "BM-01-Combined-Headings",
        "category": "Combined Headings",
        "raw_text": """
ALEX MORGAN
alex@example.com

EDUCATION
B.Tech in Computer Science, 2024

PROJECTS
EventHub | React, FastAPI, Docker
- Built event ticketing system.

LEADERSHIP, ACHIEVEMENTS & CERTIFICATIONS
● Switchuation Hackathon (Sept 2024): Winner - Best Freshers Award; led a team of 4 to build a working prototype in 12 hours, ranking 1st among 32 teams.
● Hackathon Participant (SDG-focused MVP) - Celestia
● AWS Certified Solutions Architect - Associate (In Progress) | Mastering DSA in C++ (Udemy) | Java Masterclass (Udemy)
""",
        "expected_sections": {"projects": True, "education": True, "achievements": True, "certifications": True},
        "forbidden_project_titles": [
            "Leadership, Achievements & Certifications",
            "LEADERSHIP, ACHIEVEMENTS & CERTIFICATIONS",
            "Achievements & Certifications",
            "Best Freshers Award",
            "Switchuation Hackathon",
            "AWS Certified Solutions Architect - Associate (In Progress)"
        ],
        "expected_relations": [
            ("Candidate", "WON", "Best Freshers Award"),
            ("Candidate", "PARTICIPATED_IN", "Switchuation Hackathon"),
            ("Candidate", "PURSUING_CERTIFICATION", "AWS Certified Solutions Architect - Associate"),
            ("Candidate", "COMPLETED_COURSE", "Mastering DSA in C++ (Udemy)")
        ]
    },

    # 2. Tech Stack Listed as Pipe Header or Bullet (Must not become a project)
    {
        "id": "BM-02-Tech-Stack-Titles",
        "category": "Tech Stack Disambiguation",
        "raw_text": """
SARAH CONNER
sarah@example.com

PROJECTS
SmartGrid
Python, FastAPI, Pydantic v2, Gemini, React.js
Java, Spring Boot, PostgreSQL, Docker
- Deployed grid telemetry system.
""",
        "expected_sections": {"projects": True},
        "forbidden_project_titles": [
            "Python, FastAPI, Pydantic v2, Gemini, React.js",
            "Java, Spring Boot, PostgreSQL, Docker"
        ],
        "expected_project_titles": [
            "SmartGrid"
        ]
    },

    # 3. Patent & Publication Metadata (Must not become a platform)
    {
        "id": "BM-03-Patent-Metadata",
        "category": "Patent Metadata",
        "raw_text": """
DAVID SMITH
david@example.com

PROJECTS
RoadGuard System | C++, Embedded C, ESP32-CAM (Indian Patent Filed)
202641102290 (Published)
- Real-time road anomaly detection.
""",
        "expected_sections": {"projects": True},
        "forbidden_project_titles": [
            "202641102290 (Published)",
            "Indian Patent Filed",
            "202641102290"
        ],
        "expected_project_titles": [
            "RoadGuard System"
        ]
    },

    # 4. Global Skills Isolation (Anti-Skill Bleed)
    {
        "id": "BM-04-Global-Skills-Bleed",
        "category": "Skill Bleed Prevention",
        "raw_text": """
EMILY CHEN
emily@example.com

TECHNICAL SKILLS
Languages: C++, Java, Python, Go, Rust, Ruby, PHP
Frameworks: React, Angular, Vue, Django, Spring Boot

PROJECTS
DataPipeline | Python, FastAPI
- Built streaming data pipeline with Kafka.
""",
        "expected_sections": {"skills": True, "projects": True},
        "forbidden_project_skills": {
            "DataPipeline": ["C++", "Java", "Go", "Rust", "Ruby", "PHP", "Angular", "Vue", "Spring Boot"]
        }
    },

    # 5. Coursework vs Accredited Certification
    {
        "id": "BM-05-Course-vs-Cert",
        "category": "Certification Semantics",
        "raw_text": """
MARK ROBERTS
mark@example.com

CERTIFICATIONS
- Certified Kubernetes Administrator (CKA)
- Machine Learning Specialization (Coursera)
- AWS Certified Developer - Associate (In Progress)
- Complete Python Bootcamp (Udemy)
""",
        "expected_sections": {"certifications": True},
        "expected_relations": [
            ("Candidate", "EARNED_CERTIFICATION", "Certified Kubernetes Administrator (Cka)"),
            ("Candidate", "COMPLETED_COURSE", "Machine Learning Specialization (Coursera)"),
            ("Candidate", "PURSUING_CERTIFICATION", "AWS Certified Developer - Associate"),
            ("Candidate", "COMPLETED_COURSE", "Complete Python Bootcamp (Udemy)")
        ]
    }
]


def run_benchmark() -> Dict[str, Any]:
    total_checks = 0
    passed_checks = 0
    failures = []

    # 1. Section Detection Accuracy
    sec_tp, sec_fp, sec_fn = 0, 0, 0
    for case in BENCHMARK_CASES:
        detected = detect_sections(case["raw_text"])
        for sec, expected_val in case.get("expected_sections", {}).items():
            total_checks += 1
            actual_val = detected.get(sec, False)
            if actual_val == expected_val:
                passed_checks += 1
                if expected_val:
                    sec_tp += 1
            else:
                failures.append(f"[{case['id']}] Section '{sec}' expected {expected_val}, got {actual_val}")
                if expected_val:
                    sec_fn += 1
                else:
                    sec_fp += 1

    # 2. Heading Hijacking & Forbidden Title Checks
    for case in BENCHMARK_CASES:
        for forbidden in case.get("forbidden_project_titles", []):
            total_checks += 1
            cleaned = clean_project_title(forbidden)
            if cleaned is None:
                passed_checks += 1
            else:
                failures.append(f"[{case['id']}] Forbidden title '{forbidden}' was accepted as '{cleaned}'")

        for expected in case.get("expected_project_titles", []):
            total_checks += 1
            cleaned = clean_project_title(expected)
            if cleaned is not None:
                passed_checks += 1
            else:
                failures.append(f"[{case['id']}] Valid title '{expected}' was rejected")

    # 3. Certification vs Course Classification
    for case in BENCHMARK_CASES:
        if "CERTIFICATIONS" in case["raw_text"] or "LEADERSHIP, ACHIEVEMENTS & CERTIFICATIONS" in case["raw_text"]:
            segmented = segment_resume_sections(case["raw_text"])
            cert_text = segmented.get("certifications", "")
            achieve_text = segmented.get("achievements", "")
            triplets = parse_certifications_section(cert_text) + parse_achievements_section(achieve_text)
            actual_triplet_set = {(t.source_entity, t.relation_type, t.target_entity.lower()) for t in triplets}

            for exp_src, exp_rel, exp_tgt in case.get("expected_relations", []):
                total_checks += 1
                matched = any(
                    src == exp_src and rel == exp_rel and exp_tgt.lower() in tgt
                    for src, rel, tgt in actual_triplet_set
                )
                if matched:
                    passed_checks += 1
                else:
                    failures.append(f"[{case['id']}] Expected relation ({exp_src}, {exp_rel}, {exp_tgt}) not found in {actual_triplet_set}")

    # 4. Anti-Skill Bleed Checks
    for case in BENCHMARK_CASES:
        for proj, forbidden_skills in case.get("forbidden_project_skills", {}).items():
            segmented = segment_resume_sections(case["raw_text"])
            proj_text = segmented.get("projects", "")
            # Check body sentences in project
            for line in proj_text.splitlines():
                parsed = parse_linguistic_sentence(line, proj, "Projects", {})
                for p in parsed:
                    if p.context_entity == proj:
                        total_checks += 1
                        if p.target_entity in forbidden_skills:
                            failures.append(f"[{case['id']}] Skill bleed: '{p.target_entity}' falsely linked to project '{proj}'")
                        else:
                            passed_checks += 1

    accuracy = (passed_checks / total_checks) * 100 if total_checks else 0.0
    return {
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "failed_checks": len(failures),
        "accuracy_pct": round(accuracy, 2),
        "failures": failures
    }


if __name__ == "__main__":
    res = run_benchmark()
    print("=== RESUME INTELLIGENCE BENCHMARK RESULTS ===")
    print(f"Total Test Dimensions: {res['total_checks']}")
    print(f"Passed Checks:         {res['passed_checks']}")
    print(f"Failed Checks:         {res['failed_checks']}")
    print(f"Benchmark Accuracy:    {res['accuracy_pct']}%")
    if res["failures"]:
        print("\nFailures:")
        for f in res["failures"]:
            print(f"  - {f}")
    else:
        print("\n100% of benchmark checks PASSED successfully with zero defects!")
