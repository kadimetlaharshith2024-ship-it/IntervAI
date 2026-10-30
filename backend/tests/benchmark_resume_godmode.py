import time
import pytest
from app.services.resume_service import segment_resume_sections, detect_sections
from app.services.entity_service import extract_entities_from_text
from app.services.relationship_service import chunk_project_blocks, deduplicate_and_resolve_conflicts, parse_linguistic_sentence, RelationshipTriplet
from app.core.canonicalizer import canonicalize_entity, extract_non_overlapping_entities, ALIAS_INDEX

class TestGodmodeBenchmarkSuite:
    """
    Measures quantifiable precision, recall, F1, and latency against hand-annotated benchmark ground truth.
    """

    BENCHMARK_DOCUMENT = """
ELENA ROSTOVA
Senior Full-Stack & ML Systems Engineer
elena.rostova@cloudscale.ai | +1 (555) 018-4422 | San Francisco, CA

PROFESSIONAL SUMMARY
Results-driven software architect with 6+ years of experience designing high-throughput distributed systems,
cloud-native microservices, and AI-powered applications using Java, Spring Boot, Python, and React.js.

TECHNICAL SKILLS
Languages: Java, Python, TypeScript, Go, SQL, HTML5, CSS3
Frontend: React.js, Next.js, Tailwind CSS, SVG
Backend: Spring Boot, Spring Security, FastAPI, Node.js, JDBC
Databases: PostgreSQL, MySQL, Redis, MongoDB
AI/ML: PyTorch, Generative AI, Prompt Engineering, Hugging Face, LangChain
Tools & Platforms: Docker, Kubernetes, AWS, Git, GitHub, Postman, VS Code, Vercel
Core CS Concepts: Data Structures & Algorithms, OOP, DBMS, Computer Networks, Operating Systems, System Design

PROJECTS
NeuroFlow — Real-Time Biosignal Analytics Engine | Python, FastAPI, PyTorch, Redis
• Architected an asynchronous event-streaming pipeline processing 25,000 telemetry points per second.
• Integrated PyTorch transformer models for anomaly detection with response times under 15ms.
• Persisted windowed aggregations into Redis for ultra-low latency dashboard rendering.
GitHub Live Demo

OmniCommerce — Headless E-Commerce Platform | Java, Spring Boot, PostgreSQL, Docker
• Developed modular microservices using Spring Boot and Spring Security for role-based authorization.
• Integrated PostgreSQL with connection pooling to handle 10,000 concurrent checkout sessions.
• Containerized microservices using Docker and automated deployments via Vercel and AWS.
GitHub Live Demo

EDUCATION
University of Washington
Bachelor of Science in Computer Science, 2019

ACHIEVEMENTS & CERTIFICATIONS
• Winner - First Place at CloudAI Hackathon 2024 among 50 engineering teams.
• AWS Certified Solutions Architect - Associate
"""

    def test_benchmark_section_segmentation_metrics(self):
        start = time.perf_counter()
        sections = segment_resume_sections(self.BENCHMARK_DOCUMENT)
        detected = detect_sections(self.BENCHMARK_DOCUMENT)
        latency = (time.perf_counter() - start) * 1000

        expected_sections = {"skills", "experience", "projects", "education", "certifications", "achievements"}
        ground_truth_detected = {"skills": True, "experience": False, "projects": True, "education": True, "certifications": True, "achievements": True}

        tp = sum(1 for s in ground_truth_detected if ground_truth_detected[s] and detected.get(s))
        fp = sum(1 for s in ground_truth_detected if not ground_truth_detected[s] and detected.get(s))
        fn = sum(1 for s in ground_truth_detected if ground_truth_detected[s] and not detected.get(s))

        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 1.0

        assert precision == 1.0, f"Section Precision: {precision}"
        assert recall == 1.0, f"Section Recall: {recall}"
        assert f1 == 1.0, f"Section F1: {f1}"
        assert latency < 50.0, f"Segmentation latency {latency:.2f}ms exceeds 50ms budget"

    def test_benchmark_skill_extraction_and_canonicalization(self):
        start = time.perf_counter()
        entities = extract_entities_from_text(self.BENCHMARK_DOCUMENT)
        latency = (time.perf_counter() - start) * 1000

        extracted_names = {e.name for e in entities}
        ground_truth_skills = {
            "Java", "Python", "TypeScript", "Go", "SQL", "HTML5", "CSS3",
            "React", "Next.js", "Tailwind CSS", "SVG",
            "Spring Boot", "Spring Security", "FastAPI", "Node.js", "JDBC",
            "PostgreSQL", "MySQL", "Redis", "MongoDB",
            "PyTorch", "Generative AI", "Prompt Engineering", "Hugging Face", "LangChain",
            "Docker", "Kubernetes", "AWS", "Git", "GitHub", "Postman", "VS Code", "Vercel",
            "Data Structures & Algorithms", "OOP", "DBMS", "Computer Networks", "Operating Systems", "System Design"
        }

        tp = len(extracted_names.intersection(ground_truth_skills))
        fp = len(extracted_names - ground_truth_skills)
        fn = len(ground_truth_skills - extracted_names)

        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 1.0

        # Precision must be >= 95%, Recall >= 95%
        assert precision >= 0.95, f"Skill Precision {precision:.2f} below 0.95 (fp={fp})"
        assert recall >= 0.95, f"Skill Recall {recall:.2f} below 0.95 (fn={fn})"
        assert f1 >= 0.95, f"Skill F1 {f1:.2f} below 0.95"
        assert latency < 150.0, f"Skill extraction latency {latency:.2f}ms exceeds 150ms budget"

    def test_benchmark_project_boundary_and_relationship_accuracy(self):
        sections = segment_resume_sections(self.BENCHMARK_DOCUMENT)
        blocks = chunk_project_blocks(sections["projects"])

        # Boundary accuracy
        assert len(blocks) == 2
        assert blocks[0][0] == "NeuroFlow"
        assert blocks[1][0] == "OmniCommerce"

        known_skills = {k: v for k, v in ALIAS_INDEX.items()}
        triplets = []
        for title, hdr, lines in blocks:
            if hdr and title:
                ents = extract_non_overlapping_entities(hdr, known_skills)
                for canon, _, _ in ents:
                    if canon.lower() not in ["github", "general", "unknown", "redacted"]:
                        triplets.append(RelationshipTriplet(
                            source_entity=title,
                            relation_type="BUILT_WITH",
                            target_entity=canon,
                            section="Projects",
                            context_entity=title,
                            evidence_text=hdr.strip(),
                            confidence=0.98,
                            extraction_method="section_structural"
                        ))
            for line in lines:
                if "github" in line.lower() and "demo" in line.lower():
                    continue
                triplets.extend(parse_linguistic_sentence(line, title, "Projects", known_skills))

        final_triplets = deduplicate_and_resolve_conflicts(triplets)
        edge_pairs = {(t.source_entity, t.target_entity) for t in final_triplets}

        # Verify zero unsupported GitHub links
        assert ("NeuroFlow", "GitHub") not in edge_pairs
        assert ("OmniCommerce", "GitHub") not in edge_pairs

        # Verify genuine project technologies
        assert ("NeuroFlow", "Python") in edge_pairs
        assert ("NeuroFlow", "FastAPI") in edge_pairs
        assert ("NeuroFlow", "PyTorch") in edge_pairs
        assert ("NeuroFlow", "Redis") in edge_pairs
        assert ("OmniCommerce", "Java") in edge_pairs
        assert ("OmniCommerce", "Spring Boot") in edge_pairs
        assert ("OmniCommerce", "PostgreSQL") in edge_pairs
        assert ("OmniCommerce", "Docker") in edge_pairs
