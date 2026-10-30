import pytest
from app.core.canonicalizer import canonicalize_entity, is_same_entity
from app.services.relationship_service import (
    parse_tech_stack_line,
    parse_linguistic_sentence,
    parse_education_section,
    parse_language_section,
    deduplicate_and_resolve_conflicts,
    TECH_STACK_PREFIXES
)
from app.schemas.relationship import RelationshipTriplet

def test_entity_canonicalization():
    assert canonicalize_entity("react.js")[0] == "React"
    assert canonicalize_entity("ReactJS")[0] == "React"
    assert canonicalize_entity("postgres")[0] == "PostgreSQL"
    assert canonicalize_entity("tailwind")[0] == "Tailwind CSS"
    assert canonicalize_entity("spring boot")[0] == "Spring Boot"
    assert canonicalize_entity("spring")[0] == "Spring Framework"
    assert is_same_entity("React", "react.js") is True
    assert is_same_entity("Spring Boot", "Spring Framework") is False

def test_tech_stack_anti_hallucination():
    line = "Tech Stack: React (Vite, Tailwind CSS), Java, Spring Boot, PostgreSQL, JWT"
    triplets = parse_tech_stack_line(line, "TalentBridge", "Projects")
    
    # Must connect Project to Technologies
    targets = [t.target_entity for t in triplets]
    assert "React" in targets
    assert "Spring Boot" in targets
    assert "PostgreSQL" in targets

    # STRICT ASSERTION: Zero arbitrary pairwise relations
    for t in triplets:
        assert t.source_entity == "TalentBridge"
        assert t.relation_type == "HAS_TECHNOLOGY"
        assert t.target_entity not in ["Java -> CSS", "CSS"]

def test_explicit_implementation_sentence():
    sentence = "Built a Spring Boot backend using Java and PostgreSQL."
    known = {"spring boot": "Spring Boot", "java": "Java", "postgresql": "PostgreSQL"}
    triplets = parse_linguistic_sentence(sentence, "TalentBridge", "Projects", known)
    
    # Must capture genuine relations
    found_relations = {(t.source_entity, t.relation_type, t.target_entity) for t in triplets}
    assert ("Spring Boot", "BUILT_WITH", "Java") in found_relations or ("TalentBridge", "BUILT_WITH", "Spring Boot") in found_relations
    assert ("Spring Boot", "USES", "PostgreSQL") in found_relations or ("Spring Boot", "BUILT_WITH", "PostgreSQL") in found_relations
    # Must not invent nonsense
    assert ("Java", "IMPLEMENTED_WITH", "PostgreSQL") not in found_relations

def test_security_context():
    sentence = "Secured REST APIs using JWT and Spring Security."
    known = {"rest apis": "REST APIs", "jwt": "JWT", "spring security": "Spring Security"}
    triplets = parse_linguistic_sentence(sentence, "TalentBridge", "Projects", known)
    
    relations = {(t.relation_type, t.target_entity) for t in triplets}
    assert ("SECURED_WITH", "JWT") in relations or ("SECURED_WITH", "Spring Security") in relations

def test_education_extraction():
    text = "Bachelor of Technology, Computer Science and Engineering | 2024–2028\nVIT Vellore"
    triplets = parse_education_section(text)
    
    assert any(t.relation_type == "STUDIED" and "Computer Science" in t.target_entity for t in triplets)
    assert any(t.relation_type == "ATTENDED" and "VIT Vellore" in t.target_entity for t in triplets)

def test_language_not_classified_as_achievement():
    text = "English (Professional) | Telugu (Native) | Hindi (Conversational)"
    triplets = parse_language_section(text)
    
    assert len(triplets) == 3
    for t in triplets:
        assert t.relation_type == "SPEAKS"
        assert t.relation_type != "ACHIEVED"
        assert t.section == "Languages"

def test_duplicate_alias_deduplication():
    # If React and React.js are extracted in the same scope, resolve to 1 canonical node
    raw = [
        RelationshipTriplet(
            source_entity="TalentBridge",
            relation_type="HAS_TECHNOLOGY",
            target_entity="React",
            section="Projects",
            evidence_text="Built with React",
            confidence=0.9
        ),
        RelationshipTriplet(
            source_entity="TalentBridge",
            relation_type="HAS_TECHNOLOGY",
            target_entity="React.js",
            section="Projects",
            evidence_text="Built with React.js",
            confidence=0.85
        )
    ]
    resolved = deduplicate_and_resolve_conflicts(raw)
    assert len(resolved) == 1
    assert resolved[0].target_entity == "React"
    assert resolved[0].confidence == 0.9

def test_distinct_relation_types_not_merged():
    # Genuinely distinct relation types (e.g. BUILT_WITH vs HAS_TECHNOLOGY) between the same entities must NOT merge
    raw = [
        RelationshipTriplet(
            source_entity="TalentBridge",
            relation_type="BUILT_WITH",
            target_entity="React",
            section="Projects",
            evidence_text="Engineered core UI built with React",
            confidence=0.9
        ),
        RelationshipTriplet(
            source_entity="TalentBridge",
            relation_type="HAS_TECHNOLOGY",
            target_entity="React",
            section="Projects",
            evidence_text="Tech Stack: React",
            confidence=0.85
        )
    ]
    resolved = deduplicate_and_resolve_conflicts(raw)
    assert len(resolved) == 2
    types = {r.relation_type for r in resolved}
    assert types == {"BUILT_WITH", "HAS_TECHNOLOGY"}

def test_distinct_contexts_not_merged():
    # Same entity relationship occurring in different projects/contexts must remain distinct
    raw = [
        RelationshipTriplet(
            source_entity="Candidate",
            relation_type="USES",
            target_entity="PostgreSQL",
            section="Projects",
            context_entity="ProjectAlpha",
            evidence_text="Used PostgreSQL for relational schema in ProjectAlpha",
            confidence=0.92
        ),
        RelationshipTriplet(
            source_entity="Candidate",
            relation_type="USES",
            target_entity="PostgreSQL",
            section="Projects",
            context_entity="ProjectBeta",
            evidence_text="Used PostgreSQL for transaction logging in ProjectBeta",
            confidence=0.88
        )
    ]
    resolved = deduplicate_and_resolve_conflicts(raw)
    assert len(resolved) == 2
    contexts = {r.context_entity for r in resolved}
    assert contexts == {"ProjectAlpha", "ProjectBeta"}

def test_duplicate_alias_preserves_evidence_and_confidence():
    # When canonical aliases match in the same context, preserve maximum confidence and distinct evidence provenance
    raw = [
        RelationshipTriplet(
            source_entity="TalentBridge",
            relation_type="HAS_TECHNOLOGY",
            target_entity="Postgres",
            section="Projects",
            evidence_text="Configured Postgres database.",
            confidence=0.85
        ),
        RelationshipTriplet(
            source_entity="TalentBridge",
            relation_type="HAS_TECHNOLOGY",
            target_entity="PostgreSQL",
            section="Projects",
            evidence_text="Optimized PostgreSQL queries.",
            confidence=0.95
        )
    ]
    resolved = deduplicate_and_resolve_conflicts(raw)
    assert len(resolved) == 1
    assert resolved[0].target_entity == "PostgreSQL"
    assert resolved[0].confidence == 0.95
    assert "Optimized PostgreSQL queries." in resolved[0].evidence_text
    assert "Configured Postgres database." in resolved[0].evidence_text


def test_cross_sentence_isolation():
    # Two distinct sentences should never bridge entities across them
    sentence1 = "Built a React frontend."
    sentence2 = "PostgreSQL was used in another database cluster."
    known = {"react": "React", "postgresql": "PostgreSQL"}
    
    triplets1 = parse_linguistic_sentence(sentence1, "ProjA", "Projects", known)
    triplets2 = parse_linguistic_sentence(sentence2, "ProjB", "Projects", known)
    
    combined = triplets1 + triplets2
    for t in combined:
        assert not (t.source_entity == "React" and t.target_entity == "PostgreSQL")