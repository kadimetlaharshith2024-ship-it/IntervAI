import re
import hashlib
from typing import List
from sqlalchemy.orm import Session
from app.models.relationship import Relationship
from app.models.rag import ResumeClaim
from app.core.canonicalizer import canonicalize_entity, CANONICAL_TAXONOMY
from app.services.privacy_guard import privacy_guard, ALL_REDACTION_MARKERS

class ClaimService:
    @staticmethod
    def generate_claim_id(resume_id: str, text: str) -> str:
        h = hashlib.sha256(f"{resume_id}_{text}".encode()).hexdigest()[:12]
        return f"clm_{h}"

    def extract_and_store_claims(self, resume_id, db: Session) -> List[ResumeClaim]:
        # Pull Phase 4 verified relationships to build Grounded Atomic Claims
        rels = db.query(Relationship).filter(Relationship.resume_id == resume_id).all()
        
        db.query(ResumeClaim).filter(ResumeClaim.resume_id == resume_id).delete()
        stored_claims = []

        seen_claims = set()

        for r in rels:
            # Skip any relationship built on redaction markers
            if any(rm.lower() in (r.target_entity or "").lower() or rm.lower() in (r.source_entity or "").lower() for rm in ALL_REDACTION_MARKERS):
                continue
            if "redact" in (r.target_entity or "").lower() or "redact" in (r.source_entity or "").lower():
                continue

            # Construct active claim expression based on relationship type
            if r.relation_type == "DEVELOPED":
                if r.section == "Projects":
                    claim_text = f"Developed the {r.target_entity} platform."
                else:
                    claim_text = f"Contributed to {r.target_entity}."
            elif r.relation_type in ["WORKED_AT", "EMPLOYED_AT"]:
                claim_text = f"Worked at {r.target_entity}."
            elif r.relation_type in ["WORKED_AS", "ROLE"]:
                claim_text = f"Worked as {r.target_entity}."
            elif r.relation_type in ["SECURED_WITH", "AUTHENTICATED_WITH"]:
                claim_text = f"Secured {r.source_entity} using {r.target_entity}."
            elif r.relation_type == "CONTAINERIZED_WITH":
                claim_text = f"Containerized services with {r.target_entity} in {r.source_entity}."
            elif r.relation_type == "DEPLOYED_ON":
                claim_text = f"Deployed {r.source_entity} on {r.target_entity}."
            elif r.relation_type in ["BUILT_WITH", "IMPLEMENTED_WITH"]:
                if r.source_entity == "Candidate":
                    if r.context_entity:
                        claim_text = f"Built systems using {r.target_entity} at {r.context_entity}."
                    else:
                        claim_text = f"Built systems using {r.target_entity}."
                else:
                    claim_text = f"Built {r.source_entity} using {r.target_entity}."
            elif r.relation_type == "HAS_TECHNOLOGY":
                claim_text = f"Employed {r.target_entity} in {r.source_entity}."
            elif r.relation_type == "STUDIED":
                claim_text = f"Studied {r.target_entity}."
            elif r.relation_type == "ATTENDED":
                claim_text = f"Candidate ATTENDED {r.target_entity}."
            elif r.relation_type == "EARNED_CERTIFICATION":
                claim_text = f"Earned certification in {r.target_entity}."
            elif r.relation_type == "PURSUING_CERTIFICATION":
                claim_text = f"Currently pursuing {r.target_entity} certification (in progress)."
            elif r.relation_type == "COMPLETED_COURSE":
                claim_text = f"Completed course {r.target_entity}."
            elif r.relation_type == "WON":
                if r.context_entity:
                    claim_text = f"Won {r.target_entity} at {r.context_entity}."
                else:
                    claim_text = f"Won {r.target_entity}."
            elif r.relation_type == "PARTICIPATED_IN":
                claim_text = f"Participated in {r.target_entity}."
            elif r.relation_type == "ACHIEVED":
                claim_text = f"Achieved {r.target_entity}."
            elif r.relation_type == "SPEAKS":
                claim_text = f"Fluent in {r.target_entity}."
            else:
                claim_text = f"{r.source_entity} {r.relation_type} {r.target_entity}."

            # Ensure claim_text and evidence_text are completely sanitized
            claim_text, _ = privacy_guard.sanitize_text(claim_text)
            clean_ev, _ = privacy_guard.sanitize_text(r.evidence_text or "")

            if claim_text in seen_claims:
                continue
            seen_claims.add(claim_text)

            # Verification: Check whether the claim is actually supported by the underlying evidence.
            ev_lower = clean_ev.lower()
            claim_status = "SUPPORTED"
            if r.relation_type == "DEVELOPED":
                if r.section != "Projects" or any(w in ev_lower for w in ["intern", "employed", "employee", "trainee", "student"]):
                    claim_status = "UNSUPPORTED"
            elif r.relation_type in ["WON", "PARTICIPATED_IN", "PURSUING_CERTIFICATION", "COMPLETED_COURSE", "ACHIEVED"]:
                target_tokens = [
                    tok for tok in re.split(r"[\s\-_:,|()]+", (r.target_entity or "").lower())
                    if len(tok) >= 3 and tok not in ["and", "the", "for", "with", "udemy", "course", "cert", "hackathon"]
                ]
                if target_tokens and not any(t in ev_lower for t in target_tokens):
                    claim_status = "UNSUPPORTED"
            elif r.target_entity:
                target_norm = r.target_entity.lower().strip()
                canon_name, _ = canonicalize_entity(r.target_entity)
                canon_entry = CANONICAL_TAXONOMY.get(canon_name, {})
                aliases = [a.lower().strip() for a in canon_entry.get("aliases", [])]
                match_terms = {target_norm, canon_name.lower().strip(), *aliases} - {""}
                if not any(term in ev_lower for term in match_terms):
                    claim_status = "UNSUPPORTED"

            cid = self.generate_claim_id(str(resume_id), claim_text)
            claim_rec = ResumeClaim(
                resume_id=resume_id,
                claim_id=cid,
                claim_text=claim_text,
                section=r.section,
                project=r.context_entity,
                target_entity=r.target_entity,
                relation_type=r.relation_type,
                evidence_text=clean_ev,
                confidence=r.confidence,
                status=claim_status,
                provenance=r.extraction_method
            )
            db.add(claim_rec)
            stored_claims.append(claim_rec)

        db.commit()
        return stored_claims

claim_service = ClaimService()