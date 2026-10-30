import re
import logging
from enum import Enum
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class PrivacyCategory(str, Enum):
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    LINKEDIN = "LINKEDIN"
    PROFILE = "PROFILE"
    CAMPUS_ID = "CAMPUS_ID"
    CUSTOM_IDENTIFIER = "CUSTOM_IDENTIFIER"


@dataclass
class PrivacyViolation:
    category: PrivacyCategory
    char_start: int
    char_end: int
    confidence: float
    description: str


@dataclass
class PrivacyScreeningResult:
    is_clean: bool
    violations: List[PrivacyViolation] = field(default_factory=list)
    violation_categories: Set[str] = field(default_factory=set)

    @property
    def has_violations(self) -> bool:
        return not self.is_clean or len(self.violations) > 0


@dataclass
class PrivacyScreeningConfig:
    check_email: bool = True
    check_phone: bool = True
    check_linkedin: bool = True
    check_campus_id: bool = True
    custom_patterns: List[str] = field(default_factory=list)


# Words that should NEVER be flagged as campus/student IDs
TECHNICAL_ACRONYM_EXCLUSIONS = {
    "oauth2", "http2", "sha256", "sha512", "sha1", "utf8", "utf16",
    "ec2", "s3", "uuid4", "uuid", "es6", "css3", "html5", "ipv4", "ipv6",
    "h264", "mp4", "mp3", "rsa2048", "jwt", "b2b", "b2c", "p2p", "ci/cd",
    "k8s", "c3p0", "pydantic2", "gpt3", "gpt4", "llama2", "llama3",
    "arm64", "x86_64", "win32", "web3", "web2", "react18", "react19",
    "vue3", "node20", "python312", "python311", "python3", "python2",
    "level1", "level2", "level3", "level4", "level5", "phase1", "phase2"
}

# Technical terms that follow 'github' but are NOT usernames/handles
GITHUB_TECH_TERMS = {
    "actions", "action", "pages", "api", "copilot", "enterprise", "workflow", "workflows", "cli", "desktop"
}

# Standardized redaction replacement markers
REDACTION_MARKERS: Dict[PrivacyCategory, str] = {
    PrivacyCategory.EMAIL: "[REDACTED_EMAIL]",
    PrivacyCategory.PHONE: "[REDACTED_PHONE]",
    PrivacyCategory.LINKEDIN: "[REDACTED_PROFILE]",
    PrivacyCategory.PROFILE: "[REDACTED_PROFILE]",
    PrivacyCategory.CAMPUS_ID: "[REDACTED_ID]",
    PrivacyCategory.CUSTOM_IDENTIFIER: "[REDACTED_IDENTIFIER]",
}

ALL_REDACTION_MARKERS = {
    "[REDACTED_EMAIL]",
    "[REDACTED_PHONE]",
    "[REDACTED_PROFILE]",
    "[REDACTED_ID]",
    "[REDACTED_IDENTIFIER]",
    "[REDACTED_LINKEDIN]",
    "[REDACTED]"
}

# Common technical port numbers to ignore
COMMON_TECHNICAL_PORTS = {
    "80", "443", "8080", "8443", "3000", "5000", "8000", "5432", "3306",
    "27017", "6379", "9200", "9092", "4200", "5173", "8888", "9000"
}


class PrivacyScreeningService:
    """
    Privacy screening gate executed before resume content is indexed, embedded,
    stored persistently in knowledge representations, or submitted to LLMs.
    Detects personal/institutional emails, phone numbers, LinkedIn profile URLs,
    and campus/student registration IDs with strict false-positive prevention.
    """

    def __init__(self, config: Optional[PrivacyScreeningConfig] = None):
        self.config = config or PrivacyScreeningConfig()

        # 1. Email Patterns (Personal & Institutional, Standard, Odd-Spaced, Labeled)
        self._email_standard_regex = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            re.IGNORECASE
        )
        self._email_odd_spaced_regex = re.compile(
            r"\b[A-Za-z0-9._%+-]+(?:\s*[._%+-]\s*[A-Za-z0-9]+)*\s*@\s*[A-Za-z0-9.-]+(?:\s*\.\s*[A-Za-z0-9.-]+)*\s*\.\s*[A-Za-z]{2,}\b",
            re.IGNORECASE
        )
        self._email_labeled_regex = re.compile(
            r"(?i)\b(?:e|email|mail|contact\s*email)[:\s]+([A-Za-z0-9._%+-]+(?:\s*[._%+-]\s*[A-Za-z0-9]+)*\s*@\s*[A-Za-z0-9.-]+(?:\s*\.\s*[A-Za-z0-9.-]+)*\s*\.\s*[A-Za-z]{2,})\b"
        )
        self._email_obfuscated_regex = re.compile(
            r"\b[A-Za-z0-9._%+-]+\s*(?:\[\s*at\s*\]|\(\s*at\s*\))\s*[A-Za-z0-9.-]+\s*(?:\[\s*dot\s*\]|\(\s*dot\s*\)|\.)\s*[A-Za-z]{2,}\b",
            re.IGNORECASE
        )

        # 2. LinkedIn & Social Profile Patterns (Handles, URLs, Shortlinks)
        self._linkedin_regex = re.compile(
            r"(?:https?:\/\/)?(?:www\.)?linkedin\.com\/(?:in|pub|profile)\/[A-Za-z0-9_\-\/]+",
            re.IGNORECASE
        )
        self._linkedin_short_regex = re.compile(
            r"(?<!\w)(?:in\/[A-Za-z0-9_\-]{3,50})\b",
            re.IGNORECASE
        )
        self._linkedin_labeled_regex = re.compile(
            r"(?i)\blinkedin[:\s]+(?:in\/|@)?([A-Za-z0-9_\-\.]{3,50})\b"
        )
        self._linkedin_obfuscated_regex = re.compile(
            r"\blinkedin(?:\.com|\s*\[\s*dot\s*\]\s*com)?\s*[\/\\]\s*(?:in|pub)\s*[\/\\]\s*[A-Za-z0-9_\-]+",
            re.IGNORECASE
        )

        self._github_url_regex = re.compile(
            r"(?:https?:\/\/)?(?:www\.)?github\.com\/[A-Za-z0-9_\-\.]+(?:\/[A-Za-z0-9_\-\.]+)?",
            re.IGNORECASE
        )
        self._github_labeled_regex = re.compile(
            r"(?i)\bgithub[:\s]+@?([A-Za-z0-9_\-\.]{3,39})\b"
        )
        self._other_profile_regex = re.compile(
            r"(?i)\b(?:leetcode|hackerrank|codechef|codeforces|kaggle|twitter|x\.com)[:\s]+@?([A-Za-z0-9_\-\.]{3,50})\b"
        )
        self._profile_url_regex = re.compile(
            r"(?:https?:\/\/)?(?:www\.)?(?:leetcode\.com\/u?|hackerrank\.com\/|kaggle\.com\/)[A-Za-z0-9_\-\.]+",
            re.IGNORECASE
        )

        # 3. Phone Patterns
        # Labeled phone numbers (e.g. T: 9876543210, Ph: +91 9876543210, Mobile: +91-9876543210)
        self._phone_labeled_regex = re.compile(
            r"(?i)(?:\b(?:t|tel|ph|phone|mob|mobile|cell|call|contact)[:\s]+|\b(?:t|ph|mob)[:\s]*)(?:(\+?[\d\s().-]{7,22}))\b"
        )
        # International with + (e.g. +1-202-555-0125, +91 98765 43210, +44 20 7946 0919)
        self._phone_international_regex = re.compile(
            r"(?<!\w)\+[1-9]\d{0,2}[-.\s]\(?\d{2,4}\)?[-.\s]\d{3,4}[-.\s]\d{3,5}\b"
        )
        # Domestic US/Canada standard (e.g. (123) 456-7890, 123-456-7890, 123.456.7890)
        self._phone_domestic_us_regex = re.compile(
            r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s][2-9]\d{2}[-.\s]\d{4}\b"
        )
        # 10-digit mobile (e.g. Indian/international mobile starting with 6-9, with/without +91)
        self._phone_10digit_regex = re.compile(
            r"(?<!\w)(?:\+?91[-.\s]?)?[6-9]\d{4}[-.\s]?\d{5}\b"
        )
        self._phone_10digit_spaced_regex = re.compile(
            r"(?<!\w)(?:\+?91[-.\s]?)?[6-9]\d{2}[-.\s]?\d{3}[-.\s]?\d{4}\b"
        )

        # 4. Campus Registration / Roll Number / Student ID Patterns
        # Context-labeled IDs: Roll No: 20BCE1042, Student ID: 12345678, Reg No: RA1911003010123
        self._campus_id_labeled_regex = re.compile(
            r"(?i)\b(?:roll\s*(?:no|number|#)?|student\s*(?:id|no|number|#)?|registration\s*(?:no|number|id|#)?|reg\s*(?:no|number|#)?|urn|enrollment\s*(?:no|number|#)?|admission\s*(?:no|number|#)?|hall\s*ticket\s*(?:no|number|#)?)[:\s#]*([A-Za-z0-9\/-]{4,24})\b"
        )
        # Unlabeled campus registration pattern (common in Indian/International universities):
        # 2-digit year + 2-4 letter branch + 4-6 digits (e.g. 20BCE1042, 21BCS0123, 19BIT0045)
        self._campus_id_format_regex = re.compile(
            r"\b\d{2}[A-Z]{2,4}\d{4,6}\b"
        )
        # SRM/VIT style RA format: RA followed by 13-15 digits
        self._campus_id_ra_format_regex = re.compile(
            r"\bRA\d{13,15}\b",
            re.IGNORECASE
        )

    def _is_safe_numeric_context(self, text: str, match_start: int, match_end: int) -> bool:
        """
        Validates that a numeric match is not a year, date range, port number,
        HTTP code, software version, IP address, or performance percentage.
        """
        matched_str = text[match_start:match_end].strip()
        cleaned_digits = re.sub(r"\D", "", matched_str)

        # Port number check
        if matched_str in COMMON_TECHNICAL_PORTS:
            return True

        # Year check (e.g. 1990 - 2035)
        if len(cleaned_digits) == 4 and 1980 <= int(cleaned_digits) <= 2035:
            return True

        # Date range check: "2020 - 2024", "2019-2023", "05/2021"
        window_start = max(0, match_start - 15)
        window_end = min(len(text), match_end + 15)
        context_window = text[window_start:window_end]

        if re.search(r"\b(?:19|20)\d{2}\s*[\u2013\u2014\-–/]\s*(?:(?:19|20)\d{2}|present|current)\b", context_window, re.IGNORECASE):
            return True
        if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b", context_window, re.IGNORECASE):
            return True

        # Version string check: "Python 3.12.1", "Node.js 18.2"
        if re.search(r"\b(?:v|version)?\s*\d+\.\d+(?:\.\d+)?\b", context_window, re.IGNORECASE):
            if matched_str in re.findall(r"\d+\.\d+(?:\.\d+)?", context_window):
                return True

        # HTTP status code or metric percentage: "404 Not Found", "99.9% uptime"
        if "%" in context_window or "uptime" in context_window.lower() or "latency" in context_window.lower():
            return True

        # IP address check
        if re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", context_window):
            return True

        # Phone number must contain at least 7 digits
        if len(cleaned_digits) < 7:
            return True

        return False

    def screen_text(self, text: str) -> PrivacyScreeningResult:
        """
        Screens extracted resume text for prohibited personal identifiers.
        Returns PrivacyScreeningResult with detected categories and safe descriptions.
        Does NOT echo raw PII in output descriptions or logs.
        """
        if not text or not text.strip():
            return PrivacyScreeningResult(is_clean=True)

        violations: List[PrivacyViolation] = []
        categories: Set[str] = set()

        # 1. Email Screening
        if self.config.check_email:
            for match in self._email_labeled_regex.finditer(text):
                violations.append(PrivacyViolation(
                    category=PrivacyCategory.EMAIL,
                    char_start=match.start(),
                    char_end=match.end(),
                    confidence=0.99,
                    description="Labeled email address detected."
                ))
                categories.add(PrivacyCategory.EMAIL.value)

            for match in self._email_standard_regex.finditer(text):
                violations.append(PrivacyViolation(
                    category=PrivacyCategory.EMAIL,
                    char_start=match.start(),
                    char_end=match.end(),
                    confidence=0.99,
                    description="Standard email address detected."
                ))
                categories.add(PrivacyCategory.EMAIL.value)

            for match in self._email_odd_spaced_regex.finditer(text):
                violations.append(PrivacyViolation(
                    category=PrivacyCategory.EMAIL,
                    char_start=match.start(),
                    char_end=match.end(),
                    confidence=0.97,
                    description="Odd-spaced email address detected."
                ))
                categories.add(PrivacyCategory.EMAIL.value)

            for match in self._email_obfuscated_regex.finditer(text):
                violations.append(PrivacyViolation(
                    category=PrivacyCategory.EMAIL,
                    char_start=match.start(),
                    char_end=match.end(),
                    confidence=0.95,
                    description="Obfuscated email address detected."
                ))
                categories.add(PrivacyCategory.EMAIL.value)

        # 2. LinkedIn & Profile Screening
        if self.config.check_linkedin:
            for match in self._linkedin_regex.finditer(text):
                violations.append(PrivacyViolation(
                    category=PrivacyCategory.LINKEDIN,
                    char_start=match.start(),
                    char_end=match.end(),
                    confidence=0.99,
                    description="LinkedIn profile link detected."
                ))
                categories.add(PrivacyCategory.LINKEDIN.value)
                categories.add(PrivacyCategory.PROFILE.value)

            for match in self._linkedin_short_regex.finditer(text):
                violations.append(PrivacyViolation(
                    category=PrivacyCategory.LINKEDIN,
                    char_start=match.start(),
                    char_end=match.end(),
                    confidence=0.95,
                    description="LinkedIn short handle link detected."
                ))
                categories.add(PrivacyCategory.LINKEDIN.value)
                categories.add(PrivacyCategory.PROFILE.value)

            for match in self._linkedin_labeled_regex.finditer(text):
                violations.append(PrivacyViolation(
                    category=PrivacyCategory.LINKEDIN,
                    char_start=match.start(),
                    char_end=match.end(),
                    confidence=0.96,
                    description="Labeled LinkedIn handle detected."
                ))
                categories.add(PrivacyCategory.LINKEDIN.value)
                categories.add(PrivacyCategory.PROFILE.value)

            for match in self._linkedin_obfuscated_regex.finditer(text):
                violations.append(PrivacyViolation(
                    category=PrivacyCategory.LINKEDIN,
                    char_start=match.start(),
                    char_end=match.end(),
                    confidence=0.92,
                    description="Obfuscated LinkedIn reference detected."
                ))
                categories.add(PrivacyCategory.LINKEDIN.value)
                categories.add(PrivacyCategory.PROFILE.value)

            for match in self._github_url_regex.finditer(text):
                violations.append(PrivacyViolation(
                    category=PrivacyCategory.PROFILE,
                    char_start=match.start(),
                    char_end=match.end(),
                    confidence=0.99,
                    description="GitHub profile URL detected."
                ))
                categories.add(PrivacyCategory.PROFILE.value)

            for match in self._github_labeled_regex.finditer(text):
                handle_str = match.group(1).lower().strip()
                if handle_str not in GITHUB_TECH_TERMS:
                    violations.append(PrivacyViolation(
                        category=PrivacyCategory.PROFILE,
                        char_start=match.start(),
                        char_end=match.end(),
                        confidence=0.96,
                        description="Labeled GitHub handle detected."
                    ))
                    categories.add(PrivacyCategory.PROFILE.value)

            for match in self._other_profile_regex.finditer(text):
                violations.append(PrivacyViolation(
                    category=PrivacyCategory.PROFILE,
                    char_start=match.start(),
                    char_end=match.end(),
                    confidence=0.94,
                    description="Developer profile handle detected."
                ))
                categories.add(PrivacyCategory.PROFILE.value)

            for match in self._profile_url_regex.finditer(text):
                violations.append(PrivacyViolation(
                    category=PrivacyCategory.PROFILE,
                    char_start=match.start(),
                    char_end=match.end(),
                    confidence=0.95,
                    description="Developer profile URL detected."
                ))
                categories.add(PrivacyCategory.PROFILE.value)

        # 3. Phone Screening
        if self.config.check_phone:
            # Labeled phone matches
            for match in self._phone_labeled_regex.finditer(text):
                val = match.group(1).strip()
                digits = re.sub(r"\D", "", val)
                if len(digits) >= 7 and not self._is_safe_numeric_context(text, match.start(), match.end()):
                    violations.append(PrivacyViolation(
                        category=PrivacyCategory.PHONE,
                        char_start=match.start(),
                        char_end=match.end(),
                        confidence=0.98,
                        description="Labeled phone or mobile number detected."
                    ))
                    categories.add(PrivacyCategory.PHONE.value)

            # International with prefix
            for match in self._phone_international_regex.finditer(text):
                if not self._is_safe_numeric_context(text, match.start(), match.end()):
                    violations.append(PrivacyViolation(
                        category=PrivacyCategory.PHONE,
                        char_start=match.start(),
                        char_end=match.end(),
                        confidence=0.95,
                        description="International phone number detected."
                    ))
                    categories.add(PrivacyCategory.PHONE.value)

            # Domestic US/Delimited
            for match in self._phone_domestic_us_regex.finditer(text):
                if not self._is_safe_numeric_context(text, match.start(), match.end()):
                    violations.append(PrivacyViolation(
                        category=PrivacyCategory.PHONE,
                        char_start=match.start(),
                        char_end=match.end(),
                        confidence=0.94,
                        description="Domestic phone number format detected."
                    ))
                    categories.add(PrivacyCategory.PHONE.value)

            # 10-digit mobile
            for match in self._phone_10digit_regex.finditer(text):
                if not self._is_safe_numeric_context(text, match.start(), match.end()):
                    violations.append(PrivacyViolation(
                        category=PrivacyCategory.PHONE,
                        char_start=match.start(),
                        char_end=match.end(),
                        confidence=0.92,
                        description="10-digit mobile number detected."
                    ))
                    categories.add(PrivacyCategory.PHONE.value)

            for match in self._phone_10digit_spaced_regex.finditer(text):
                if not self._is_safe_numeric_context(text, match.start(), match.end()):
                    violations.append(PrivacyViolation(
                        category=PrivacyCategory.PHONE,
                        char_start=match.start(),
                        char_end=match.end(),
                        confidence=0.92,
                        description="10-digit spaced mobile number detected."
                    ))
                    categories.add(PrivacyCategory.PHONE.value)

        # 4. Campus / Student ID Screening
        if self.config.check_campus_id:
            # Labeled registration/roll numbers
            for match in self._campus_id_labeled_regex.finditer(text):
                id_val = match.group(1).strip()
                id_clean = id_val.lower()
                if id_clean not in TECHNICAL_ACRONYM_EXCLUSIONS and len(re.sub(r"\D", "", id_val)) >= 3:
                    violations.append(PrivacyViolation(
                        category=PrivacyCategory.CAMPUS_ID,
                        char_start=match.start(),
                        char_end=match.end(),
                        confidence=0.98,
                        description="Labeled campus registration or roll number detected."
                    ))
                    categories.add(PrivacyCategory.CAMPUS_ID.value)

            # Structured university format (e.g. 20BCE1042)
            for match in self._campus_id_format_regex.finditer(text):
                matched_id = match.group(0)
                if matched_id.lower() not in TECHNICAL_ACRONYM_EXCLUSIONS:
                    violations.append(PrivacyViolation(
                        category=PrivacyCategory.CAMPUS_ID,
                        char_start=match.start(),
                        char_end=match.end(),
                        confidence=0.92,
                        description="University registration number format detected."
                    ))
                    categories.add(PrivacyCategory.CAMPUS_ID.value)

            # SRM/VIT style RA format
            for match in self._campus_id_ra_format_regex.finditer(text):
                violations.append(PrivacyViolation(
                    category=PrivacyCategory.CAMPUS_ID,
                    char_start=match.start(),
                    char_end=match.end(),
                    confidence=0.96,
                    description="Institutional student identifier detected."
                ))
                categories.add(PrivacyCategory.CAMPUS_ID.value)

        # 5. Custom configured patterns
        for custom_pat in self.config.custom_patterns:
            try:
                c_regex = re.compile(custom_pat, re.IGNORECASE)
                for match in c_regex.finditer(text):
                    violations.append(PrivacyViolation(
                        category=PrivacyCategory.CUSTOM_IDENTIFIER,
                        char_start=match.start(),
                        char_end=match.end(),
                        confidence=0.90,
                        description="Configured custom prohibited identifier detected."
                    ))
                    categories.add(PrivacyCategory.CUSTOM_IDENTIFIER.value)
            except Exception as pat_err:
                logger.warning("Invalid custom privacy pattern '%s': %s", custom_pat, pat_err)

        is_clean = len(violations) == 0

        if not is_clean:
            # Generic diagnostic log: strictly avoids logging any raw identifier string
            logger.info(
                "Resume privacy screening identified sensitive categories: %s (Total occurrences: %d)",
                sorted(list(categories)),
                len(violations)
            )

        return PrivacyScreeningResult(
            is_clean=is_clean,
            violations=violations,
            violation_categories=categories
        )

    def sanitize_text(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Sanitizes raw or previously stored text by redacting detected personal identifiers
        (emails, phone numbers, LinkedIn URLs, campus IDs) with standardized markers.
        Preserves technical terms, project titles, versions, ports, metrics, code, and surrounding context.
        Returns (sanitized_text, metadata).
        Guarantees zero raw PII is logged, persisted, or passed downstream.
        """
        if not text or not text.strip():
            return text or "", {
                "redaction_count": 0,
                "categories_redacted": [],
                "is_sanitized": False
            }

        # Perform detection pass
        screening_result = self.screen_text(text)
        if not screening_result.has_violations:
            return text, {
                "redaction_count": 0,
                "categories_redacted": [],
                "is_sanitized": False
            }

        # Merge overlapping violation spans
        sorted_violations = sorted(
            screening_result.violations,
            key=lambda v: (v.char_start, -v.char_end)
        )
        merged_spans: List[Tuple[int, int, PrivacyCategory]] = []
        for v in sorted_violations:
            if not merged_spans:
                merged_spans.append((v.char_start, v.char_end, v.category))
            else:
                last_start, last_end, last_cat = merged_spans[-1]
                if v.char_start <= last_end:
                    merged_spans[-1] = (last_start, max(last_end, v.char_end), last_cat)
                else:
                    merged_spans.append((v.char_start, v.char_end, v.category))

        # Perform redactions from right to left to maintain offset stability
        result_chars = list(text)
        categories_redacted: Set[str] = set()
        for start, end, category in sorted(merged_spans, key=lambda s: s[0], reverse=True):
            marker = REDACTION_MARKERS.get(category, "[REDACTED]")
            result_chars[start:end] = list(marker)
            categories_redacted.add(category.value)

        sanitized = "".join(result_chars)

        # Secondary verification pass for any remaining obfuscated tokens
        secondary_result = self.screen_text(sanitized)
        if secondary_result.has_violations:
            sec_sorted = sorted(secondary_result.violations, key=lambda v: (v.char_start, -v.char_end))
            sec_merged: List[Tuple[int, int, PrivacyCategory]] = []
            for v in sec_sorted:
                if not sec_merged:
                    sec_merged.append((v.char_start, v.char_end, v.category))
                else:
                    last_s, last_e, last_c = sec_merged[-1]
                    if v.char_start <= last_e:
                        sec_merged[-1] = (last_s, max(last_e, v.char_end), last_c)
                    else:
                        sec_merged.append((v.char_start, v.char_end, v.category))

            sec_chars = list(sanitized)
            for s, e, cat in sorted(sec_merged, key=lambda x: x[0], reverse=True):
                marker = REDACTION_MARKERS.get(cat, "[REDACTED]")
                sec_chars[s:e] = list(marker)
                categories_redacted.add(cat.value)
            sanitized = "".join(sec_chars)

        logger.info(
            "Privacy sanitization applied: %d redactions across categories: %s (zero raw PII logged)",
            len(merged_spans),
            sorted(list(categories_redacted))
        )

        return sanitized, {
            "redaction_count": len(merged_spans),
            "categories_redacted": sorted(list(categories_redacted)),
            "is_sanitized": True
        }


# Class aliases for unified sanitization service naming
PrivacySanitizationService = PrivacyScreeningService
privacy_guard = PrivacySanitizationService()
privacy_sanitizer = privacy_guard
