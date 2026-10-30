import re
from typing import Dict, Tuple, List, Optional

DEPLOYMENT_TARGETS = {
    "vercel": "Vercel",
    "netlify": "Netlify",
    "render": "Render",
    "heroku": "Heroku",
    "aws": "AWS",
    "gcp": "Google Cloud Platform",
    "azure": "Microsoft Azure",
    "firebase": "Firebase"
}

PROJECT_TITLE_BLACKLIST = {
    "github", "gitlab", "live demo", "demo", "view project", "link",
    "website", "code", "repo", "repository", "deployed", "live",
    "tech stack", "technologies", "tools", "environment", "frontend",
    "backend", "full stack developer", "developer", "engineer", "project",
    # Section heading names & combined headings
    "education", "academic background", "experience", "work experience",
    "professional experience", "employment history", "internships", "projects",
    "academic projects", "personal projects", "technical skills", "skills",
    "certifications", "licenses", "courses", "achievements", "honors", "awards",
    "leadership", "positions of responsibility", "extracurricular",
    "extracurricular activities", "publications", "patents", "languages",
    "summary", "professional summary", "profile", "objective",
    "leadership, achievements & certifications", "leadership and achievements",
    "leadership & achievements", "achievements & certifications",
    "achievements and certifications", "honors & awards", "awards & certifications",
    "awards & achievements", "certifications & courses", "leadership & extracurricular"
}

SECTION_KEYWORDS = {
    "education", "experience", "skills", "projects", "certifications", "achievements",
    "leadership", "honors", "awards", "publications", "patents", "extracurricular",
    "coursework", "summary", "objective"
}

CANONICAL_TAXONOMY: Dict[str, Dict[str, any]] = {
    # ------------------ LANGUAGES ------------------
    "Java": {"category": "Language", "aliases": ["java", "core java", "java 8", "java 11", "java 17", "java 21"], "parent": None},
    "Python": {"category": "Language", "aliases": ["python", "python3", "python 3", "py"], "parent": None},
    "JavaScript": {"category": "Language", "aliases": ["javascript", "js", "ecmascript", "vanilla javascript"], "parent": None},
    "TypeScript": {"category": "Language", "aliases": ["typescript", "ts"], "parent": "JavaScript"},
    "C++": {"category": "Language", "aliases": ["c++", "cpp"], "parent": "C"},
    "C": {"category": "Language", "aliases": ["c language", r"\bc\b"], "parent": None},
    "C#": {"category": "Language", "aliases": ["c#", "csharp", "c-sharp"], "parent": None},
    "Go": {"category": "Language", "aliases": ["go", "golang"], "parent": None},
    "Rust": {"category": "Language", "aliases": ["rust"], "parent": None},
    "SQL": {"category": "Language", "aliases": ["sql", "structured query language"], "parent": None},
    "HTML": {"category": "Language", "aliases": ["html"], "parent": None},
    "HTML5": {"category": "Frontend", "aliases": ["html5", "html 5"], "parent": "HTML"},
    "CSS": {"category": "Language", "aliases": ["css"], "parent": None},
    "CSS3": {"category": "Frontend", "aliases": ["css3", "css 3"], "parent": "CSS"},

    # ------------------ FRAMEWORKS & RUNTIMES ------------------
    "Spring Boot": {"category": "Framework", "aliases": ["spring boot", "springboot", "spring-boot"], "parent": "Spring Framework"},
    "Spring Framework": {"category": "Framework", "aliases": ["spring framework", "spring core", "spring"], "parent": "Java"},
    "Spring Security": {"category": "Framework", "aliases": ["spring security", "spring-security"], "parent": "Spring Framework"},
    "Spring Data JPA": {"category": "Framework", "aliases": ["spring data jpa", "jpa", "spring data"], "parent": "Spring Boot"},
    "Hibernate": {"category": "Framework", "aliases": ["hibernate", "hibernate orm"], "parent": "Java"},
    "JDBC": {"category": "Backend", "aliases": ["jdbc", "java database connectivity"], "parent": "Java"},
    "React": {"category": "Framework", "aliases": ["react", "react.js", "reactjs", "react js"], "parent": "JavaScript"},
    "Next.js": {"category": "Framework", "aliases": ["next.js", "nextjs", "next js"], "parent": "React"},
    "Node.js": {"category": "Runtime", "aliases": ["node.js", "nodejs", "node js", "node"], "parent": "JavaScript"},
    "Express.js": {"category": "Framework", "aliases": ["express", "express.js", "expressjs"], "parent": "Node.js"},
    "FastAPI": {"category": "Framework", "aliases": ["fastapi", "fast api"], "parent": "Python"},
    "Django": {"category": "Framework", "aliases": ["django", "django rest framework", "drf"], "parent": "Python"},
    "Flask": {"category": "Framework", "aliases": ["flask"], "parent": "Python"},
    "Tailwind CSS": {"category": "Framework", "aliases": ["tailwind", "tailwind css", "tailwindcss"], "parent": "CSS"},
    ".NET": {"category": "Framework", "aliases": [".net", "dotnet", ".net core", "asp.net"], "parent": "C#"},

    # ------------------ DATABASES & STORAGE ------------------
    "PostgreSQL": {"category": "Database", "aliases": ["postgresql", "postgres", "psql", "postgre sql"], "parent": "SQL"},
    "MySQL": {"category": "Database", "aliases": ["mysql", "my sql", "my-sql"], "parent": "SQL"},
    "MongoDB": {"category": "Database", "aliases": ["mongodb", "mongo", "mongo db"], "parent": "NoSQL"},
    "Redis": {"category": "Database", "aliases": ["redis"], "parent": "NoSQL"},
    "Elasticsearch": {"category": "Database", "aliases": ["elasticsearch", "elastic search", "elk"], "parent": None},
    "SQLite": {"category": "Database", "aliases": ["sqlite", "sqlite3"], "parent": "SQL"},

    # ------------------ CLOUD & DEVOPS ------------------
    "Docker": {"category": "Tool/DevOps", "aliases": ["docker", "dockerized", "docker-compose"], "parent": None},
    "Kubernetes": {"category": "Tool/DevOps", "aliases": ["kubernetes", "k8s"], "parent": None},
    "Vercel": {"category": "Cloud Platform", "aliases": ["vercel"], "parent": None},
    "Netlify": {"category": "Cloud Platform", "aliases": ["netlify"], "parent": None},
    "Render": {"category": "Cloud Platform", "aliases": ["render"], "parent": None},
    "AWS": {"category": "Cloud Platform", "aliases": ["aws", "amazon web services"], "parent": None},
    "Google Cloud Platform": {"category": "Cloud Platform", "aliases": ["gcp", "google cloud", "google cloud platform"], "parent": None},
    "Microsoft Azure": {"category": "Cloud Platform", "aliases": ["azure", "microsoft azure"], "parent": None},
    "Git": {"category": "Tool/DevOps", "aliases": ["git"], "parent": None},
    "GitHub": {"category": "Tool/DevOps", "aliases": ["github"], "parent": "Git"},
    "VS Code": {"category": "Tool/DevOps", "aliases": ["vs code", "vscode", "visual studio code"], "parent": None},
    "Postman": {"category": "Tool/DevOps", "aliases": ["postman"], "parent": None},
    "Linux": {"category": "Tool/DevOps", "aliases": ["linux", "ubuntu"], "parent": None},
    "Maven": {"category": "Tool/DevOps", "aliases": ["maven", "apache maven"], "parent": "Java"},
    "Vite": {"category": "Tool/DevOps", "aliases": ["vite", "vitejs", "vite.js"], "parent": "JavaScript"},

    # ------------------ AI & MACHINE LEARNING ------------------
    "Generative AI": {"category": "AI/ML", "aliases": ["generative ai", "genai", "gen ai"], "parent": None},
    "Prompt Engineering": {"category": "AI/ML", "aliases": ["prompt engineering", "prompt design"], "parent": "Generative AI"},
    "Hugging Face": {"category": "AI/ML", "aliases": ["hugging face", "huggingface", "hugging-face", "transformers"], "parent": None},
    "PyTorch": {"category": "AI/ML", "aliases": ["pytorch", "torch"], "parent": "Python"},
    "TensorFlow": {"category": "AI/ML", "aliases": ["tensorflow", "tf"], "parent": "Python"},
    "Scikit-Learn": {"category": "AI/ML", "aliases": ["scikit-learn", "sklearn"], "parent": "Python"},
    "LangChain": {"category": "AI/ML", "aliases": ["langchain"], "parent": "Python"},
    "LlamaIndex": {"category": "AI/ML", "aliases": ["llamaindex", "llama-index"], "parent": "Python"},

    # ------------------ CORE CS CONCEPTS & ARCHITECTURE ------------------
    "Data Structures & Algorithms": {"category": "Core Concept", "aliases": ["data structures & algorithms", "data structures and algorithms", "dsa", "data structures", "algorithms"], "parent": None},
    "OOP": {"category": "Core Concept", "aliases": ["oop", "oops", "object oriented programming", "object-oriented programming"], "parent": None},
    "DBMS": {"category": "Core Concept", "aliases": ["dbms", "database management system", "database management systems", "rdbms"], "parent": None},
    "Computer Networks": {"category": "Core Concept", "aliases": ["computer networks", "computer networking", "networking", "cn"], "parent": None},
    "Operating Systems": {"category": "Core Concept", "aliases": ["operating systems", "operating system", "os"], "parent": None},
    "System Design": {"category": "Core Concept", "aliases": ["system design"], "parent": None},
    "REST APIs": {"category": "Core Concept", "aliases": ["rest api", "rest apis", "restful api", "restful apis", "rest", "restful"], "parent": None},
    "JWT": {"category": "Core Concept", "aliases": ["jwt", "json web token", "json web tokens"], "parent": None},
    "OAuth2": {"category": "Core Concept", "aliases": ["oauth", "oauth2", "oauth 2.0"], "parent": None},
    "Microservices": {"category": "Core Concept", "aliases": ["microservices", "microservice architecture"], "parent": None},

    # ------------------ FRONTEND GRAPHICS & WEB LIBS ------------------
    "SVG": {"category": "Frontend", "aliases": ["svg", "scalable vector graphics"], "parent": None},
    "Google Fonts": {"category": "Frontend", "aliases": ["google fonts"], "parent": None},
    "MealDB API": {"category": "API", "aliases": ["mealdb api", "mealdb", "themealdb"], "parent": None},
    "Lovable": {"category": "Tool/DevOps", "aliases": ["lovable"], "parent": None},

    # ------------------ GEOSPATIAL & DATA STRUCTURES ------------------
    "QGIS": {"category": "Tool/GIS", "aliases": ["qgis", "quantum gis"], "parent": None},
    "Shapely": {"category": "Library", "aliases": ["shapely"], "parent": "Python"},
    "GeoJSON": {"category": "Data Format", "aliases": ["geojson", "geo json", "geo-json"], "parent": None},
    "R-Tree Indexing": {"category": "Data Structure/Algorithm", "aliases": ["r-tree indexing", "rtree indexing", "r-tree", "rtree", "spatial indexing", "r-tree spatial indexing"], "parent": None},

    # ------------------ LIBRARIES & PROTOCOLS ------------------
    "AsyncIO": {"category": "Library", "aliases": ["asyncio", "async-io", "python asyncio"], "parent": "Python"},
    "NumPy": {"category": "Library", "aliases": ["numpy"], "parent": "Python"},
    "Pandas": {"category": "Library", "aliases": ["pandas"], "parent": "Python"},
    "Flutter": {"category": "Framework", "aliases": ["flutter"], "parent": "Dart"},
    "Asterisk PBX": {"category": "Telephony/Infrastructure", "aliases": ["asterisk", "asterisk pbx", "pjsip", "audiosocket"], "parent": None},
    "Sarvam AI": {"category": "AI/ML Service", "aliases": ["sarvam ai", "sarvam", "bulbul"], "parent": None}
}

ALIAS_INDEX: Dict[str, str] = {}
for canonical, metadata in CANONICAL_TAXONOMY.items():
    ALIAS_INDEX[canonical.lower()] = canonical
    for alias in metadata["aliases"]:
        ALIAS_INDEX[alias.lower().strip()] = canonical

def strip_formatting_artifacts(text: str) -> str:
    """Removes bullet symbols and list indices while preserving ordinals (1st, 2nd, 3rd)."""
    if not text:
        return ""
    cleaned = re.sub(r"^[\s\u2022\u25cf\u25cb\u25aa\u25ab\u2013\u2014\-\*\#\:\)]+", "", text).strip()
    cleaned = re.sub(r"^\d+\s*[\.\)]\s*", "", cleaned).strip()
    cleaned = re.sub(r"(?i)\s*\((?:view[-_ ]*(?:certificate|credential|demo)?|cert|online)\)", "", cleaned).strip()
    return re.sub(r"\s+", " ", cleaned).strip()

sanitize_entity_label = strip_formatting_artifacts

def normalize_text_token(token: str) -> str:
    cleaned = strip_formatting_artifacts(token).lower()
    cleaned = re.sub(r"[\s\-_]+", " ", cleaned)
    return cleaned

def canonicalize_entity(raw_name: str) -> Tuple[str, str]:
    if not raw_name or not raw_name.strip():
        return "", "Unknown"

    raw_clean = strip_formatting_artifacts(raw_name)
    norm = normalize_text_token(raw_clean)

    # Never canonicalize redaction markers as entities
    if "redact" in norm or norm.startswith("["):
        return "", "Redacted"

    if norm in ALIAS_INDEX:
        canonical = ALIAS_INDEX[norm]
        category = CANONICAL_TAXONOMY[canonical]["category"]
        return canonical, category

    no_punct = re.sub(r"[^\w\s]", "", norm).strip()
    if no_punct in ALIAS_INDEX:
        canonical = ALIAS_INDEX[no_punct]
        category = CANONICAL_TAXONOMY[canonical]["category"]
        return canonical, category

    # Preserve CamelCase naming (e.g. CampusBridge, TalentBridge)
    if any(c.isupper() for c in raw_clean[1:]) and not raw_clean.isupper():
        return raw_clean, "General"

    return raw_clean.title(), "General"

def is_same_entity(entity_a: str, entity_b: str) -> bool:
    canon_a, _ = canonicalize_entity(entity_a)
    canon_b, _ = canonicalize_entity(entity_b)
    return canon_a.lower() == canon_b.lower()

def is_patent_or_metadata(text: str) -> bool:
    """Detects patent numbers, publication status, date ranges, and performance metrics."""
    if not text:
        return False
    norm = text.lower().strip()
    # 6+ consecutive digits (e.g. patent application or registration IDs like 202641102290)
    if re.search(r"\b\d{6,}\b", norm):
        return True
    # Patent / publication / IPR markers
    if re.search(r"(?i)\b(?:patent|published|filing|filed|provisional|application\s+no|ipr|in\s+progress)\b", norm):
        return True
    # Performance metrics / algorithmic notation / benchmark tokens
    if re.search(r"(?i)\b(?:o\(log\s*n\)|o\(n\)|rps|ms|lsb|cgpa|gpa|aggregate|\d+%\s+detection|\d+\s+hours?|\d+(?:st|nd|rd|th)\s+among)\b", norm):
        return True
    return False

def is_tech_stack_or_tool_list(text: str) -> bool:
    """Detects comma-separated or delimited technology/framework lists."""
    if not text:
        return False
    norm = text.lower().strip()
    if any(re.search(pref, norm) for pref in [
        r"(?i)^tech(?:nology)?\s*stack\s*:",
        r"(?i)^technologies\s*(?:used)?\s*:",
        r"(?i)^tools\s*(?:&|and)?\s*technologies\s*:",
        r"(?i)^environment\s*:",
        r"(?i)^core\s*stack\s*:"
    ]):
        return True

    # Check if string is a comma-separated list of items
    parts = [p.strip() for p in re.split(r"[,;]+", norm) if p.strip()]
    if len(parts) >= 2:
        tech_matches = 0
        for p in parts:
            p_clean = re.sub(r"[^\w\s+#.-]", "", p).strip()
            if p_clean in ALIAS_INDEX or p_clean in [
                "python", "fastapi", "react", "react.js", "docker", "gcp", "aws",
                "c++", "c", "java", "gemini", "gemini api", "sarvam ai", "qgis",
                "shapely", "geojson", "r-tree indexing", "asyncio", "numpy", "pandas",
                "esp32-cam", "mpu-6050", "embedded c", "iot", "iot edge computing",
                "pydantic", "pydantic v2", "spring boot", "postgresql", "redis"
            ]:
                tech_matches += 1
        # If at least 2 tokens are recognized technologies or > 40% are technologies
        if tech_matches >= 2 or (tech_matches / len(parts) >= 0.40):
            return True

    return False

def clean_project_title(raw_title: str) -> Optional[str]:
    """
    Validates and extracts genuine project names.
    Strictly filters out section headings, tech stack enumerations, patent metadata,
    and normalizes leading action verbs (e.g. 'Developed the Platform' -> 'Platform').
    """
    if not raw_title:
        return None

    cleaned = strip_formatting_artifacts(raw_title).strip()
    if not cleaned:
        return None

    # Defense 1: Reject patent numbers and publication metadata (e.g. '202641102290 (Published)')
    if is_patent_or_metadata(cleaned):
        return None

    # Defense 2: Reject tech stack lists (e.g. 'Python, FastAPI, Pydantic v2, Gemini, React.js')
    if is_tech_stack_or_tool_list(cleaned):
        return None

    # Defense 3: Normalize action verbs from project headings
    # (e.g. 'Developed the Real-Time Voicebot' -> 'Real-Time Voicebot')
    action_prefix_match = re.match(
        r"(?i)^(?:developed|architected|engineered|built|designed|implemented|created|co-developed)\s+(?:the\s+|a\s+|an\s+)?",
        cleaned
    )
    if action_prefix_match:
        cleaned = cleaned[action_prefix_match.end():].strip()

    if not cleaned or not cleaned[0].isupper():
        return None
    norm = cleaned.lower()

    if cleaned.startswith((")", "]", "}", ">", "-", "*", "•", "●")):
        return None

    # Reject lines starting with articles/prepositions that indicate wrapped sentence fragments
    if re.match(r"(?i)^(?:an|a|the|with|and|or|for|of|in|to|from|by)\s+", cleaned):
        return None

    # Reject fragments ending with punctuation (sentences, clauses)
    if cleaned.endswith((".", ",", ";", ":", "...")):
        return None

    # Reject lines starting with gerunds or verbs (sentence continuations)
    if re.match(r"(?i)^(?:requiring|enabling|built|developed|using|allowing|integrating|providing|securing|engineered|scaling|optimizing|maintaining|monitoring)\b", norm):
        return None

    # Defense 4: Reject section headings (e.g. 'Leadership, Achievements & Certifications')
    if norm in PROJECT_TITLE_BLACKLIST:
        return None

    # Defense: Reject awards, hackathons, competitions, certifications as project titles
    if re.search(r"(?i)\b(?:award|hackathon|contest|competition|championship|fellowship|scholarship|certification|certificate|license)\b", norm):
        return None

    # Check for combined heading terms
    heading_hits = sum(1 for kw in SECTION_KEYWORDS if kw in norm)
    if heading_hits >= 2:
        return None
    if any(h in norm for h in [
        "leadership, achievements", "achievements & certifications", "honors & awards",
        "awards & certifications", "certifications & courses"
    ]):
        return None

    if any(norm.startswith(b) for b in ["github", "http", "www.", "live demo"]):
        return None
    if re.search(r"\b(?:github\.com|vercel\.app|netlify\.app)\b", norm):
        return None
    if "redact" in norm or norm.startswith("["):
        return None

    cleaned = re.sub(r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4}|present)\b.*", "", cleaned).strip()
    cleaned = re.sub(r"(?i)\b(?:full stack developer|developer|engineer|lead|intern|personal project)\b", "", cleaned).strip()
    cleaned = re.sub(r"[\|\–\—\-]+$", "", cleaned).strip()

    if 3 <= len(cleaned) <= 60 and len(cleaned.split()) <= 6:
        # Re-check after cleaning
        if is_patent_or_metadata(cleaned) or is_tech_stack_or_tool_list(cleaned):
            return None
        canon, _ = canonicalize_entity(cleaned)
        if canon.lower() in PROJECT_TITLE_BLACKLIST:
            return None
        return canon
    return None

def extract_non_overlapping_entities(sentence: str, known_skills: Dict[str, str]) -> List[Tuple[str, int, int]]:
    lower_sent = sentence.lower()
    candidates: List[Tuple[str, int, int, int]] = []

    sorted_aliases = sorted(known_skills.keys(), key=len, reverse=True)

    for alias in sorted_aliases:
        canonical = known_skills[alias]
        pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
        for match in re.finditer(pattern, lower_sent):
            start, end = match.start(), match.end()
            candidates.append((canonical, start, end, end - start))

    for deploy_alias, deploy_canon in DEPLOYMENT_TARGETS.items():
        pattern = rf"(?<!\w){re.escape(deploy_alias)}(?!\w)"
        for match in re.finditer(pattern, lower_sent):
            start, end = match.start(), match.end()
            candidates.append((deploy_canon, start, end, end - start))

    candidates.sort(key=lambda x: x[3], reverse=True)
    claimed_spans: List[Tuple[int, int]] = []
    resolved: List[Tuple[str, int, int]] = []

    for canon, start, end, _ in candidates:
        overlaps = any(not (end <= c_start or start >= c_end) for c_start, c_end in claimed_spans)
        if not overlaps:
            claimed_spans.append((start, end))
            resolved.append((canon, start, end))

    return sorted(resolved, key=lambda x: x[1])