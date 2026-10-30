from app.core.database import engine
from sqlalchemy import text

queries = [
    "ALTER TABLE relationships ADD COLUMN IF NOT EXISTS context_entity VARCHAR(150);",
    "ALTER TABLE relationships ADD COLUMN IF NOT EXISTS extraction_method VARCHAR(50) DEFAULT 'dependency_parsed';",
    "ALTER TABLE relationships ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW();",
    "CREATE INDEX IF NOT EXISTS idx_relationships_method ON relationships(extraction_method);",
]

with engine.connect() as conn:
    for query in queries:
        conn.execute(text(query))

    conn.commit()

print("Phase 4 schema synced successfully.")

engine.dispose()