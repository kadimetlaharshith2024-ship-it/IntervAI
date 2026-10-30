from app.core.database import engine
from sqlalchemy import text

statements = [
    "ALTER TABLE relationships ADD COLUMN IF NOT EXISTS section VARCHAR(100) DEFAULT 'General';",
    "ALTER TABLE relationships ADD COLUMN IF NOT EXISTS evidence_text TEXT DEFAULT '';",
    "ALTER TABLE relationships ALTER COLUMN source_entity TYPE TEXT;",
    "ALTER TABLE relationships ALTER COLUMN target_entity TYPE TEXT;",
    "ALTER TABLE relationships ALTER COLUMN relation_type TYPE VARCHAR(100);",
    "CREATE INDEX IF NOT EXISTS idx_relationships_section ON relationships(section);",
]

with engine.connect() as conn:
    for stmt in statements:
        conn.execute(text(stmt))
    conn.commit()

print("Columns committed successfully.")