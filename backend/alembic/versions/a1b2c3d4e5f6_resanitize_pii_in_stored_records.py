"""resanitize_pii_in_stored_records

Revision ID: a1b2c3d4e5f6
Revises: e1f2a3b4c5d6
Create Date: 2026-10-29 20:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.orm import Session

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'e1f2a3b4c5d6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Executes data resanitization across all existing records."""
    import sys
    import os
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    bind = op.get_bind()
    session = Session(bind=bind)
    try:
        from scripts.resanitize_database import resanitize_existing_records
        resanitize_existing_records(session)
    finally:
        session.close()


def downgrade() -> None:
    """Irreversible data redaction operation."""
    pass
