"""fix answer status default to pending

Revision ID: d9a421b5c345
Revises: c8f319a4b123
Create Date: 2026-10-10 21:35:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'd9a421b5c345'
down_revision: Union[str, Sequence[str], None] = 'c8f319a4b123'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column('interview_answers', 'status', server_default='PENDING')


def downgrade() -> None:
    op.alter_column('interview_answers', 'status', server_default='EVALUATED')
