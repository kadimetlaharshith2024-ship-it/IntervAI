"""add_interview_session_state_and_difficulty

Revision ID: e7b8c9d0a1b2
Revises: f464dc8859ce
Create Date: 2026-10-12 23:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e7b8c9d0a1b2'
down_revision: Union[str, Sequence[str], None] = 'f464dc8859ce'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: Add current_difficulty and session_state columns to interview_sessions."""
    op.add_column(
        'interview_sessions',
        sa.Column('current_difficulty', sa.Integer(), nullable=False, server_default='3')
    )
    op.add_column(
        'interview_sessions',
        sa.Column('session_state', sa.JSON(), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema: Drop session_state and current_difficulty columns from interview_sessions."""
    op.drop_column('interview_sessions', 'session_state')
    op.drop_column('interview_sessions', 'current_difficulty')
