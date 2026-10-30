"""add_attempt_number_to_interview_answers

Revision ID: f464dc8859ce
Revises: d9a421b5c345
Create Date: 2026-10-11 12:52:04.615864

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f464dc8859ce'
down_revision: Union[str, Sequence[str], None] = 'd9a421b5c345'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: Add attempt_number column and performance indexes."""
    op.add_column(
        'interview_answers',
        sa.Column('attempt_number', sa.Integer(), nullable=False, server_default='1')
    )
    op.create_index(
        'idx_answers_question_attempt',
        'interview_answers',
        ['question_id', 'attempt_number']
    )
    op.create_index(
        'idx_answers_session_question_attempt',
        'interview_answers',
        ['session_id', 'question_id', 'attempt_number']
    )


def downgrade() -> None:
    """Downgrade schema: Drop performance indexes and attempt_number column."""
    op.drop_index('idx_answers_session_question_attempt', table_name='interview_answers')
    op.drop_index('idx_answers_question_attempt', table_name='interview_answers')
    op.drop_column('interview_answers', 'attempt_number')
