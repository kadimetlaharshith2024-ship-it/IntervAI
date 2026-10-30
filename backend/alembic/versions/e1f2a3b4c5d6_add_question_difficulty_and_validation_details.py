"""add_question_difficulty_and_validation_details

Revision ID: e1f2a3b4c5d6
Revises: c1d2e3f4a5b6
Create Date: 2026-10-17 21:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e1f2a3b4c5d6'
down_revision: Union[str, Sequence[str], None] = 'c1d2e3f4a5b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add nullable assessed_difficulty, rubric_version, and validation_details columns to interview_questions.
    Preserves all existing interview records.
    """
    op.add_column('interview_questions', sa.Column('assessed_difficulty', sa.Integer(), nullable=True))
    op.add_column('interview_questions', sa.Column('rubric_version', sa.String(length=50), nullable=True, server_default='v2.0-canonical-5level'))
    op.add_column('interview_questions', sa.Column('validation_details', sa.JSON(), nullable=True))


def downgrade() -> None:
    """
    Remove assessed_difficulty, rubric_version, and validation_details columns from interview_questions.
    """
    op.drop_column('interview_questions', 'validation_details')
    op.drop_column('interview_questions', 'rubric_version')
    op.drop_column('interview_questions', 'assessed_difficulty')
