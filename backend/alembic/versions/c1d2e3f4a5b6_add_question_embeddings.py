"""add_question_embeddings

Revision ID: c1d2e3f4a5b6
Revises: b8c9d0e1f2a3
Create Date: 2026-10-15 17:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c1d2e3f4a5b6'
down_revision: Union[str, Sequence[str], None] = 'b8c9d0e1f2a3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add nullable embedding and embedding_model columns to interview_questions.
    Preserves existing rows with NULL.
    """
    op.add_column('interview_questions', sa.Column('embedding', sa.JSON(), nullable=True))
    op.add_column('interview_questions', sa.Column('embedding_model', sa.String(length=100), nullable=True))


def downgrade() -> None:
    """
    Remove embedding and embedding_model columns from interview_questions.
    """
    op.drop_column('interview_questions', 'embedding_model')
    op.drop_column('interview_questions', 'embedding')
