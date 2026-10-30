"""add answer evaluation and session tracking columns

Revision ID: c8f319a4b123
Revises: e41eef7d6d72
Create Date: 2026-10-10 21:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = 'c8f319a4b123'
down_revision: Union[str, Sequence[str], None] = 'e41eef7d6d72'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Update interview_answers
    op.add_column('interview_answers', sa.Column('session_id', UUID(as_uuid=True), sa.ForeignKey('interview_sessions.id', ondelete='CASCADE'), nullable=True))
    op.create_index('idx_answers_session', 'interview_answers', ['session_id'])
    op.add_column('interview_answers', sa.Column('status', sa.String(30), nullable=False, server_default='EVALUATED'))
    op.add_column('interview_answers', sa.Column('overall_score', sa.Float(), nullable=True))
    op.add_column('interview_answers', sa.Column('conceptual_understanding_score', sa.Float(), nullable=True))
    op.add_column('interview_answers', sa.Column('completeness_score', sa.Float(), nullable=True))
    op.add_column('interview_answers', sa.Column('depth_score', sa.Float(), nullable=True))
    op.add_column('interview_answers', sa.Column('reasoning_score', sa.Float(), nullable=True))
    op.add_column('interview_answers', sa.Column('evidence_alignment_score', sa.Float(), nullable=True))
    op.add_column('interview_answers', sa.Column('evaluation_confidence', sa.Float(), nullable=True, server_default='1.0'))
    op.add_column('interview_answers', sa.Column('strengths', sa.JSON(), nullable=True))
    op.add_column('interview_answers', sa.Column('weaknesses', sa.JSON(), nullable=True))
    op.add_column('interview_answers', sa.Column('missing_concepts', sa.JSON(), nullable=True))
    op.add_column('interview_answers', sa.Column('key_takeaways', sa.Text(), nullable=True))
    op.add_column('interview_answers', sa.Column('dimension_scores', sa.JSON(), nullable=True))
    op.add_column('interview_answers', sa.Column('evaluator_provider', sa.String(50), nullable=True))
    op.add_column('interview_answers', sa.Column('evaluator_model', sa.String(100), nullable=True))
    op.add_column('interview_answers', sa.Column('rubric_version', sa.String(20), nullable=False, server_default='v1.0'))
    op.add_column('interview_answers', sa.Column('evaluation_latency_ms', sa.Float(), nullable=True))
    op.add_column('interview_answers', sa.Column('error_message', sa.Text(), nullable=True))

    # 2. Update interview_sessions
    op.add_column('interview_sessions', sa.Column('completed_questions', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('interview_sessions', sa.Column('total_questions', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('interview_sessions', sa.Column('category_scores', sa.JSON(), nullable=True))
    op.add_column('interview_sessions', sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True))

    # 3. Update interview_questions (optional expected_concepts)
    op.add_column('interview_questions', sa.Column('expected_concepts', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('interview_questions', 'expected_concepts')
    op.drop_column('interview_sessions', 'completed_at')
    op.drop_column('interview_sessions', 'category_scores')
    op.drop_column('interview_sessions', 'total_questions')
    op.drop_column('interview_sessions', 'completed_questions')
    op.drop_index('idx_answers_session', table_name='interview_answers')
    op.drop_column('interview_answers', 'error_message')
    op.drop_column('interview_answers', 'evaluation_latency_ms')
    op.drop_column('interview_answers', 'rubric_version')
    op.drop_column('interview_answers', 'evaluator_model')
    op.drop_column('interview_answers', 'evaluator_provider')
    op.drop_column('interview_answers', 'dimension_scores')
    op.drop_column('interview_answers', 'key_takeaways')
    op.drop_column('interview_answers', 'missing_concepts')
    op.drop_column('interview_answers', 'weaknesses')
    op.drop_column('interview_answers', 'strengths')
    op.drop_column('interview_answers', 'evaluation_confidence')
    op.drop_column('interview_answers', 'evidence_alignment_score')
    op.drop_column('interview_answers', 'reasoning_score')
    op.drop_column('interview_answers', 'depth_score')
    op.drop_column('interview_answers', 'completeness_score')
    op.drop_column('interview_answers', 'conceptual_understanding_score')
    op.drop_column('interview_answers', 'overall_score')
    op.drop_column('interview_answers', 'status')
    op.drop_column('interview_answers', 'session_id')
