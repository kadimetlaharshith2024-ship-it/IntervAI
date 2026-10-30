"""add_users_and_ownership

Revision ID: b8c9d0e1f2a3
Revises: e7b8c9d0a1b2
Create Date: 2026-10-13 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'b8c9d0e1f2a3'
down_revision: Union[str, Sequence[str], None] = 'e7b8c9d0a1b2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Upgrade schema:
    1. Create 'users' table with UUID, unique normalized email, password_hash, full_name, and created_at.
    2. Add nullable 'user_id' foreign key column to 'resumes' (preserves legacy unowned resumes as NULL).
    3. Add nullable 'user_id' foreign key column to 'interview_sessions' (preserves legacy unowned sessions as NULL).
    """
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

    # Resumes ownership
    op.add_column('resumes', sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_index(op.f('ix_resumes_user_id'), 'resumes', ['user_id'], unique=False)
    op.create_foreign_key(
        'fk_resumes_user_id_users',
        'resumes',
        'users',
        ['user_id'],
        ['id'],
        ondelete='SET NULL'
    )

    # Interview sessions ownership
    op.add_column('interview_sessions', sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_index(op.f('ix_interview_sessions_user_id'), 'interview_sessions', ['user_id'], unique=False)
    op.create_foreign_key(
        'fk_interview_sessions_user_id_users',
        'interview_sessions',
        'users',
        ['user_id'],
        ['id'],
        ondelete='SET NULL'
    )


def downgrade() -> None:
    """
    Downgrade schema:
    1. Drop foreign key constraints and columns from interview_sessions and resumes.
    2. Drop 'users' table.
    """
    op.drop_constraint('fk_interview_sessions_user_id_users', 'interview_sessions', type_='foreignkey')
    op.drop_index(op.f('ix_interview_sessions_user_id'), table_name='interview_sessions')
    op.drop_column('interview_sessions', 'user_id')

    op.drop_constraint('fk_resumes_user_id_users', 'resumes', type_='foreignkey')
    op.drop_index(op.f('ix_resumes_user_id'), table_name='resumes')
    op.drop_column('resumes', 'user_id')

    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_table('users')
