"""add relationship evidence and section

Revision ID: e41eef7d6d72
Revises: 5bc5344129eb
Create Date: 2026-10-08 15:56:57.403173

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e41eef7d6d72'
down_revision: Union[str, Sequence[str], None] = '5bc5344129eb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
