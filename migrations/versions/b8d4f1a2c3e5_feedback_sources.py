"""feedback sources: источники RAG-ответа, которые пользователь видел, когда ставил оценку (block 5.5)

Revision ID: b8d4f1a2c3e5
Revises: 7c1d2e4f5a6b
Create Date: 2026-10-11 21:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b8d4f1a2c3e5'
down_revision: Union[str, Sequence[str], None] = '7c1d2e4f5a6b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # NULL — оценка без источников (ответ без базы знаний или клиент их не прислал).
    op.add_column('message_feedback', sa.Column('sources', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('message_feedback', 'sources')
