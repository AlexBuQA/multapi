"""production tables: message_feedback, broadcast_queue, moderation_incidents (block 4.4)

Revision ID: 7c1d2e4f5a6b
Revises: de37b49a9e5c
Create Date: 2026-10-09 23:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '7c1d2e4f5a6b'
down_revision: Union[str, Sequence[str], None] = 'de37b49a9e5c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'message_feedback',
        sa.Column('id', sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column('message_id', sa.Uuid(), nullable=False),
        sa.Column('chat_id', sa.Uuid(), nullable=False),
        sa.Column('owner_external_id', sa.Text(), nullable=False),
        sa.Column('value', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint("value IN ('up', 'down')", name='ck_message_feedback_value'),
        sa.ForeignKeyConstraint(['chat_id'], ['chats.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['message_id'], ['chat_messages.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('owner_external_id', 'message_id', name='uq_message_feedback_owner_message'),
    )
    op.create_index('ix_message_feedback_created', 'message_feedback', ['created_at'], unique=False)
    op.create_table(
        'broadcast_queue',
        sa.Column('id', sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('interface', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), server_default='pending', nullable=False),
        sa.Column('recipients', sa.Integer(), nullable=True),
        sa.Column('sent', sa.Integer(), nullable=True),
        sa.Column('failed', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('claimed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('pending', 'sending', 'sent', 'failed')", name='ck_broadcast_queue_status'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_broadcast_queue_status_created', 'broadcast_queue', ['status', 'created_at'], unique=False)
    op.create_table(
        'moderation_incidents',
        sa.Column('id', sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column('chat_id', sa.Uuid(), nullable=True),
        sa.Column('direction', sa.Text(), nullable=False),
        sa.Column('blocked_by', sa.Text(), nullable=False),
        sa.Column('categories', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('text_hash', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['chat_id'], ['chats.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_moderation_incidents_created', 'moderation_incidents', ['created_at'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_moderation_incidents_created', table_name='moderation_incidents')
    op.drop_table('moderation_incidents')
    op.drop_index('ix_broadcast_queue_status_created', table_name='broadcast_queue')
    op.drop_table('broadcast_queue')
    op.drop_index('ix_message_feedback_created', table_name='message_feedback')
    op.drop_table('message_feedback')
