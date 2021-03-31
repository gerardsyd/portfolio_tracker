"""trades table

Revision ID: 9d7979eb3dda
Revises: 99cacc149d23
Create Date: 2020-09-23 15:34:25.746974

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9d7979eb3dda'
down_revision = '99cacc149d23'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('trades',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('date', sa.DateTime(), nullable=True),
    sa.Column('ticker', sa.String(length=10), nullable=True),
    sa.Column('quantity', sa.Float(), nullable=True),
    sa.Column('price', sa.Float(), nullable=True),
    sa.Column('fees', sa.Float(), nullable=True),
    sa.Column('direction', sa.String(length=10), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_trades_date'), 'trades', ['date'], unique=False)
    op.create_index(op.f('ix_trades_direction'), 'trades', ['direction'], unique=False)
    op.create_index(op.f('ix_trades_fees'), 'trades', ['fees'], unique=False)
    op.create_index(op.f('ix_trades_price'), 'trades', ['price'], unique=False)
    op.create_index(op.f('ix_trades_quantity'), 'trades', ['quantity'], unique=False)
    op.create_index(op.f('ix_trades_ticker'), 'trades', ['ticker'], unique=False)
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_trades_ticker'), table_name='trades')
    op.drop_index(op.f('ix_trades_quantity'), table_name='trades')
    op.drop_index(op.f('ix_trades_price'), table_name='trades')
    op.drop_index(op.f('ix_trades_fees'), table_name='trades')
    op.drop_index(op.f('ix_trades_direction'), table_name='trades')
    op.drop_index(op.f('ix_trades_date'), table_name='trades')
    op.drop_table('trades')
    # ### end Alembic commands ###