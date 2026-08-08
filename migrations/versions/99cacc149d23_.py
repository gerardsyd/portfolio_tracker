"""empty message — no-op: user table already created by e1a1ccf64985

Revision ID: 99cacc149d23
Revises: e1a1ccf64985
Create Date: 2020-09-13 22:32:53.797235

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '99cacc149d23'
down_revision = 'e1a1ccf64985'
branch_labels = None
depends_on = None


def upgrade():
    # No-op — this migration duplicated e1a1ccf64985 (user table already created)
    pass


def downgrade():
    pass
