"""Apply the vsfmed_v2 schema to the Neon database referenced by VSF_MED_DATABASE_URL.

Idempotent — every CREATE is IF NOT EXISTS / OR REPLACE.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import sqlalchemy as sa

ROOT = Path(__file__).resolve().parents[1]


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k.strip(), v)


def main() -> int:
    _load_env()
    url = os.environ.get("VSF_MED_DATABASE_URL")
    if not url:
        print("ERROR: VSF_MED_DATABASE_URL not set", file=sys.stderr)
        return 2

    schema_sql = (ROOT / "src/database/vsfmed_v2_schema.sql").read_text()
    engine = sa.create_engine(url)
    # Execute statements one by one for clearer error reporting
    statements = [s.strip() for s in schema_sql.split(";") if s.strip()]
    print(f"Applying {len(statements)} DDL statements...")
    with engine.begin() as conn:
        for i, stmt in enumerate(statements, 1):
            try:
                conn.execute(sa.text(stmt))
            except Exception as exc:
                print(f"  [{i}] FAILED: {stmt[:80]}...")
                raise
    print("Schema applied.")

    # Sanity: list tables in vsfmed_v2 schema
    with engine.connect() as conn:
        rows = conn.execute(sa.text(
            "SELECT table_name, "
            "(SELECT COUNT(*) FROM information_schema.columns "
            " WHERE table_schema='vsfmed_v2' AND table_name=t.table_name) AS n_cols "
            "FROM information_schema.tables t "
            "WHERE table_schema='vsfmed_v2' "
            "ORDER BY table_name"
        )).all()
        print(f"\nvsfmed_v2 objects ({len(rows)}):")
        for name, n_cols in rows:
            print(f"  {name}  ({n_cols} cols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
