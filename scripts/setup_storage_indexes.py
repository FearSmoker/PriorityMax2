#!/usr/bin/env python3
"""
PriorityMax — Storage Bootstrap Utility
---------------------------------------

This script ensures MongoDB indexes and TTL settings are created correctly for
PriorityMax collections.

Usage:
  python3 backend/scripts/setup_storage_indexes.py [--tenant ORG] [--dry-run]

Example:
  python3 backend/scripts/setup_storage_indexes.py --tenant acme --dry-run

Environment:
  PRIORITYMAX_MONGO_URI  (default: mongodb://localhost:27017)
  PRIORITYMAX_MONGO_DB   (default: prioritymax)
"""

import os
import sys
import asyncio
import logging
import argparse
import datetime
from typing import Optional, List, Tuple

# Add app path for import
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

try:
    from app.storage import MongoStorage
except Exception as e:
    print(f"❌ Failed to import MongoStorage: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
LOG = logging.getLogger("prioritymax.setup_storage_indexes")
LOG.setLevel(os.getenv("PRIORITYMAX_LOG_LEVEL", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not LOG.handlers:
    LOG.addHandler(_handler)

# ---------------------------------------------------------------------
# Index definitions
# ---------------------------------------------------------------------
def get_index_specs() -> List[Tuple[str, List[Tuple[str, int]], dict]]:
    """
    Returns a list of tuples:
      (collection_name, key_list, options)
    """
    day = 86400
    return [
        ("scaling_actions", [("ts", -1)], {"expireAfterSeconds": 30 * day}),
        ("scaling_summary", [("ts", -1)], {"expireAfterSeconds": 7 * day}),
        ("dlq_promotions", [("ts", -1)], {"expireAfterSeconds": 30 * day}),
        ("drift_events", [("ts", -1)], {"expireAfterSeconds": 90 * day}),
        ("model_registry", [("version_id", 1)], {"unique": True}),
        ("model_registry_latest", [("model_type", 1)], {"unique": True}),
        ("correlation_stats", [("ts", -1)], {"expireAfterSeconds": 30 * day}),
        ("prioritymax_audit", [("ts", -1)], {"expireAfterSeconds": 180 * day}),
    ]

# ---------------------------------------------------------------------
# Async main
# ---------------------------------------------------------------------
async def setup_indexes(tenant: Optional[str] = None, dry_run: bool = False):
    LOG.info("🔧 Starting Mongo index setup (tenant=%s dry_run=%s)", tenant, dry_run)
    store = MongoStorage()
    await store.connect()
    if not store.db:
        LOG.error("❌ Mongo connection failed; aborting index setup.")
        return

    created = 0
    specs = get_index_specs()
    for collection, keys, opts in specs:
        try:
            opt_str = ", ".join(f"{k}={v}" for k, v in opts.items())
            LOG.info("Preparing index for %s on %s (%s)", collection, keys, opt_str)
            if not dry_run:
                await store.ensure_index(collection, keys, tenant=tenant, **opts)
                created += 1
        except Exception:
            LOG.exception("Failed to ensure index for %s", collection)

    if not dry_run:
        LOG.info("✅ Successfully created/ensured %d indexes.", created)
    else:
        LOG.info("🧪 Dry-run completed. %d indexes would be created.", len(specs))

    # sanity check ping
    ok = await store.ping()
    LOG.info("Mongo ping status: %s", ok)
    await store.close()

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PriorityMax MongoDB Index Setup Utility")
    parser.add_argument("--tenant", type=str, default=None, help="Optional tenant/org namespace prefix")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned operations")
    args = parser.parse_args()
    asyncio.run(setup_indexes(tenant=args.tenant, dry_run=args.dry_run))

if __name__ == "__main__":
    main()
