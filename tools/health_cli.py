#!/usr/bin/env python3
"""
PriorityMax Health CLI Helper
---------------------------------

Lightweight CLI for interacting with PriorityMax backend health endpoints.
Designed for use in:
 - CI/CD pipelines
 - Kubernetes readiness/liveness tests
 - Local dev diagnostics

Example usage:
  python3 backend/tools/health_cli.py --url http://localhost:8000/health/readiness
  python3 backend/tools/health_cli.py readiness --deep
  python3 backend/tools/health_cli.py diagnostics
  python3 backend/tools/health_cli.py wait --timeout 60

Exit codes:
  0 = healthy
  1 = degraded
  2 = fail / not ready / unreachable
"""

import argparse
import asyncio
import json
import sys
import time
import aiohttp

DEFAULT_BASE_URL = "http://localhost:8000/health"

async def fetch_json(url: str) -> dict:
    """
    Perform an HTTP GET and return JSON. Raises on failure.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"{url} -> HTTP {resp.status}")
            try:
                return await resp.json(content_type=None)
            except Exception:
                txt = await resp.text()
                raise RuntimeError(f"Invalid JSON response: {txt[:120]}")

async def call_health(endpoint: str, base_url: str, deep: bool = False):
    """
    Call the health endpoint and print summarized output.
    """
    url = f"{base_url}/{endpoint}"
    if endpoint == "readiness" and deep:
        url += "?deep=true"
    try:
        data = await fetch_json(url)
    except Exception as e:
        print(f"[ERROR] Failed to reach {url}: {e}", file=sys.stderr)
        sys.exit(2)

    # Simplify output
    if "overall" in data:
        overall = data.get("overall")
        print(f"Overall status: {overall}")
        for comp in data.get("components", []):
            print(f"  - {comp.get('name')}: {comp.get('status')}")
        if overall == "ok":
            sys.exit(0)
        elif overall == "degraded":
            sys.exit(1)
        else:
            sys.exit(2)
    else:
        # assume simple liveness/dict
        print(json.dumps(data, indent=2))
        sys.exit(0 if "alive" in str(data).lower() or data.get("status") == "alive" else 1)

async def wait_for_ready(base_url: str, timeout: int = 60, interval: int = 5):
    """
    Poll /health/readiness until overall == 'ok' or timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            data = await fetch_json(f"{base_url}/readiness")
            overall = data.get("overall")
            if overall == "ok":
                print("[OK] Service ready.")
                sys.exit(0)
            print(f"[WAIT] Overall: {overall}, retrying in {interval}s...")
        except Exception as e:
            print(f"[WAIT] {e}")
        await asyncio.sleep(interval)
    print("[ERROR] Timeout waiting for readiness.")
    sys.exit(2)

async def main():
    parser = argparse.ArgumentParser(description="PriorityMax Health CLI")
    sub = parser.add_subparsers(dest="cmd", required=False)

    # Simple endpoints
    parser.add_argument("--url", help="Direct health endpoint URL", default=None)
    parser.add_argument("--base", help="Base health URL (default http://localhost:8000/health)", default=DEFAULT_BASE_URL)

    # Specific modes
    parser_readiness = sub.add_parser("readiness", help="Run readiness probe")
    parser_readiness.add_argument("--deep", action="store_true", help="Run deep readiness check")

    sub.add_parser("liveness", help="Run liveness probe")
    sub.add_parser("diagnostics", help="Run deep diagnostics probe")

    parser_wait = sub.add_parser("wait", help="Wait for readiness (loop)")
    parser_wait.add_argument("--timeout", type=int, default=60)
    parser_wait.add_argument("--interval", type=int, default=5)

    args = parser.parse_args()

    if args.url:
        # direct URL mode
        print(f"Calling direct URL: {args.url}")
        try:
            data = await fetch_json(args.url)
            print(json.dumps(data, indent=2))
            sys.exit(0)
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            sys.exit(2)

    base = args.base.rstrip("/")

    if args.cmd == "readiness":
        await call_health("readiness", base, deep=args.deep)
    elif args.cmd == "liveness":
        await call_health("liveness", base)
    elif args.cmd == "diagnostics":
        await call_health("diagnostics", base, deep=True)
    elif args.cmd == "wait":
        await wait_for_ready(base, timeout=args.timeout, interval=args.interval)
    else:
        # default behavior
        await call_health("readiness", base)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
