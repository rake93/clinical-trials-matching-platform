#!/usr/bin/env python3
"""Generate a bcrypt hash for a plaintext password.

Usage:
    python scripts/hash_password.py <password>

Paste the printed hash into .env.local as AUTH_PASSWORD_HASH=<hash>.
"""
from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) != 2:  # noqa: PLR2004
        print("Usage: python scripts/hash_password.py <password>", file=sys.stderr)
        sys.exit(1)

    try:
        import bcrypt
    except ImportError:
        print("bcrypt not found. Install with:\n  pip install bcrypt", file=sys.stderr)
        sys.exit(1)

    hashed = bcrypt.hashpw(sys.argv[1].encode(), bcrypt.gensalt()).decode()
    print(hashed)


if __name__ == "__main__":
    main()
