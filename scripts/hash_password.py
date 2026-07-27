"""One-off helper to bcrypt-hash a password for .streamlit/secrets.toml.

Usage:
    uv run python scripts/hash_password.py            # prompts for the password
    uv run python scripts/hash_password.py "s3cret"   # hash given as argument

Copy the printed hash into a [credentials.usernames.<login>] block's `password`.
"""

from __future__ import annotations

import getpass
import sys

import bcrypt


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def main() -> None:
    if len(sys.argv) > 1:
        password = sys.argv[1]
    else:
        password = getpass.getpass("Password: ")
    if not password:
        print("Empty password — aborting.", file=sys.stderr)
        sys.exit(1)
    print(hash_password(password))


if __name__ == "__main__":
    main()
