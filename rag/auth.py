"""
Users and permissions: who is asking, what they may see, and whether they may manage documents.

Users live in config.USERS_FILE (TOML). Passwords are never stored, only a salted scrypt hash.
What a role may do is policy and lives in config.ROLES, so that a user file cannot grant rights.

This is a small local user store for learning. A company would sign users in with its identity
provider (Microsoft Entra ID, Okta...) over OIDC and map groups to these roles; the rest of the
code only needs a User, so it would not change.
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import tomllib
from dataclasses import dataclass
from pathlib import Path

import config

USERNAME = re.compile(r"[a-z0-9._-]{3,32}")
USER_FIELDS = {"username", "name", "role", "password"}

# scrypt cost: about 16 MB of memory and a few tens of milliseconds per check, which makes
# guessing slow even for someone who has stolen the user file.
SCRYPT_N, SCRYPT_R, SCRYPT_P, SCRYPT_BYTES = 2 ** 14, 8, 1, 32


class AuthError(ValueError):
    """The user file is broken, or a user or password does not meet the rules."""


@dataclass(frozen=True)
class User:
    username: str
    name: str
    role: str

    @property
    def access_levels(self) -> tuple[str, ...]:
        return config.ROLES[self.role]["access_levels"]

    @property
    def is_admin(self) -> bool:
        """May manage documents and use the developer tools."""
        return config.ROLES[self.role]["admin"]


# --- Passwords ----------------------------------------------------------------------

def hash_password(password: str) -> str:
    """A new random salt every time, so two users with the same password get different hashes."""
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
                            dklen=SCRYPT_BYTES)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, expected = stored.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt), n=int(n), r=int(r),
                                p=int(p), dklen=len(bytes.fromhex(expected)))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), expected)     # same time whether the first or last byte differs


def check_password_rules(password: str) -> None:
    if len(password) < config.MIN_PASSWORD_LENGTH:
        raise AuthError(f"The password must be at least {config.MIN_PASSWORD_LENGTH} characters long.")


# A valid hash of a random password: used when the user does not exist, so that a wrong user name
# takes as long as a wrong password and the response time does not reveal which user names exist.
_DUMMY_HASH = hash_password(secrets.token_hex(16))


# --- The user file ------------------------------------------------------------------

def load_users(path: Path = config.USERS_FILE) -> dict[str, tuple[User, str]]:
    """User name -> (user, password hash). A missing file means no users yet."""
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as file:
            data = tomllib.load(file)
    except tomllib.TOMLDecodeError as exc:
        raise AuthError(f"{path.name} is not valid TOML: {exc}") from exc

    users: dict[str, tuple[User, str]] = {}
    for number, item in enumerate(data.get("user", []), start=1):
        where = f"{path.name}, user #{number}"
        unknown, missing = set(item) - USER_FIELDS, USER_FIELDS - set(item)
        if unknown or missing:
            raise AuthError(f"{where}: unknown field(s) {sorted(unknown)}, missing field(s) {sorted(missing)}.")
        if not USERNAME.fullmatch(item["username"]):
            raise AuthError(f"{where}: invalid user name {item['username']!r}.")
        if item["role"] not in config.ROLES:
            raise AuthError(f"{where}: role must be one of {list(config.ROLES)}, not {item['role']!r}.")
        if item["username"] in users:
            raise AuthError(f"{where}: '{item['username']}' appears twice.")
        users[item["username"]] = (User(item["username"], item["name"], item["role"]), item["password"])
    return users


def save_users(users: dict[str, tuple[User, str]], path: Path = config.USERS_FILE) -> None:
    """Write the whole file at once through a temporary file, so a crash never leaves half a file."""
    lines = ["# CIA users. Manage with: python -m scripts.manage_users --help", ""]
    for user, password_hash in sorted(users.values(), key=lambda entry: entry[0].username):
        lines += ["[[user]]",
                  f"username = {json.dumps(user.username)}",
                  f"name = {json.dumps(user.name, ensure_ascii=False)}",     # a JSON string is a TOML string
                  f"role = {json.dumps(user.role)}",
                  f"password = {json.dumps(password_hash)}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def authenticate(username: str, password: str, path: Path = config.USERS_FILE) -> User | None:
    """The user, if the name and password match; None otherwise, without saying which one was wrong."""
    entry = load_users(path).get(username.strip().lower())
    if entry is None:
        verify_password(password, _DUMMY_HASH)
        return None
    user, password_hash = entry
    return user if verify_password(password, password_hash) else None


def require_admin(user: User | None) -> None:
    """Checked inside every function that changes documents, not only by hiding buttons."""
    if user is None or not user.is_admin:
        raise PermissionError("Only an admin may manage documents.")
