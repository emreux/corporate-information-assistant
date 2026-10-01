"""
Step 14: check users, roles and where confidential content may go.

Part 1 needs nothing else: passwords and the user file (in a temporary folder), signing in, roles.
Part 2 searches the real collection with different roles (needs Qdrant and the model server).
Part 3 asks confidential questions (needs Ollama on the A5000; three LLM calls). They must be answered
by the local model, a follow-up question of a manager must also be rewritten by it (step 17), and
with the local model unreachable there must be no answer at all.

Usage (from the project root):
    python -m scripts.check_access
"""
import logging
import sys
import tempfile
from pathlib import Path

import config
import rag.llm
from rag.answer import ConfidentialModelUnavailable, answer_question
from rag.auth import AuthError, User, authenticate, hash_password, load_users, require_admin, save_users, \
    verify_password
from rag.retriever import search
from scripts.check_env import run_check

SALARY_QUESTION = "Müdürlerin maaş aralığı ne kadar?"
FOLLOW_UP = "Peki yıllık prim hedefi ne kadar?"
SALARY_FILE = "Yonetici_Maas_Skalasi_2026.xlsx"
BROKEN_USER_FILES = [
    ("an unknown role", 'role = "superuser"'),
    ("a typo in a field name", 'rol = "employee"'),
    ("an invalid user name", 'role = "employee"', "Emre Seyhan"),
]


# --- Part 1 -------------------------------------------------------------------------

def check_passwords() -> str:
    first, second = hash_password("dogru-parola-1"), hash_password("dogru-parola-1")
    if first == second:
        raise AssertionError("two hashes of the same password are equal: the salt is not random")
    if not verify_password("dogru-parola-1", first) or verify_password("yanlis-parola", first):
        raise AssertionError("verify_password accepts a wrong password or rejects the right one")
    if verify_password("x", "garbage") or verify_password("x", "scrypt$1$2"):
        raise AssertionError("a damaged hash was accepted")
    return "salted, right password accepted, wrong and damaged ones rejected"


def expect_rejected(folder: Path, description: str, role_line: str, username: str = "emre") -> str:
    path = folder / "users.toml"
    path.write_text(f'[[user]]\nusername = "{username}"\nname = "x"\n{role_line}\npassword = "x"\n',
                    encoding="utf-8")
    try:
        load_users(path)
    except AuthError as exc:
        return f"rejected: {exc}"
    raise AssertionError(f"a user file with {description} was accepted")


def check_sign_in(folder: Path) -> str:
    path = folder / "users.toml"
    save_users({"can": (User("can", "Can", "intern"), hash_password("stajyer-parola"))}, path)
    if authenticate(" Can ", "stajyer-parola", path) != User("can", "Can", "intern"):
        raise AssertionError("the right name and password were rejected")
    if authenticate("can", "yanlis-parola", path) or authenticate("hacker", "stajyer-parola", path):
        raise AssertionError("a wrong password or an unknown user was accepted")
    return "right password in, wrong password and unknown user out"


def check_admin_only() -> str:
    require_admin(User("emre", "Emre", "admin"))
    for user in (User("ayse", "Ayşe", "manager"), User("can", "Can", "employee"), None):
        try:
            require_admin(user)
        except PermissionError:
            continue
        raise AssertionError(f"{user} may manage documents")
    return "only the admin role may upload and sync"


def check_role_table() -> str:
    for name, role in config.ROLES.items():
        unknown = set(role["access_levels"]) - set(config.ACCESS_LEVELS)
        if unknown:
            raise AssertionError(f"role {name!r} has unknown access levels {unknown}")
    if not any(role["admin"] for role in config.ROLES.values()):
        raise AssertionError("no role may manage documents")
    return f"{len(config.ROLES)} roles: " + ", ".join(
        f"{name}={'+'.join(role['access_levels'])}" for name, role in config.ROLES.items())


# --- Part 2 -------------------------------------------------------------------------

def check_search(role: str, salary_expected: bool, mode: str | None = None) -> str:
    hits = search(SALARY_QUESTION, access_levels=config.ROLES[role]["access_levels"], mode=mode)
    allowed = config.ROLES[role]["access_levels"]
    outside = [hit.source for hit in hits if hit.access_level not in allowed]
    if outside:
        raise AssertionError(f"{role} got chunks outside its rights: {outside}")
    found = any(hit.source == SALARY_FILE for hit in hits)
    if found != salary_expected:
        raise AssertionError(f"salary scale {'not ' if salary_expected else ''}found for {role}")
    return f"{len(hits)} chunks, salary scale {'found' if found else 'not found'}"


# --- Part 3 -------------------------------------------------------------------------

def check_local_answer() -> str:
    answer = answer_question(SALARY_QUESTION, access_levels=config.ROLES["manager"]["access_levels"])
    if not answer.confidential:
        raise AssertionError("the salary scale was not among the sources, so nothing was tested")
    provider = rag.llm.get_providers()[answer.llm.provider]
    if not provider.is_local:
        raise AssertionError(f"confidential content was answered by {answer.llm.provider}")
    return f"answered by {answer.llm.summary()}"


def check_local_rewrite() -> str:
    """Before the search nobody knows whether a manager's conversation is confidential: rewrite locally."""
    answer = answer_question(FOLLOW_UP, access_levels=config.ROLES["manager"]["access_levels"],
                             history=[SALARY_QUESTION])
    if answer.rewrite is None:
        raise AssertionError("the follow-up question was not rewritten (is the local model running?)")
    if not rag.llm.get_providers()[answer.rewrite.provider].is_local:
        raise AssertionError(f"the conversation was sent to {answer.rewrite.provider} for rewriting")
    return f"rewritten by {answer.rewrite.summary()}: {answer.search_question!r}"


def check_fail_closed() -> str:
    """Point the local model at a closed port: the answer must fail, never go to the cloud.
    (With a conversation, so that the rewrite must not go to the cloud either.)"""
    real_url = config.OLLAMA_BASE_URL
    config.OLLAMA_BASE_URL = "http://127.0.0.1:9/v1"
    rag.llm.get_providers.cache_clear()
    try:
        answer = answer_question(SALARY_QUESTION, access_levels=config.ROLES["manager"]["access_levels"],
                                 history=["Yıllık izin kaç gün?"])
    except ConfidentialModelUnavailable:
        return "no answer, and no request to a cloud model"
    finally:
        config.OLLAMA_BASE_URL = real_url
        rag.llm.get_providers.cache_clear()
    raise AssertionError(f"answered by {answer.llm.summary() if answer.llm else 'nobody'} instead of failing")


def main() -> None:
    logging.basicConfig(level=logging.ERROR, format="       (log) %(message)s")
    results: list[bool] = []

    print("Part 1: users and roles")
    results.append(run_check("Password hashing", check_passwords))
    with tempfile.TemporaryDirectory() as tmp:
        for description, role_line, *username in BROKEN_USER_FILES:
            results.append(run_check(f"User file with {description}",
                                     lambda d=description, r=role_line, u=username: expect_rejected(
                                         Path(tmp), d, r, *u)))
        results.append(run_check("Signing in", lambda: check_sign_in(Path(tmp))))
    results.append(run_check("Only admins manage documents", check_admin_only))
    results.append(run_check("Role table in config.py", check_role_table))

    print("\nPart 2: search with the rights of a role (Qdrant and the model server)")
    results.append(run_check("Intern: nothing restricted", lambda: check_search("intern", salary_expected=False)))
    for mode in ("dense", "sparse"):           # each half of the hybrid search must apply the filter (step 15)
        results.append(run_check(f"Intern, {mode} search only: nothing restricted",
                                 lambda m=mode: check_search("intern", salary_expected=False, mode=m)))
    results.append(run_check("Manager: salary scale found", lambda: check_search("manager", salary_expected=True)))

    print("\nPart 3: confidential content stays on our machines (Ollama)")
    results.append(run_check("Confidential question answered locally", check_local_answer))
    results.append(run_check("Manager's follow-up question rewritten locally", check_local_rewrite))
    results.append(run_check("Local model down: fail closed", check_fail_closed))

    print(f"\nTOTAL: {sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
