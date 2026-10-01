"""
Step 14: manage the users of the web interface. Passwords are asked for on the screen, never given
on the command line (that would leave them in the terminal history).

Usage (from the project root):
    python -m scripts.manage_users add emre --role admin --name "Emre Seyhan"
    python -m scripts.manage_users passwd emre
    python -m scripts.manage_users role emre manager
    python -m scripts.manage_users remove emre
    python -m scripts.manage_users list
"""
import argparse
import getpass
import sys

import config
from rag.auth import USERNAME, AuthError, User, check_password_rules, hash_password, load_users, save_users


def ask_password() -> str:
    password = getpass.getpass("New password: ")
    check_password_rules(password)
    if getpass.getpass("Repeat it: ") != password:
        raise AuthError("The two passwords are different.")
    return password


def admins_left(users: dict, without: str) -> int:
    return sum(1 for name, (user, _) in users.items() if name != without and user.is_admin)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage CIA users.")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="Add a user (asks for the password).")
    add.add_argument("username")
    add.add_argument("--role", required=True, choices=list(config.ROLES))
    add.add_argument("--name", help="Name shown in the interface (default: the user name).")
    commands.add_parser("passwd", help="Set a new password.").add_argument("username")
    role = commands.add_parser("role", help="Change the role of a user.")
    role.add_argument("username")
    role.add_argument("new_role", choices=list(config.ROLES))
    commands.add_parser("remove", help="Remove a user.").add_argument("username")
    commands.add_parser("list", help="List the users (never the passwords).")
    args = parser.parse_args()

    try:
        users = load_users()
        username = getattr(args, "username", "").strip().lower()
        if args.command == "list":
            for user, _ in sorted(users.values(), key=lambda entry: entry[0].username):
                print(f"{user.username:20s} {user.role:10s} {user.name}")
            print(f"{len(users)} user(s) in {config.USERS_FILE}")
            return
        if args.command == "add":
            if not USERNAME.fullmatch(username):
                raise AuthError("User names are 3-32 characters: lowercase letters, digits, '.', '_' or '-'.")
            if username in users:
                raise AuthError(f"'{username}' already exists. Use 'passwd' or 'role' to change it.")
            users[username] = (User(username, args.name or username, args.role), hash_password(ask_password()))
        elif username not in users:
            raise AuthError(f"There is no user '{username}'.")
        elif args.command == "passwd":
            users[username] = (users[username][0], hash_password(ask_password()))
        elif args.command == "role":
            user, password_hash = users[username]
            if user.is_admin and not config.ROLES[args.new_role]["admin"] and not admins_left(users, username):
                raise AuthError("This is the last admin: add another admin first.")
            users[username] = (User(user.username, user.name, args.new_role), password_hash)
        elif args.command == "remove":
            if users[username][0].is_admin and not admins_left(users, username):
                raise AuthError("This is the last admin: add another admin first.")
            del users[username]
    except (AuthError, KeyboardInterrupt) as exc:
        print(f"STOPPED: {exc or 'cancelled'}")
        sys.exit(1)
    save_users(users)
    print(f"Done ({args.command} {username}). {len(users)} user(s) in {config.USERS_FILE}")


if __name__ == "__main__":
    main()
