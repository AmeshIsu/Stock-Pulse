"""app/backend/cli.py
Management CLI commands:
- create-admin: Create an administrator account
- seed-demo: Seed demo user and admin (dev only, refuses in prod)
- run-dev: Start local FastAPI development server
"""
import sys
import argparse
import secrets
from sqlalchemy.orm import Session

from app.backend.core.config import settings
from app.backend.db.session import SessionLocal, engine
from app.backend.db.base import Base
from app.backend.models.entities import User
from app.backend.core.security import get_password_hash


def create_admin(email: str, password: str):
    Base.metadata.create_all(bind=engine)
    db: Session = SessionLocal()
    try:
        email = email.lower().strip()
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            existing.role = "admin"
            existing.hashed_password = get_password_hash(password)
            existing.is_active = True
            db.commit()
            print(f"Updated existing user {email} to admin role.")
        else:
            admin = User(
                email=email,
                hashed_password=get_password_hash(password),
                role="admin",
                is_active=True
            )
            db.add(admin)
            db.commit()
            print(f"Successfully created administrator: {email}")
    finally:
        db.close()


def seed_demo():
    if settings.is_prod:
        print("ERROR: Demo seeding is strictly forbidden in production mode (ENV=prod).", file=sys.stderr)
        sys.exit(1)

    Base.metadata.create_all(bind=engine)
    db: Session = SessionLocal()
    try:
        user_pwd = f"DemoUser_{secrets.token_urlsafe(8)}!"
        admin_pwd = f"DemoAdmin_{secrets.token_urlsafe(8)}!"

        # Demo User
        u = db.query(User).filter(User.email == "demo.user@example.com").first()
        if not u:
            u = User(
                email="demo.user@example.com",
                hashed_password=get_password_hash(user_pwd),
                role="user",
                is_active=True
            )
            db.add(u)
        else:
            u.hashed_password = get_password_hash(user_pwd)
            u.is_active = True

        # Demo Admin
        a = db.query(User).filter(User.email == "demo.admin@example.com").first()
        if not a:
            a = User(
                email="demo.admin@example.com",
                hashed_password=get_password_hash(admin_pwd),
                role="admin",
                is_active=True
            )
            db.add(a)
        else:
            a.hashed_password = get_password_hash(admin_pwd)
            a.role = "admin"
            a.is_active = True

        db.commit()
        print("\n=======================================================")
        print(" DEMO ACCOUNTS SEEDED (DEV ONLY)")
        print("=======================================================")
        print(f"Demo User:  demo.user@example.com  | Password: {user_pwd}")
        print(f"Demo Admin: demo.admin@example.com | Password: {admin_pwd}")
        print("=======================================================\n")
    finally:
        db.close()


def run_dev(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    print(f"Starting Stock Pulse API server at http://{host}:{port}")
    uvicorn.run("app.backend.main:app", host=host, port=port, reload=True)


def main():
    parser = argparse.ArgumentParser(description="Stock Pulse Backend CLI")
    subparsers = parser.add_subparsers(dest="command")

    # create-admin
    p_admin = subparsers.add_parser("create-admin", help="Create an administrator account")
    p_admin.add_argument("--email", default=settings.ADMIN_EMAIL, help="Admin email")
    p_admin.add_argument("--password", default=settings.ADMIN_PASSWORD, help="Admin password")

    # seed-demo
    subparsers.add_parser("seed-demo", help="Seed demo user and admin (dev only)")

    # run-dev
    p_run = subparsers.add_parser("run-dev", help="Start development server")
    p_run.add_argument("--host", default="127.0.0.1")
    p_run.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    if args.command == "create-admin":
        create_admin(args.email, args.password)
    elif args.command == "seed-demo":
        seed_demo()
    elif args.command == "run-dev":
        run_dev(args.host, args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
