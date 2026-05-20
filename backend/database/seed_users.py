"""
backend/database/seed_users.py
Simplified seed script: Only ensures a single master admin exists.
Demo users are now handled dynamically via /demo-login.
"""

import asyncio
from sqlalchemy import select, delete
from database.db import engine, AsyncSessionLocal
from database.models import User, Organization, Base
from security.auth import hash_password

async def seed():
    print("🌱 Initializing database...")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        # 1. Clean up old demo users (Isolated Demo Strategy)
        await session.execute(delete(User).where(User.email.like('%demo%')))
        await session.execute(delete(User).where(User.is_demo == True))
        
        # 2. Ensure Master Admin
        admin_email = "admin@gmail.com"
        admin_pass  = "admin2481"
        
        # Ensure Org
        res_org = await session.execute(select(Organization).where(Organization.name == "CodePerfect Hospital"))
        org = res_org.scalar_one_or_none()
        if not org:
            org = Organization(name="CodePerfect Hospital")
            session.add(org)
            await session.commit()
            await session.refresh(org)

        # Ensure Admin
        res_admin = await session.execute(select(User).where(User.email == admin_email))
        admin = res_admin.scalar_one_or_none()

        if not admin:
            admin = User(
                name="System Administrator",
                email=admin_email,
                password_hash=hash_password(admin_pass),
                role="ADMIN",
                org_id=org.id,
                is_active=True,
                is_demo=False
            )
            session.add(admin)
            print(f"✅ Master Admin created: {admin_email}")
        else:
            admin.role = "ADMIN"
            admin.is_active = True
            admin.is_demo = False
            admin.password_hash = hash_password(admin_pass)
            print(f"ℹ️ Master Admin verified")

        await session.commit()

    print("🎉 Database initialization complete!")

if __name__ == "__main__":
    asyncio.run(seed())