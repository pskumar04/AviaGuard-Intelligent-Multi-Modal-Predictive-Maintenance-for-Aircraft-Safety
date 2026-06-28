"""
Reset and recreate database tables
"""

from run import app, db

with app.app_context():
    # Drop all tables
    db.drop_all()
    print("✅ Dropped all existing tables")
    
    # Create all tables
    db.create_all()
    print("✅ Created all tables")
    
    # Verify
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()
    print(f"📊 Tables in database: {tables}")