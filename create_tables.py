"""
Create database tables for Aircraft PM System
"""

from run import app, db
from sqlalchemy import text

with app.app_context():
    # Create all tables
    db.create_all()
    print("✅ Database tables created successfully!")
    
    # Verify tables exist
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()
    print(f"📊 Tables in database: {tables}")
    
    # Check if flight_surveys table exists
    if 'flight_surveys' in tables:
        print("✅ flight_surveys table exists!")
    else:
        print("❌ flight_surveys table does NOT exist!")