"""
Initialize PostgreSQL Database for Aircraft PM System
"""

from run import app, db, FlightSurvey
from datetime import datetime, timezone

with app.app_context():
    # Create all tables
    db.create_all()
    print("✅ Database tables created successfully!")
    
    # Verify tables
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()
    print(f"📊 Tables in database: {tables}")
    
    print("\n" + "="*50)
    print("DATABASE INITIALIZATION COMPLETE")
    print("="*50)
    print("PostgreSQL Database: aircraft_pm")
    print("Connection String: postgresql://postgres:Panduru@7013@localhost:5432/aircraft_pm")
    print("="*50)