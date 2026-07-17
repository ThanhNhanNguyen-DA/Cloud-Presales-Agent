"""
Migration helper: Reads SQL files and provides instructions for Supabase setup.

NOTE: The Supabase free tier has limitations with direct Python DDL execution.
The most reliable approach is to manually copy-paste SQL into the Supabase Dashboard.
This script prints those instructions with the SQL content for easy copy-paste.
"""

import os
from pathlib import Path


def read_sql_file(filepath):
    """Read SQL file content."""
    with open(filepath, 'r') as f:
        return f.read()


def print_migration_instructions():
    """Print clear instructions for running migration in Supabase Dashboard."""
    
    base_dir = Path(__file__).parent.parent.parent
    schema_path = base_dir / 'sql' / 'schema.sql'
    seed_path = base_dir / 'sql' / 'seed_data.sql'
    
    print("\n" + "="*80)
    print("PHASE 1 DATABASE MIGRATION — SUPABASE SETUP")
    print("="*80)
    
    print("\n📋 INSTRUCTIONS:")
    print("\n1. Go to your Supabase Dashboard: https://app.supabase.com")
    print("2. Select your project")
    print("3. Navigate to: SQL Editor (left sidebar)")
    print("4. Click 'New query' button")
    print("\n5. Copy the entire content from sql/schema.sql and paste it into the editor")
    print("6. Click 'Run' (or Ctrl+Enter)")
    print("7. Wait for success ✓")
    print("\n8. Create ANOTHER new query")
    print("9. Copy the entire content from sql/seed_data.sql and paste it into the editor")
    print("10. Click 'Run'")
    print("11. Wait for success ✓")
    
    print("\n" + "-"*80)
    print("SQL FILES CONTENT FOR EASY REFERENCE:")
    print("-"*80)
    
    print("\n📄 FILE: sql/schema.sql")
    print("-"*80)
    if schema_path.exists():
        schema_content = read_sql_file(schema_path)
        print(schema_content)
    else:
        print(f"⚠️  {schema_path} not found")
    
    print("\n" + "-"*80)
    print("📄 FILE: sql/seed_data.sql")
    print("-"*80)
    if seed_path.exists():
        seed_content = read_sql_file(seed_path)
        print(seed_content)
    else:
        print(f"⚠️  {seed_path} not found")
    
    print("\n" + "="*80)
    print("✅ After running both queries:")
    print("   - 4 tables created (cloud_documents, cloud_competitor_mapping,")
    print("     cloud_service_pricing, cloud_feedback_memory)")
    print("   - 2 RPC functions created (match_past_experience, match_cloud_documents)")
    print("   - Seed data inserted (5 competitor mappings, 5 pricing rows)")
    print("="*80 + "\n")


if __name__ == '__main__':
    print_migration_instructions()
