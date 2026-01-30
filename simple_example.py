#!/usr/bin/env python3
"""
Simple example for using View Similarity Finder with your exact JSON structure
This shows the three main ways to load your data:
1. From Starburst (production)
2. From a Pandas DataFrame  
3. From a Python list (best for testing)

Run any of the three methods or run the main() function which demonstrates all three.
"""

import json
import sys
import pandas as pd
from view_similarity_finder import ViewNormalizer, ViewIndex


def unescape_json(json_str):
    """Convert Starburst format with "" quotes to standard JSON with " quotes"""
    return json_str.replace('""', '"')

# =============================================================================
# SAMPLE DATA: Complex JSON with escaped quotes (from Starburst)
# =============================================================================

STARBURST_JSON_STRINGS = [
    # View 1: Mixed alias_map - some direct tables, some subqueries
    '''{
        ""view_name"": ""sales.daily_deals"",
        ""lineage"": {
            ""columns"": [
                {
                    ""column_name"": ""deal_id"",
                    ""expression"": ""t01.deal_id"",
                    ""lineage"": [""deals.deal_id""]
                },
                {
                    ""column_name"": ""sale_amount"",
                    ""expression"": ""t01.amount""
                },
                {
                    ""column_name"": ""*""
                }
            ],
            ""joins"": [
                {
                    ""type"": ""inner"",
                    ""table"": ""customers"",
                    ""condition"": ""t01.customer_id = t02.id""
                }
            ],
            ""filters"": [""t01.active = true""],
            ""alias_map"": {
                ""t01"": ""deals"",
                ""t02"": ""customers""
            }
        }
    }''',
    
    # View 2: Subquery in table field as direct string
    '''{
        ""view_name"": ""sales.deals_summary"",
        ""lineage"": {
            ""columns"": [
                {
                    ""column_name"": ""deal_id"",
                    ""expression"": ""t01.deal_id""
                },
                {
                    ""column_name"": ""customer_name""
                }
            ],
            ""joins"": [
                {
                    ""type"": ""left"",
                    ""table"": ""SELECT id, customer_id FROM deals WHERE status='active'"",
                    ""condition"": ""t02.id = t01.id""
                }
            ],
            ""filters"": [],
            ""alias_map"": {
                ""t01"": ""customers"",
                ""t02"": {
                    ""type"": ""subquery"",
                    ""sql"": ""SELECT id, customer_id FROM deals WHERE status='active'""
                }
            }
        }
    }''',
    
    # View 3: Subquery in table field as string with parentheses
    '''{
        ""view_name"": ""finance.revenue_stream"",
        ""lineage"": {
            ""columns"": [
                {
                    ""column_name"": ""deal_id"",
                    ""expression"": ""deals.deal_id""
                },
                {
                    ""column_name"": ""amount"",
                    ""expression"": ""SUM(deals.amount)""
                }
            ],
            ""joins"": [
                {
                    ""type"": ""outer"",
                    ""table"": ""(SELECT id, amount FROM deals WHERE active = 1)"",
                    ""condition"": ""t01.id = t02.id""
                }
            ],
            ""filters"": [""year(date) = 2024""],
            ""alias_map"": {
                ""deals"": ""deals"",
                ""cust"": ""customers""
            }
        }
    }''',
    
    # View 4: All alias_map are subqueries
    '''{
        ""view_name"": ""analytics.customer_profile"",
        ""lineage"": {
            ""columns"": [
                {
                    ""column_name"": ""customer_id""
                },
                {
                    ""column_name"": ""total_deals"",
                    ""expression"": ""COUNT(*)""
                }
            ],
            ""joins"": [],
            ""filters"": [],
            ""alias_map"": {
                ""agg1"": {
                    ""type"": ""subquery"",
                    ""sql"": ""SELECT id, COUNT(*) as total FROM deals GROUP BY id""
                },
                ""agg2"": {
                    ""type"": ""subquery"",
                    ""sql"": ""SELECT customer_id, name FROM customers""
                }
            }
        }
    }''',
    
    # View 5: Complex - mix of everything
    '''{
        ""view_name"": ""reporting.sales_analytics"",
        ""lineage"": {
            ""columns"": [
                {
                    ""column_name"": ""*""
                }
            ],
            ""joins"": [
                {
                    ""type"": ""inner"",
                    ""table"": ""deals"",
                    ""condition"": ""t01.id = t02.id""
                },
                {
                    ""type"": ""left"",
                    ""table"": ""SELECT customer_id, COUNT(*) as count FROM deals GROUP BY customer_id"",
                    ""condition"": ""t02.customer_id = t03.customer_id""
                }
            ],
            ""filters"": null,
            ""alias_map"": {
                ""t01"": ""customers"",
                ""t02"": ""deals"",
                ""t03"": {
                    ""type"": ""subquery"",
                    ""sql"": ""SELECT customer_id, COUNT(*) as count FROM deals GROUP BY customer_id""
                }
            }
        }
    }'''
]

# Parse the escaped JSON strings to get actual dicts
SAMPLE_JSON_VIEWS = []
for json_str in STARBURST_JSON_STRINGS:
    try:
        unescaped = unescape_json(json_str)
        view_data = json.loads(unescaped)
        SAMPLE_JSON_VIEWS.append(view_data)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)


# =============================================================================
# METHOD 1: Direct from Starburst (recommended for production)
# =============================================================================
def method_1_from_starburst():
    """Load directly from Starburst database"""
    
    print("\n" + "=" * 80)
    print("METHOD 1: Loading from Starburst")
    print("=" * 80)
    
    print("""
    This method connects to your Starburst instance and fetches view lineage data.
    
    To use this method:
    1. Update the DSN, USERNAME, PASSWORD below
    2. Update the SQL query to match your table structure
    3. Ensure your JSON has 'view_name' and 'lineage' fields
    """)
    
    # Configuration
    DSN = "your_starburst_dsn"
    USERNAME = "your_username"
    PASSWORD = "your_password"
    
    # This is a demo - comment this out and uncomment the real implementation below
    print("  [SKIPPING] - Set DSN/USERNAME/PASSWORD to actually connect to Starburst")
    
    """
    # REAL IMPLEMENTATION (uncomment when ready)
    from starburst_connector import StarburstConnector
    
    try:
        connector = StarburstConnector(DSN, USERNAME, PASSWORD)
        
        # Query to fetch views - ADJUST FOR YOUR TABLE STRUCTURE
        # Your Starburst table has: db_name, schema_name, object_name, linear_json
        query = '''
            SELECT 
                CONCAT(db_name, '.', schema_name, '.', object_name) as view_name,
                linear_json as lineage
            FROM your_catalog.your_schema.views_metadata
            WHERE object_name IS NOT NULL
            LIMIT 3500
        '''
        
        # Fetch views as DataFrame
        views_df = connector.fetch_views_with_lineage(query)
        
        # Now use METHOD 2 (from DataFrame) with this data
        method_2_from_dataframe(views_df)
        
        connector.close()
    except Exception as e:
        print(f"Error connecting to Starburst: {e}")
    """


# =============================================================================
# METHOD 2: From a Pandas DataFrame (if you already fetched the data)
# =============================================================================
def method_2_from_dataframe(views_df=None):
    """Load from a DataFrame you already have"""
    
    print("\n" + "=" * 80)
    print("METHOD 2: Loading from DataFrame")
    print("=" * 80)
    
    # If no DataFrame provided, create one from sample data
    if views_df is None:
        views_df = pd.DataFrame(SAMPLE_JSON_VIEWS)
    
    print(f"\nDataFrame shape: {views_df.shape}")
    print(f"Columns: {list(views_df.columns)}")
    print(f"\nFirst few views:")
    # Support both simple format (view_name) and Starburst format (object_name)
    if 'view_name' in views_df.columns:
        print(views_df[['view_name']].head())
    elif 'object_name' in views_df.columns:
        print(views_df[['db_name', 'schema_name', 'object_name']].head())
    else:
        print(views_df.head())
    
    # Initialize the index
    index = ViewIndex()
    
    # Load views from DataFrame
    for idx, row in views_df.iterrows():
        try:
            view_id = idx + 1
            # Support both formats: simple (view_name) or Starburst (db_name, schema_name, object_name)
            if 'view_name' in row.index:
                view_name = row.get('view_name', f'view_{view_id}')
            else:
                # Construct from Starburst columns
                db = row.get('db_name', 'unknown')
                schema = row.get('schema_name', 'unknown')
                obj = row.get('object_name', f'view_{view_id}')
                view_name = f"{db}.{schema}.{obj}"
            
            # Get lineage JSON (could be 'lineage' or 'linear_json')
            if 'lineage' in row.index:
                view_json = row.get('lineage', row)
            else:
                view_json = row.get('linear_json', row)
            
            # If view_json is a string, parse it as JSON
            if isinstance(view_json, str):
                view_json = json.loads(view_json)
            
            success = index.add_view(view_id, view_name, view_json)
            if not success:
                print(f"  ⚠ Failed to add view: {view_name}")
        except Exception as e:
            print(f"  ✗ Error loading view: {e}")
    
    print(f"\nLoaded {len(index.view_names)} views from DataFrame")
    print(f"Unique tables: {index.all_tables}")
    print(f"Total columns: {len(index.all_columns)}")
    
    return index


# =============================================================================
# METHOD 3: From a Python list (simplest for testing)
# =============================================================================
def method_3_from_list():
    """Load from a Python list of dictionaries"""
    
    print("\n" + "=" * 80)
    print("METHOD 3: Loading from Python List")
    print("=" * 80)
    
    print(f"\nUsing sample data with {len(SAMPLE_JSON_VIEWS)} views...")
    
    # Initialize the index
    index = ViewIndex()
    
    # Load views from list
    for idx, view_data in enumerate(SAMPLE_JSON_VIEWS, 1):
        try:
            view_id = idx
            view_name = view_data.get('view_name', f'view_{view_id}')
            view_json = view_data.get('lineage', view_data)
            
            success = index.add_view(view_id, view_name, view_json)
            status = "✓" if success else "✗"
            print(f"{status} View {view_id}: {view_name}")
        except Exception as e:
            print(f"✗ View {idx}: Error - {e}")
    
    print(f"\n{'='*80}")
    print(f"Index Summary:")
    print(f"  Total views loaded: {len(index.view_names)}")
    print(f"  Unique tables: {len(index.all_tables)}")
    print(f"  Unique columns: {len(index.all_columns)}")
    print(f"  Tables found: {index.all_tables}")
    
    return index


# =============================================================================
# MAIN: Run all methods or choose specific one
# =============================================================================
def main():
    """
    Demonstrates all three methods for loading views:
    1. From Starburst (production)
    2. From DataFrame (when you already have data)
    3. From Python list (testing/demo)
    
    You can uncomment any method to run it independently.
    """
    
    print("\n")
    print("#" * 80)
    print("# VIEW SIMILARITY FINDER - QUICK EXAMPLES")
    print("# Demonstrates 3 methods to load and analyze view lineage")
    print("#" * 80)
    
    # Run METHOD 1: Starburst (show explanation only)
    method_1_from_starburst()
    
    # Run METHOD 2: DataFrame (using sample data)
    index_df = method_2_from_dataframe()
    
    # Run METHOD 3: Python list (using sample data)
    index_list = method_3_from_list()
    
    # Show example analysis
    print("\n" + "=" * 80)
    print("EXAMPLE ANALYSIS")
    print("=" * 80)
    print("\nView Summary:")
    for idx, view_name in enumerate(index_list.view_names):
        features = index_list.view_features[idx] if idx < len(index_list.view_features) else {}
        tables = features.get('tables', set())
        columns = features.get('columns', set())
        print(f"  {view_name:40s} | Tables: {tables} | Columns: {len(columns)}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS FOR PRODUCTION")
    print("=" * 80)
    print("""
1. For Starburst integration:
   - Uncomment the "REAL IMPLEMENTATION" section in method_1_from_starburst()
   - Update DSN, USERNAME, PASSWORD
   - Update the SQL query to match your metadata table
   
2. For DataFrame:
   - Load your CSV/Excel/database data into a pandas DataFrame
   - Call method_2_from_dataframe(your_df)
   
3. For Python list:
   - Build your list of view JSON objects
   - Call method_3_from_list() with your data
   
Your JSON structure (EXACT):
{
    "view_name": "schema.view_name",
    "lineage": {
        "columns": [{"column_name": "col", "expression": "...", "lineage": [...]}],
        "joins": [{"type": "inner|left|outer|subquery", "table": "...", "condition": "..."}],
        "filters": [...],
        "alias_map": {"alias": "table_name" or {"type": "subquery", "sql": "SELECT..."}}
    }
}

Quote formats handled:
- Escaped quotes: ""value""
- Regular quotes: "value"
- Single quotes: 'value'
- Backticks: `value`

Subqueries supported:
- In joins: "table": "SELECT * FROM base_table"
- In alias_map: {"type": "subquery", "sql": "SELECT..."}
""")


if __name__ == "__main__":
    main()