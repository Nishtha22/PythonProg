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
from datetime import datetime


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
# EXPORT RESULTS: Analysis to Excel/CSV
# =============================================================================
# =============================================================================
# EXPORT RESULTS: Analysis to Excel/CSV
# =============================================================================
def get_view_details(view_json):
    """Extract detailed lineage information from view JSON"""
    lineage = view_json.get('lineage', view_json)
    
    # Get lineage structure
    joins_data = lineage.get('joins', [])
    filters_data = lineage.get('filters', [])
    columns_data = lineage.get('columns', [])
    alias_map = lineage.get('alias_map', {})
    
    # Extract join info
    join_types = []
    join_tables = []
    join_conditions = []
    for join in joins_data:
        if isinstance(join, dict):
            jtype = join.get('type', 'unknown')
            jtable = join.get('table', 'unknown')
            jcond = join.get('condition', 'N/A')
            join_types.append(jtype)
            join_tables.append(str(jtable)[:50])  # Truncate long SQL
            join_conditions.append(jcond[:100])
    
    # Extract filter info
    filter_str = '; '.join([str(f)[:80] for f in filters_data]) if filters_data else 'None'
    
    # Extract column info
    col_names = []
    for col in columns_data:
        if isinstance(col, dict):
            col_name = col.get('column_name', '')
            if col_name:
                col_names.append(col_name)
    
    return {
        'join_types': join_types,
        'join_tables': join_tables,
        'join_conditions': join_conditions,
        'filters': filter_str,
        'columns': col_names,
        'alias_map_count': len(alias_map)
    }


def compare_lineage(view_json_1, view_json_2):
    """Compare detailed lineage between two views"""
    details_1 = get_view_details(view_json_1)
    details_2 = get_view_details(view_json_2)
    
    comparison = {
        'joins_match': details_1['join_types'] == details_2['join_types'],
        'join_types_1': ', '.join(details_1['join_types']) or 'None',
        'join_types_2': ', '.join(details_2['join_types']) or 'None',
        'filters_match': details_1['filters'] == details_2['filters'],
        'filters_1': details_1['filters'],
        'filters_2': details_2['filters'],
        'columns_1': len(details_1['columns']),
        'columns_2': len(details_2['columns']),
    }
    
    # Generate recommendation
    recommendation = "EXACT DUPLICATE - Consider consolidating"
    if not comparison['joins_match']:
        recommendation = "DIFFERENT JOIN TYPES - Review join logic"
    elif not comparison['filters_match']:
        recommendation = "DIFFERENT FILTERS - Check business requirements"
    elif comparison['columns_1'] != comparison['columns_2']:
        recommendation = "DIFFERENT COLUMNS - Subset one view or merge"
    
    comparison['recommendation'] = recommendation
    
    return comparison


def export_similarity_report(index, output_file='view_similarity_report.xlsx', 
                            lineage_threshold=1.0):
    """
    Export similarity analysis to Excel file with detailed lineage comparison
    Shows: View pairs, similarity %, common tables, join/filter differences, recommendations
    
    Args:
        index: ViewIndex object with loaded views
        output_file: Output Excel filename
        lineage_threshold: Table overlap % to show lineage comparison (0.0-1.0)
                          1.0 = only 100% identical tables (default)
                          0.9 = show for 90%+ table overlap
                          0.5 = show for 50%+ table overlap (more detailed, slower)
                          0.3 = show for 30%+ table overlap (most detailed, much slower)
    """
    
    print(f"\nGenerating detailed similarity report...")
    print(f"  Lineage comparison threshold: {lineage_threshold*100:.0f}% table overlap")
    
    similarities = []
    detailed_comparisons = []
    
    # Calculate all similarity pairs
    for i in range(len(index.view_names)):
        for j in range(i + 1, len(index.view_names)):
            feat_i = index.view_features[i] if i < len(index.view_features) else {}
            feat_j = index.view_features[j] if j < len(index.view_features) else {}
            
            tables_i = feat_i.get('tables', set())
            tables_j = feat_j.get('tables', set())
            columns_i = feat_i.get('columns', set())
            columns_j = feat_j.get('columns', set())
            
            # Calculate table overlap (Jaccard similarity)
            if tables_i or tables_j:
                intersection = tables_i & tables_j
                union = tables_i | tables_j
                table_similarity = len(intersection) / len(union) if union else 0
            else:
                table_similarity = 0
            
            # Only include pairs with some similarity
            if table_similarity >= 0.3 or len(intersection) > 0:
                view_name_i = index.view_names[i]
                view_name_j = index.view_names[j]
                
                common_tables = intersection
                diff_i = tables_i - tables_j
                diff_j = tables_j - tables_i
                
                similarities.append({
                    'View_1': view_name_i,
                    'View_2': view_name_j,
                    'Similarity_%': round(table_similarity * 100, 2),
                    'Common_Tables': ', '.join(sorted(common_tables)) if common_tables else 'None',
                    'Tables_Only_in_View_1': ', '.join(sorted(diff_i)) if diff_i else 'None',
                    'Tables_Only_in_View_2': ', '.join(sorted(diff_j)) if diff_j else 'None',
                    'Total_Tables_View_1': len(tables_i),
                    'Total_Tables_View_2': len(tables_j),
                    'Columns_View_1': len(columns_i),
                    'Columns_View_2': len(columns_j),
                })
                
                # Add detailed comparison based on table_overlap probability threshold
                if table_similarity >= lineage_threshold:
                    view_json_i = index.view_features[i].get('raw_json', {}) if i < len(index.view_features) else {}
                    view_json_j = index.view_features[j].get('raw_json', {}) if j < len(index.view_features) else {}
                    
                    lineage_comp = compare_lineage(view_json_i, view_json_j)
                    
                    detailed_comparisons.append({
                        'View_1': view_name_i,
                        'View_2': view_name_j,
                        'Table_Overlap_%': round(table_similarity * 100, 1),
                        'Common_Tables': ', '.join(sorted(common_tables)),
                        'Join_Types_View_1': lineage_comp['join_types_1'],
                        'Join_Types_View_2': lineage_comp['join_types_2'],
                        'Joins_Match': 'YES' if lineage_comp['joins_match'] else 'NO',
                        'Filters_View_1': lineage_comp['filters_1'],
                        'Filters_View_2': lineage_comp['filters_2'],
                        'Filters_Match': 'YES' if lineage_comp['filters_match'] else 'NO',
                        'Columns_Count_View_1': lineage_comp['columns_1'],
                        'Columns_Count_View_2': lineage_comp['columns_2'],
                        'Recommendation': lineage_comp['recommendation'],
                    })
    
    # Sort by similarity descending
    similarities.sort(key=lambda x: x['Similarity_%'], reverse=True)
    
    # Create DataFrames
    df_similarities = pd.DataFrame(similarities)
    
    # Create summary statistics
    summary_stats = {
        'Metric': [
            'Total Views Analyzed',
            'Total View Pairs',
            'Identical Views (100% match)',
            'High Similarity (>80%)',
            'Medium Similarity (50-80%)',
            'Low Similarity (30-50%)',
            'Unique Tables',
            'Total Unique Columns',
            'Report Generated'
        ],
        'Value': [
            len(index.view_names),
            len(similarities),
            len([s for s in similarities if s['Similarity_%'] == 100]),
            len([s for s in similarities if 80 < s['Similarity_%'] < 100]),
            len([s for s in similarities if 50 <= s['Similarity_%'] <= 80]),
            len([s for s in similarities if 30 <= s['Similarity_%'] < 50]),
            len(index.all_tables),
            len(index.all_columns),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]
    }
    df_summary = pd.DataFrame(summary_stats)
    
    # Write to Excel with multiple sheets
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed similarities sheet
            if len(df_similarities) > 0:
                df_similarities.to_excel(writer, sheet_name='Similarities', index=False)
            
            # Detailed lineage comparison sheet (for identical tables)
            if len(detailed_comparisons) > 0:
                df_detailed = pd.DataFrame(detailed_comparisons)
                df_detailed.to_excel(writer, sheet_name='Lineage Comparison', index=False)
            
            # View details sheet
            view_details = []
            for idx, view_name in enumerate(index.view_names):
                features = index.view_features[idx] if idx < len(index.view_features) else {}
                tables = features.get('tables', set())
                columns = features.get('columns', set())
                
                view_details.append({
                    'View_Name': view_name,
                    'Tables': ', '.join(sorted(tables)) if tables else 'None',
                    'Table_Count': len(tables),
                    'Columns': ', '.join(sorted(columns)) if columns else 'None',
                    'Column_Count': len(columns),
                })
            
            df_views = pd.DataFrame(view_details)
            df_views.to_excel(writer, sheet_name='View Details', index=False)
        
        print(f"  ✓ Report exported to: {output_file}")
        print(f"  ✓ Total similar pairs found: {len(similarities)}")
        print(f"  ✓ Identical table pairs: {len(detailed_comparisons)}")
        if len(similarities) > 0:
            print(f"  ✓ Highest similarity: {similarities[0]['Similarity_%']}%")
        
        return output_file
    
    except ImportError:
        # If openpyxl not available, try CSV format
        print("  ⚠ Excel export not available (openpyxl not installed)")
        csv_file = output_file.replace('.xlsx', '.csv')
        
        # Export similarities to CSV
        df_similarities.to_csv(csv_file, index=False)
        print(f"  ✓ Report exported to CSV: {csv_file}")
        
        # Export summary to separate CSV
        summary_file = csv_file.replace('.csv', '_summary.csv')
        df_summary.to_csv(summary_file, index=False)
        print(f"  ✓ Summary exported to: {summary_file}")
        
        # Export detailed comparisons if available
        if len(detailed_comparisons) > 0:
            detailed_file = csv_file.replace('.csv', '_lineage.csv')
            pd.DataFrame(detailed_comparisons).to_csv(detailed_file, index=False)
            print(f"  ✓ Lineage comparison exported to: {detailed_file}")
        
        return csv_file


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
    
    # Export results to Excel file
    print("\n" + "=" * 80)
    print("EXPORTING RESULTS")
    print("=" * 80)
    export_file = export_similarity_report(index_list, 'view_similarity_report.xlsx')
    
    # Show example analysis with actual similarity detection
    print("\n" + "=" * 80)
    print("SIMILARITY ANALYSIS")
    print("=" * 80)
    
    print("\nView Summary:")
    for idx, view_name in enumerate(index_list.view_names):
        features = index_list.view_features[idx] if idx < len(index_list.view_features) else {}
        tables = features.get('tables', set())
        columns = features.get('columns', set())
        print(f"  {view_name:40s} | Tables: {tables} | Columns: {len(columns)}")
    
    # Find similar views
    print("\n" + "=" * 80)
    print("SIMILAR VIEWS DETECTED")
    print("=" * 80)
    
    similarities_found = 0
    for i in range(len(index_list.view_names)):
        for j in range(i + 1, len(index_list.view_names)):
            feat_i = index_list.view_features[i] if i < len(index_list.view_features) else {}
            feat_j = index_list.view_features[j] if j < len(index_list.view_features) else {}
            
            tables_i = feat_i.get('tables', set())
            tables_j = feat_j.get('tables', set())
            
            # Check if same tables (potential duplicates/similar views)
            if tables_i == tables_j:
                similarity_score = 1.0
                view_name_i = index_list.view_names[i]
                view_name_j = index_list.view_names[j]
                print(f"\n  ✓ EXACT MATCH (100% similar)")
                print(f"    View 1: {view_name_i}")
                print(f"    View 2: {view_name_j}")
                print(f"    Shared tables: {tables_i}")
                similarities_found += 1
            # Check for partial overlap
            elif tables_i & tables_j:  # Intersection
                intersection = tables_i & tables_j
                similarity_score = len(intersection) / max(len(tables_i), len(tables_j))
                if similarity_score >= 0.5:  # At least 50% match
                    view_name_i = index_list.view_names[i]
                    view_name_j = index_list.view_names[j]
                    print(f"\n  ◆ PARTIAL MATCH ({similarity_score*100:.0f}% similar)")
                    print(f"    View 1: {view_name_i} {tables_i}")
                    print(f"    View 2: {view_name_j} {tables_j}")
                    print(f"    Common tables: {intersection}")
                    similarities_found += 1
    
    if similarities_found == 0:
        print("  No similar views found (all views use different table combinations)")
    
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