#!/usr/bin/env python3
"""
IMPROVED: View Similarity with FULL comparison (tables, columns, joins, filters)

This script demonstrates that the ViewSimilarityFinder DOES compare:
- Tables (50% weight)
- Columns (25% weight) 
- Join types (15% weight)
- Structure size (10% weight)

The composite similarity score includes ALL these factors.
"""

from view_similarity_finder_1 import ViewSimilarityFinder
import pandas as pd

# =============================================================================
# CONNECT TO STARBURST AND LOAD VIEWS
# =============================================================================

def load_views_from_starburst(dsn, username, password, query):
    """
    Load views from Starburst with FULL similarity comparison
    """
    
    print("="*80)
    print("LOADING VIEWS FROM STARBURST")
    print("="*80)
    
    finder = ViewSimilarityFinder()
    
    # Connect
    if not finder.connect_to_starburst(dsn, username, password):
        print("Failed to connect!")
        return None
    
    # Load views
    try:
        finder.load_views_from_query(
            query,
            view_name_col='view_name',
            view_json_col='view_json',
            batch_size=100
        )
        
        print(f"\n✓ Successfully loaded {len(finder.index.view_names)} views")
        return finder
        
    except Exception as e:
        print(f"✗ Error loading views: {e}")
        return None


# =============================================================================
# FIND SIMILARITIES WITH FULL COMPARISON
# =============================================================================

def find_and_analyze_similarities(finder, min_similarity=0.3, output_file='full_similarity_report.csv'):
    """
    Find similar views using COMPOSITE similarity (tables + columns + joins + structure)
    
    The similarity_score in the output IS the composite score that includes:
    - 50% table overlap
    - 25% column overlap
    - 15% join pattern similarity
    - 10% structure size similarity
    """
    
    print("\n" + "="*80)
    print("COMPUTING SIMILARITIES (Tables + Columns + Joins + Structure)")
    print("="*80)
    
    # This method DOES compare everything
    results = finder.find_all_similarities(
        top_k=10,
        min_similarity=min_similarity,
        min_table_overlap=0.3,
        output_file=output_file
    )
    
    if results.empty:
        print("\nNo similar views found above threshold")
        return results
    
    print("\n" + "="*80)
    print("SIMILARITY REPORT")
    print("="*80)
    
    print(f"\nTotal similar pairs found: {len(results)}")
    print(f"\nBreakdown:")
    print(f"  Exact matches (similarity = 1.0): {len(results[results['similarity_score'] >= 0.99])}")
    print(f"  Very high (0.8-0.99): {len(results[(results['similarity_score'] >= 0.8) & (results['similarity_score'] < 0.99)])}")
    print(f"  High (0.6-0.8): {len(results[(results['similarity_score'] >= 0.6) & (results['similarity_score'] < 0.8)])}")
    print(f"  Medium (0.4-0.6): {len(results[(results['similarity_score'] >= 0.4) & (results['similarity_score'] < 0.6)])}")
    print(f"  Low (0.3-0.4): {len(results[(results['similarity_score'] >= 0.3) & (results['similarity_score'] < 0.4)])}")
    
    print(f"\nAverage scores:")
    print(f"  Composite similarity: {results['similarity_score'].mean():.2%}")
    print(f"  Table overlap: {results['table_overlap'].mean():.2%}")
    print(f"  Column overlap: {results['column_overlap'].mean():.2%}")
    
    # Show top 10 most similar
    print(f"\nTop 10 most similar view pairs:")
    print("-" * 80)
    top_10 = results.nlargest(10, 'similarity_score')
    
    for idx, row in top_10.iterrows():
        print(f"\n{row['source_view']} ↔ {row['similar_view']}")
        print(f"  Composite Score: {row['similarity_score']:.2%}")
        print(f"  Table overlap: {row['table_overlap']:.2%}")
        print(f"  Column overlap: {row['column_overlap']:.2%}")
        print(f"  Exact match: {'YES' if row['is_exact_match'] else 'NO'}")
        print(f"  Common tables: {row['common_tables']}")
    
    # Analyze cases where tables match but overall score differs
    print("\n" + "="*80)
    print("ANALYZING: Same Tables, Different Columns/Joins")
    print("="*80)
    
    same_tables_diff_score = results[
        (results['table_overlap'] >= 0.9) &  # Same tables
        (results['similarity_score'] < 0.9)   # But different overall
    ]
    
    if len(same_tables_diff_score) > 0:
        print(f"\nFound {len(same_tables_diff_score)} view pairs with same tables but different columns/joins:")
        print("-" * 80)
        
        for idx, row in same_tables_diff_score.head(10).iterrows():
            print(f"\n{row['source_view']} ↔ {row['similar_view']}")
            print(f"  Tables: {row['table_overlap']:.0%} match (basically same)")
            print(f"  Columns: {row['column_overlap']:.0%} match")
            print(f"  Overall: {row['similarity_score']:.0%} similar")
            print(f"  → Difference is due to: ", end="")
            if row['column_overlap'] < 0.5:
                print("different columns selected")
            elif row['table_count_diff'] > 2:
                print("different number of tables")
            else:
                print("different join patterns or filters")
    else:
        print("\nNo views with same tables but different scores found.")
        print("This means all views with same tables also have similar columns/joins.")
    
    return results


# =============================================================================
# DETAILED ANALYSIS: Column-Level Comparison
# =============================================================================

def analyze_column_differences(finder, view1_name, view2_name):
    """
    Show detailed column-level differences between two views
    """
    
    print("\n" + "="*80)
    print(f"DETAILED COMPARISON: {view1_name} vs {view2_name}")
    print("="*80)
    
    # Get view IDs
    view1_id = finder.index.view_ids_map.get(view1_name)
    view2_id = finder.index.view_ids_map.get(view2_name)
    
    if view1_id is None or view2_id is None:
        print("One or both views not found")
        return
    
    # Get features
    feat1 = finder.index.view_features[view1_id]
    feat2 = finder.index.view_features[view2_id]
    
    # Compare tables
    print("\nTABLES:")
    tables1 = feat1['tables']
    tables2 = feat2['tables']
    common_tables = tables1 & tables2
    only_in_1 = tables1 - tables2
    only_in_2 = tables2 - tables1
    
    print(f"  Common tables ({len(common_tables)}): {', '.join(sorted(common_tables))}")
    if only_in_1:
        print(f"  Only in {view1_name}: {', '.join(sorted(only_in_1))}")
    if only_in_2:
        print(f"  Only in {view2_name}: {', '.join(sorted(only_in_2))}")
    
    table_overlap = len(common_tables) / len(tables1 | tables2) if (tables1 | tables2) else 0
    print(f"  Table overlap: {table_overlap:.0%}")
    
    # Compare columns
    print("\nCOLUMNS:")
    columns1 = feat1['columns']
    columns2 = feat2['columns']
    
    if '*' in columns1 or '*' in columns2:
        print("  One or both views use SELECT *")
        print(f"  {view1_name}: {'*' if '*' in columns1 else f'{len(columns1)} specific columns'}")
        print(f"  {view2_name}: {'*' if '*' in columns2 else f'{len(columns2)} specific columns'}")
    else:
        common_cols = columns1 & columns2
        only_in_1 = columns1 - columns2
        only_in_2 = columns2 - columns1
        
        print(f"  Common columns ({len(common_cols)}): {', '.join(sorted(list(common_cols)[:10]))}")
        if len(common_cols) > 10:
            print(f"    ... and {len(common_cols) - 10} more")
        
        if only_in_1:
            print(f"  Only in {view1_name} ({len(only_in_1)}): {', '.join(sorted(list(only_in_1)[:5]))}")
            if len(only_in_1) > 5:
                print(f"    ... and {len(only_in_1) - 5} more")
        
        if only_in_2:
            print(f"  Only in {view2_name} ({len(only_in_2)}): {', '.join(sorted(list(only_in_2)[:5]))}")
            if len(only_in_2) > 5:
                print(f"    ... and {len(only_in_2) - 5} more")
        
        col_overlap = len(common_cols) / len(columns1 | columns2) if (columns1 | columns2) else 0
        print(f"  Column overlap: {col_overlap:.0%}")
    
    # Compare joins
    print("\nJOINS:")
    joins1 = feat1['joins']
    joins2 = feat2['joins']
    
    print(f"  {view1_name}: {', '.join(joins1) if joins1 else 'None'}")
    print(f"  {view2_name}: {', '.join(joins2) if joins2 else 'None'}")
    
    if joins1 == joins2:
        print("  ✓ Join types match")
    else:
        print("  ✗ Join types differ")
    
    # Compute composite similarity
    print("\nCOMPOSITE SIMILARITY BREAKDOWN:")
    
    # Use the same logic as the engine
    if feat1['has_wildcard'] or feat2['has_wildcard']:
        col_sim = 0.5
    else:
        col_sim = len(columns1 & columns2) / len(columns1 | columns2) if (columns1 | columns2) else 0
    
    # Join similarity
    if not joins1 and not joins2:
        join_sim = 1.0
    elif not joins1 or not joins2:
        join_sim = 0.0
    else:
        from collections import Counter
        counter1 = Counter(joins1)
        counter2 = Counter(joins2)
        common = sum((counter1 & counter2).values())
        join_sim = common / max(len(joins1), len(joins2))
    
    # Size similarity
    size_sim = 1.0 - abs(feat1['table_count'] - feat2['table_count']) / max(
        feat1['table_count'], feat2['table_count']
    ) if (feat1['table_count'] > 0 and feat2['table_count'] > 0) else 0.0
    
    composite = 0.50 * table_overlap + 0.25 * col_sim + 0.15 * join_sim + 0.10 * size_sim
    
    print(f"  Table similarity:     {table_overlap:.2%} (weight: 50%) = {0.50 * table_overlap:.2%}")
    print(f"  Column similarity:    {col_sim:.2%} (weight: 25%) = {0.25 * col_sim:.2%}")
    print(f"  Join similarity:      {join_sim:.2%} (weight: 15%) = {0.15 * join_sim:.2%}")
    print(f"  Size similarity:      {size_sim:.2%} (weight: 10%) = {0.10 * size_sim:.2%}")
    print(f"  " + "-"*70)
    print(f"  COMPOSITE SCORE:      {composite:.2%}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function showing FULL similarity comparison
    """
    
    print("\n" + "="*80)
    print("VIEW SIMILARITY FINDER - FULL COMPARISON DEMO")
    print("Demonstrates that columns, joins, and structure ARE compared")
    print("="*80)
    
    # Configuration - UPDATE THESE
    DSN = "your_starburst_dsn"
    USERNAME = "your_username"
    PASSWORD = "your_password"
    
    # Query - UPDATE THIS
    QUERY = """
        SELECT 
            view_name,
            view_definition_json as view_json
        FROM your_catalog.your_schema.views_table
        WHERE view_name IS NOT NULL
        LIMIT 3500
    """
    
    # Load views
    finder = load_views_from_starburst(DSN, USERNAME, PASSWORD, QUERY)
    
    if finder is None:
        print("\nFailed to load views. Exiting.")
        return
    
    # Find similarities with FULL comparison
    results = find_and_analyze_similarities(
        finder,
        min_similarity=0.3,
        output_file='full_similarity_report.csv'
    )
    
    # If we have results, show detailed comparison for top pair
    if not results.empty:
        top_pair = results.iloc[0]
        analyze_column_differences(
            finder,
            top_pair['source_view'],
            top_pair['similar_view']
        )
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: What's Being Compared?")
    print("="*80)
    print("""
The similarity_score in your results IS a composite score that includes:

1. TABLES (50% weight):
   - Jaccard similarity of tables used
   - If tables don't overlap, similarity = 0 (early exit)

2. COLUMNS (25% weight):
   - Jaccard similarity of column names
   - If view uses SELECT *, column similarity = 0.5 (neutral)
   - Otherwise, compares specific column names

3. JOIN PATTERNS (15% weight):
   - Compares join types (INNER, LEFT, RIGHT, OUTER, etc.)
   - Counts how many join types match between views

4. STRUCTURE SIZE (10% weight):
   - Compares number of tables, columns, joins
   - Views with similar complexity score higher

EXAMPLE:
  View A: customers + orders, 5 columns, INNER JOIN
  View B: customers + orders, 3 columns, LEFT JOIN
  
  Result:
    - Table overlap: 100% (same tables)
    - Column overlap: 60% (some columns differ)
    - Join similarity: 0% (INNER vs LEFT)
    - Size similarity: 80% (5 vs 3 tables)
    
    Composite = 0.5×100% + 0.25×60% + 0.15×0% + 0.1×80%
              = 50% + 15% + 0% + 8%
              = 73% similarity

So even with same tables, you get different scores based on columns/joins!
    """)
    
    print("\n" + "="*80)
    print("FILES CREATED:")
    print("="*80)
    print("  1. full_similarity_report.csv - All similarity pairs with composite scores")
    print("     Columns: similarity_score, table_overlap, column_overlap, is_exact_match")
    print("\nThe similarity_score column IS the composite score!")
    print("="*80)


if __name__ == "__main__":
    main()