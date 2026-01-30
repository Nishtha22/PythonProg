#!/usr/bin/env python3
"""
View Similarity Finder for Starburst SQL Views
High-performance similarity detection for 3k-4k views without LLM
"""

import json
import hashlib
import time
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False
    print("Warning: pyodbc not available. Database connection features will be disabled.")
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SimilarityResult:
    """Structure for similarity results"""
    source_view: str
    similar_view: str
    similarity_score: float
    table_overlap: float
    column_overlap: float
    common_tables: Set[str]
    common_columns: Set[str]
    is_exact_match: bool
    table_count_diff: int
    column_count_diff: int


class ViewNormalizer:
    """Fast normalization of SQL view JSON structures"""
    
    @staticmethod
    def normalize_identifier(identifier: str) -> str:
        """
        Normalize identifiers (remove quotes, lowercase)
        Handles:
        - Regular quotes: "identifier"
        - Escaped quotes: ""identifier""
        - Single quotes: 'identifier'
        - Backticks: `identifier`
        - Subqueries with quotes: ""(SELECT...)""
        """
        if not identifier:
            return ""
        
        identifier = identifier.strip()
        
        # Handle escaped double quotes "" at start and end
        while identifier.startswith('""') and identifier.endswith('""'):
            identifier = identifier[2:-2].strip()
        
        # Handle regular quotes
        identifier = identifier.strip('"').strip("'").strip('`')
        
        # Handle parentheses for subqueries
        identifier = identifier.strip('()')
        
        return identifier.lower()
    
    @staticmethod
    def extract_tables(view_json: dict) -> Set[str]:
        """Extract all tables from alias_map (handles both top-level and nested under 'lineage')"""
        tables = set()
        
        # Check if alias_map is at top level or nested under 'lineage'
        alias_map = view_json.get('alias_map', {})
        
        # If not found at top level, check under 'lineage'
        if not alias_map and 'lineage' in view_json:
            lineage = view_json['lineage']
            if isinstance(lineage, dict):
                alias_map = lineage.get('alias_map', {})
        
        if not alias_map:
            return tables
        
        for alias, table_ref in alias_map.items():
            try:
                if isinstance(table_ref, str):
                    # Check if it's a SQL subquery
                    if 'select' in table_ref.lower() or table_ref.strip().startswith('('):
                        # Extract tables from SQL
                        sql_tables = ViewNormalizer._extract_tables_from_sql(table_ref)
                        tables.update(sql_tables)
                    else:
                        # Regular table name
                        normalized = ViewNormalizer.normalize_identifier(table_ref)
                        if normalized:
                            tables.add(normalized)
                elif isinstance(table_ref, dict):
                    # Handle subquery dict format: {"type": "subquery", "sql": "SELECT..."}
                    if table_ref.get('type') == 'subquery' and 'sql' in table_ref:
                        sql_tables = ViewNormalizer._extract_tables_from_sql(table_ref['sql'])
                        tables.update(sql_tables)
                    # Handle nested table references
                    elif 'table' in table_ref:
                        normalized = ViewNormalizer.normalize_identifier(table_ref['table'])
                        if normalized:
                            tables.add(normalized)
                    elif 'name' in table_ref:
                        normalized = ViewNormalizer.normalize_identifier(table_ref['name'])
                        if normalized:
                            tables.add(normalized)
            except Exception:
                # Skip problematic entries
                continue
        
        # Extract tables from joins
        joins = view_json.get('joins', [])
        if not joins and 'lineage' in view_json:
            lineage = view_json['lineage']
            if isinstance(lineage, dict):
                joins = lineage.get('joins', [])
        
        for join in joins:
            if isinstance(join, dict):
                # Handle lowercase keys (all keys are lowercase in Starburst format)
                table_name = join.get('table')
                if table_name:
                    if isinstance(table_name, str):
                        # Check if it's a SQL subquery
                        if 'select' in table_name.lower() or table_name.strip().startswith('('):
                            sql_tables = ViewNormalizer._extract_tables_from_sql(table_name)
                            tables.update(sql_tables)
                        else:
                            normalized = ViewNormalizer.normalize_identifier(table_name)
                            if normalized:
                                tables.add(normalized)
        
        return tables
    
    @staticmethod
    def _extract_tables_from_sql(sql_str: str) -> Set[str]:
        """
        Extract table names from SQL subquery strings
        Simple regex-based extraction for FROM and JOIN clauses
        """
        import re
        
        tables = set()
        
        if not isinstance(sql_str, str):
            return tables
        
        # Normalize the SQL - remove quotes
        sql_str = ViewNormalizer.normalize_identifier(sql_str)
        
        # Remove newlines and extra spaces
        sql_str = re.sub(r'\s+', ' ', sql_str)
        
        # Extract FROM clause tables
        from_pattern = r'from\s+([a-zA-Z0-9_.,\s\(\)]+?)(?:where|join|group|order|limit|$)'
        from_matches = re.findall(from_pattern, sql_str, re.IGNORECASE)
        for match in from_matches:
            for table in match.split(','):
                table = table.strip().strip('()').strip()
                if table and not table.lower().startswith('select'):
                    tables.add(table.lower())
        
        # Extract JOIN table names
        join_pattern = r'join\s+([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)?)'
        join_matches = re.findall(join_pattern, sql_str, re.IGNORECASE)
        for table in join_matches:
            if table:
                tables.add(table.lower())
        
        return tables
    
    @staticmethod
    def extract_columns(view_json: dict) -> Set[str]:
        """Extract column names from lineage.columns structure"""
        columns = set()
        
        # Check for columns at top level or nested under 'lineage'
        col_list = view_json.get('columns', [])
        
        # If not found at top level, check under 'lineage'
        if not col_list and 'lineage' in view_json:
            lineage = view_json['lineage']
            if isinstance(lineage, dict):
                col_list = lineage.get('columns', [])
        
        if not col_list:
            # If still no columns found, assume wildcard
            return {'*'}
        
        for col in col_list:
            try:
                if isinstance(col, dict):
                    # Try different possible keys
                    col_name = col.get('column_name', col.get('name', col.get('column', col.get('alias', ''))))
                elif isinstance(col, str):
                    col_name = col
                else:
                    continue
                
                if col_name == '*':
                    return {'*'}
                
                normalized = ViewNormalizer.normalize_identifier(col_name)
                if normalized:
                    columns.add(normalized)
            except Exception:
                continue
        
        return columns if columns else {'*'}
    
    @staticmethod
    def extract_join_types(view_json: dict) -> List[str]:
        """Extract join types from the view (handles lineage.joins structure)"""
        joins = []
        
        # Check for joins at top level
        joins_list = view_json.get('joins', [])
        
        # If not found at top level, check under 'lineage'
        if not joins_list and 'lineage' in view_json:
            lineage = view_json['lineage']
            if isinstance(lineage, dict):
                joins_list = lineage.get('joins', [])
        
        # Extract join types
        for join in joins_list:
            try:
                if isinstance(join, dict):
                    join_type = join.get('type', join.get('join_type'))
                    if join_type and isinstance(join_type, str):
                        joins.append(join_type.lower().strip())
                elif isinstance(join, str):
                    joins.append(join.lower().strip())
            except Exception:
                continue
        
        # Also traverse the entire structure in case joins are elsewhere
        def traverse(node, depth=0):
            if depth > 20:  # Prevent infinite recursion
                return
            
            if isinstance(node, dict):
                if 'join_type' in node or 'type' in node:
                    join_type = node.get('join_type', node.get('type'))
                    if isinstance(join_type, str) and join_type not in [None, 'null', '']:
                        joins.append(join_type.lower().strip())
                
                for value in node.values():
                    traverse(value, depth + 1)
            elif isinstance(node, list):
                for item in node:
                    traverse(item, depth + 1)
        
        try:
            traverse(view_json)
        except RecursionError:
            pass
        
        # Remove duplicates while preserving order
        seen = set()
        unique_joins = []
        for j in joins:
            if j not in seen and j not in ['null', 'none', '']:
                seen.add(j)
                unique_joins.append(j)
        
        return unique_joins
    
    @staticmethod
    def get_structure_hash(tables: Set[str], columns: Set[str], joins: List[str]) -> str:
        """Create a hash of the view structure for exact match detection"""
        fingerprint = f"{','.join(sorted(tables))}|{','.join(sorted(columns))}|{','.join(sorted(joins))}"
        return hashlib.md5(fingerprint.encode()).hexdigest()


class ViewIndex:
    """Multi-level index for fast similarity search"""
    
    def __init__(self):
        # Primary index: table set -> view IDs
        self.table_index: Dict[frozenset, List[int]] = defaultdict(list)
        
        # Secondary indexes
        self.view_features: List[Dict] = []
        self.view_names: List[str] = []
        self.view_ids_map: Dict[str, int] = {}  # view_name -> view_id
        
        # Exact match detection
        self.structure_hash_index: Dict[str, List[int]] = defaultdict(list)
        
        # Feature vocabularies
        self.all_tables: Set[str] = set()
        self.all_columns: Set[str] = set()
        
        # Statistics
        self.load_errors: List[Dict] = []
    
    def add_view(self, view_id: int, view_name: str, view_json: dict) -> bool:
        """Add a view to all indexes. Returns True if successful."""
        try:
            tables = ViewNormalizer.extract_tables(view_json)
            columns = ViewNormalizer.extract_columns(view_json)
            joins = ViewNormalizer.extract_join_types(view_json)
            
            # Skip views with no tables
            if not tables:
                self.load_errors.append({
                    'view_name': view_name,
                    'error': 'No tables found in view definition'
                })
                return False
            
            structure_hash = ViewNormalizer.get_structure_hash(tables, columns, joins)
            
            # Update indexes
            self.table_index[frozenset(tables)].append(view_id)
            self.structure_hash_index[structure_hash].append(view_id)
            
            # Store features
            features = {
                'tables': tables,
                'columns': columns,
                'joins': joins,
                'structure_hash': structure_hash,
                'table_count': len(tables),
                'column_count': len(columns) if '*' not in columns else 0,
                'join_count': len(joins),
                'has_wildcard': '*' in columns
            }
            
            self.view_features.append(features)
            self.view_names.append(view_name)
            self.view_ids_map[view_name] = view_id
            
            # Update vocabularies
            self.all_tables.update(tables)
            if '*' not in columns:
                self.all_columns.update(columns)
            
            return True
            
        except Exception as e:
            self.load_errors.append({
                'view_name': view_name,
                'error': str(e)
            })
            return False
    
    def find_candidates_by_tables(self, target_tables: Set[str], 
                                   min_overlap: float = 0.3) -> List[int]:
        """Fast candidate retrieval using table overlap"""
        if not target_tables:
            return []
        
        candidates = set()
        target_frozen = frozenset(target_tables)
        
        # Exact table match (fastest path)
        if target_frozen in self.table_index:
            return self.table_index[target_frozen]
        
        # Partial overlap
        for indexed_tables, view_ids in self.table_index.items():
            overlap = len(target_tables & indexed_tables)
            
            # Early continue if no overlap
            if overlap == 0:
                continue
            
            union = len(target_tables | indexed_tables)
            
            if union > 0 and (overlap / union) >= min_overlap:
                candidates.update(view_ids)
        
        return list(candidates)
    
    def get_exact_matches(self, structure_hash: str) -> List[int]:
        """Find views with identical structure"""
        return self.structure_hash_index.get(structure_hash, [])


class ViewSimilarityEngine:
    """Efficient similarity computation"""
    
    def __init__(self, index: ViewIndex):
        self.index = index
    
    def compute_similarity_batch(self, view_id: int, candidate_ids: List[int], 
                                 top_k: int = 10, min_score: float = 0.0) -> List[SimilarityResult]:
        """Compute detailed similarity for candidates"""
        if not candidate_ids:
            return []
        
        results = []
        source_features = self.index.view_features[view_id]
        source_name = self.index.view_names[view_id]
        
        # Check for exact matches first
        exact_match_ids = self.index.get_exact_matches(source_features['structure_hash'])
        
        for candidate_id in candidate_ids:
            if candidate_id == view_id:
                continue
            
            candidate_features = self.index.view_features[candidate_id]
            candidate_name = self.index.view_names[candidate_id]
            
            # Compute similarity
            sim_score, details = self._compute_composite_similarity(
                source_features, candidate_features
            )
            
            if sim_score >= min_score:
                is_exact = candidate_id in exact_match_ids
                
                result = SimilarityResult(
                    source_view=source_name,
                    similar_view=candidate_name,
                    similarity_score=round(sim_score, 4),
                    table_overlap=details['table_overlap'],
                    column_overlap=details['column_overlap'],
                    common_tables=details['common_tables'],
                    common_columns=details['common_columns'],
                    is_exact_match=is_exact,
                    table_count_diff=abs(source_features['table_count'] - 
                                        candidate_features['table_count']),
                    column_count_diff=abs(source_features['column_count'] - 
                                         candidate_features['column_count'])
                )
                results.append(result)
        
        # Sort by similarity and return top-k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]
    
    def _compute_composite_similarity(self, feat1: Dict, feat2: Dict) -> Tuple[float, Dict]:
        """Weighted similarity across multiple aspects"""
        # Table similarity (Jaccard) - CRITICAL
        table_sim = self._jaccard_similarity(feat1['tables'], feat2['tables'])
        
        # Early exit if no table overlap
        if table_sim == 0:
            return 0.0, self._get_empty_details()
        
        # Column similarity (only if both don't have wildcards)
        if feat1['has_wildcard'] or feat2['has_wildcard']:
            col_sim = 0.5  # Neutral score for wildcard cases
        else:
            col_sim = self._jaccard_similarity(feat1['columns'], feat2['columns'])
        
        # Join pattern similarity
        join_sim = self._sequence_similarity(feat1['joins'], feat2['joins'])
        
        # Structure size similarity
        if feat1['table_count'] > 0 and feat2['table_count'] > 0:
            size_sim = 1.0 - abs(feat1['table_count'] - feat2['table_count']) / max(
                feat1['table_count'], feat2['table_count']
            )
        else:
            size_sim = 0.0
        
        # Weighted combination (adjust weights based on your needs)
        composite = (
            0.50 * table_sim +      # Tables are most important
            0.25 * col_sim +        # Columns matter (but wildcards complicate this)
            0.15 * join_sim +       # Join patterns indicate similar logic
            0.10 * size_sim         # Structural complexity
        )
        
        # Get detailed breakdown
        details = {
            'table_overlap': table_sim,
            'column_overlap': col_sim,
            'common_tables': feat1['tables'] & feat2['tables'],
            'common_columns': feat1['columns'] & feat2['columns'] if not (
                feat1['has_wildcard'] or feat2['has_wildcard']) else set(),
        }
        
        return composite, details
    
    @staticmethod
    def _jaccard_similarity(set1: Set, set2: Set) -> float:
        """Jaccard similarity coefficient"""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def _sequence_similarity(seq1: List, seq2: List) -> float:
        """Simple sequence similarity for joins"""
        if not seq1 and not seq2:
            return 1.0
        if not seq1 or not seq2:
            return 0.0
        
        # Count common elements
        counter1 = Counter(seq1)
        counter2 = Counter(seq2)
        common = sum((counter1 & counter2).values())
        total = max(len(seq1), len(seq2))
        
        return common / total if total > 0 else 0.0
    
    @staticmethod
    def _get_empty_details() -> Dict:
        """Return empty details for zero similarity"""
        return {
            'table_overlap': 0.0,
            'column_overlap': 0.0,
            'common_tables': set(),
            'common_columns': set()
        }


class ViewSimilarityFinder:
    """Main orchestrator for finding similar views"""
    
    def __init__(self):
        self.index = ViewIndex()
        self.engine = None
        self.connection = None
    
    def connect_to_starburst(self, dsn: str, username: Optional[str] = None, 
                            password: Optional[str] = None):
        """Connect to Starburst using DSN"""
        if not PYODBC_AVAILABLE:
            print("✗ pyodbc module not available. Please install it with: pip install pyodbc")
            return False
        
        print(f"Connecting to Starburst DSN: {dsn}")
        
        try:
            if username and password:
                conn_string = f"DSN={dsn};UID={username};PWD={password}"
            else:
                conn_string = f"DSN={dsn}"
            
            self.connection = pyodbc.connect(conn_string)
            print("✓ Connected successfully")
            return True
            
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def load_views_from_query(self, query: str, view_name_col: str = 'view_name', 
                              view_json_col: str = 'view_json', batch_size: int = 100):
        """Load views from Starburst table"""
        if not self.connection:
            raise ValueError("Not connected to Starburst. Call connect_to_starburst() first.")
        
        print(f"\nExecuting query to load views...")
        print(f"Query: {query[:100]}...")
        
        start_time = time.time()
        
        try:
            # Use pandas for efficient loading
            df = pd.read_sql(query, self.connection)
            total_views = len(df)
            
            print(f"✓ Retrieved {total_views} views from Starburst")
            return self._load_from_dataframe(df, view_name_col, view_json_col, batch_size)
            
        except Exception as e:
            print(f"✗ Error loading views: {e}")
            raise
    
    def load_views_from_dataframe(self, df: pd.DataFrame, view_name_col: str = 'view_name',
                                  view_json_col: str = 'view_json', batch_size: int = 100):
        """Load views from a pandas DataFrame (useful when you already have the data)"""
        print(f"\nLoading {len(df)} views from DataFrame...")
        return self._load_from_dataframe(df, view_name_col, view_json_col, batch_size)
    
    def load_views_from_list(self, views_list: List[Dict], view_name_key: str = 'view_name',
                            batch_size: int = 100):
        """Load views from a list of dictionaries"""
        print(f"\nLoading {len(views_list)} views from list...")
        
        start_time = time.time()
        loaded_count = 0
        error_count = 0
        
        for idx, view_data in enumerate(views_list):
            view_name = view_data.get(view_name_key, f'view_{idx}')
            
            try:
                # The view_data itself is the JSON structure
                if self.index.add_view(idx, view_name, view_data):
                    loaded_count += 1
                else:
                    error_count += 1
                
                # Progress indicator
                if (idx + 1) % batch_size == 0:
                    elapsed = time.time() - start_time
                    rate = (idx + 1) / elapsed
                    print(f"  Processed {idx + 1}/{len(views_list)} views "
                          f"({rate:.1f} views/sec, {error_count} errors)")
            
            except Exception as e:
                self.index.load_errors.append({
                    'view_name': view_name,
                    'error': f'Unexpected error: {str(e)}'
                })
                error_count += 1
        
        elapsed = time.time() - start_time
        
        print(f"\n✓ Loading complete!")
        print(f"  Successfully loaded: {loaded_count} views")
        print(f"  Errors/Skipped: {error_count} views")
        print(f"  Total time: {elapsed:.2f} seconds")
        print(f"  Average rate: {len(views_list)/elapsed:.1f} views/sec")
        
        # Build similarity engine
        print(f"\nBuilding similarity engine...")
        self.engine = ViewSimilarityEngine(self.index)
        print(f"✓ Ready to find similarities!")
        
        # Print statistics
        self._print_statistics()
        
        return loaded_count
    
    def _load_from_dataframe(self, df: pd.DataFrame, view_name_col: str,
                            view_json_col: str, batch_size: int):
        """Internal method to load from DataFrame"""
        total_views = len(df)
        print(f"Loading into index...")
        
        start_time = time.time()
        loaded_count = 0
        error_count = 0
        
        for idx, row in df.iterrows():
            view_name = row[view_name_col]
            view_json_str = row[view_json_col]
            
            try:
                # Parse JSON if it's a string
                if isinstance(view_json_str, str):
                    view_json = json.loads(view_json_str)
                else:
                    view_json = view_json_str
                
                # Add to index
                if self.index.add_view(idx, view_name, view_json):
                    loaded_count += 1
                else:
                    error_count += 1
                
                # Progress indicator
                if (idx + 1) % batch_size == 0:
                    elapsed = time.time() - start_time
                    rate = (idx + 1) / elapsed
                    print(f"  Processed {idx + 1}/{total_views} views "
                          f"({rate:.1f} views/sec, {error_count} errors)")
            
            except json.JSONDecodeError as e:
                self.index.load_errors.append({
                    'view_name': view_name,
                    'error': f'JSON parse error: {str(e)}'
                })
                error_count += 1
            
            except Exception as e:
                self.index.load_errors.append({
                    'view_name': view_name,
                    'error': f'Unexpected error: {str(e)}'
                })
                error_count += 1
        
        elapsed = time.time() - start_time
        
        print(f"\n✓ Loading complete!")
        print(f"  Successfully loaded: {loaded_count} views")
        print(f"  Errors/Skipped: {error_count} views")
        print(f"  Total time: {elapsed:.2f} seconds")
        print(f"  Average rate: {total_views/elapsed:.1f} views/sec")
        
        # Build similarity engine
        print(f"\nBuilding similarity engine...")
        self.engine = ViewSimilarityEngine(self.index)
        print(f"✓ Ready to find similarities!")
        
        # Print statistics
        self._print_statistics()
        
        return loaded_count
    
    def find_similar_views(self, view_name: str, top_k: int = 10, 
                          min_similarity: float = 0.3,
                          min_table_overlap: float = 0.3) -> pd.DataFrame:
        """Find similar views for a given view"""
        if not self.engine:
            raise ValueError("Engine not initialized. Load views first.")
        
        # Find view ID
        if view_name not in self.index.view_ids_map:
            print(f"View '{view_name}' not found in index")
            return pd.DataFrame()
        
        view_id = self.index.view_ids_map[view_name]
        source_features = self.index.view_features[view_id]
        
        # Step 1: Fast candidate retrieval by tables
        candidates = self.index.find_candidates_by_tables(
            source_features['tables'], min_table_overlap
        )
        
        if not candidates:
            print(f"No candidates found for view '{view_name}'")
            return pd.DataFrame()
        
        print(f"Found {len(candidates)} candidates, computing similarities...")
        
        # Step 2: Detailed similarity computation
        results = self.engine.compute_similarity_batch(
            view_id, candidates, top_k, min_similarity
        )
        
        if not results:
            print(f"No similar views found above threshold {min_similarity}")
            return pd.DataFrame()
        
        # Format results
        return self._format_results(results)
    
    def find_all_similarities(self, top_k: int = 5, 
                             min_similarity: float = 0.4,
                             min_table_overlap: float = 0.3,
                             output_file: Optional[str] = None) -> pd.DataFrame:
        """Find similar views for ALL views (batch processing)"""
        if not self.engine:
            raise ValueError("Engine not initialized. Load views first.")
        
        all_results = []
        n_views = len(self.index.view_features)
        
        print(f"\nComputing similarities for all {n_views} views...")
        print(f"Parameters: top_k={top_k}, min_similarity={min_similarity}, "
              f"min_table_overlap={min_table_overlap}")
        
        start_time = time.time()
        batch_size = 100
        
        for view_id in range(n_views):
            source_features = self.index.view_features[view_id]
            
            # Fast candidate retrieval
            candidates = self.index.find_candidates_by_tables(
                source_features['tables'], min_table_overlap
            )
            
            if candidates:
                results = self.engine.compute_similarity_batch(
                    view_id, candidates, top_k, min_similarity
                )
                all_results.extend(results)
            
            # Progress indicator
            if (view_id + 1) % batch_size == 0:
                elapsed = time.time() - start_time
                rate = (view_id + 1) / elapsed
                eta = (n_views - view_id - 1) / rate
                print(f"  Processed {view_id + 1}/{n_views} views "
                      f"({rate:.1f} views/sec, ETA: {eta:.0f}s)")
        
        elapsed = time.time() - start_time
        
        print(f"\n✓ Similarity computation complete!")
        print(f"  Total time: {elapsed:.2f} seconds")
        print(f"  Average rate: {n_views/elapsed:.1f} views/sec")
        print(f"  Similar pairs found: {len(all_results)}")
        
        # Convert to DataFrame
        df = self._format_results(all_results)
        
        # Save to file if requested
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"  Results saved to: {output_file}")
        
        return df
    
    def get_view_clusters(self, min_similarity: float = 0.7) -> Dict[str, List[str]]:
        """Group views into clusters based on high similarity"""
        if not self.engine:
            raise ValueError("Engine not initialized. Load views first.")
        
        print(f"\nFinding view clusters (similarity >= {min_similarity})...")
        
        # Build adjacency based on high similarity
        clusters = {}
        processed = set()
        
        n_views = len(self.index.view_features)
        
        for view_id in range(n_views):
            view_name = self.index.view_names[view_id]
            
            if view_name in processed:
                continue
            
            source_features = self.index.view_features[view_id]
            candidates = self.index.find_candidates_by_tables(
                source_features['tables'], min_table_overlap=0.5
            )
            
            if not candidates:
                continue
            
            results = self.engine.compute_similarity_batch(
                view_id, candidates, top_k=100, min_score=min_similarity
            )
            
            if results:
                cluster = [view_name] + [r.similar_view for r in results]
                clusters[view_name] = cluster
                processed.update(cluster)
        
        print(f"✓ Found {len(clusters)} clusters")
        return clusters
    
    def _format_results(self, results: List[SimilarityResult]) -> pd.DataFrame:
        """Format results as DataFrame"""
        rows = []
        for result in results:
            rows.append({
                'source_view': result.source_view,
                'similar_view': result.similar_view,
                'similarity_score': result.similarity_score,
                'table_overlap': round(result.table_overlap, 4),
                'column_overlap': round(result.column_overlap, 4),
                'is_exact_match': result.is_exact_match,
                'table_count_diff': result.table_count_diff,
                'column_count_diff': result.column_count_diff,
                'common_tables': ', '.join(sorted(result.common_tables)),
                'common_columns': ', '.join(sorted(list(result.common_columns)[:20])),
            })
        
        return pd.DataFrame(rows)
    
    def _print_statistics(self):
        """Print index statistics"""
        print(f"\n{'='*60}")
        print(f"INDEX STATISTICS")
        print(f"{'='*60}")
        print(f"Total views loaded: {len(self.index.view_features)}")
        print(f"Unique table sets: {len(self.index.table_index)}")
        print(f"Unique structures: {len(self.index.structure_hash_index)}")
        print(f"Total unique tables: {len(self.index.all_tables)}")
        print(f"Total unique columns: {len(self.index.all_columns)}")
        
        if self.index.load_errors:
            print(f"\nLoad errors: {len(self.index.load_errors)}")
            print(f"First 5 errors:")
            for error in self.index.load_errors[:5]:
                print(f"  - {error['view_name']}: {error['error']}")
        
        print(f"{'='*60}\n")
    
    def export_errors(self, filename: str = 'load_errors.csv'):
        """Export loading errors to CSV"""
        if self.index.load_errors:
            df = pd.DataFrame(self.index.load_errors)
            df.to_csv(filename, index=False)
            print(f"Exported {len(self.index.load_errors)} errors to {filename}")
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("Database connection closed")


def main():
    """Example usage"""
    
    # Initialize finder
    finder = ViewSimilarityFinder()
    
    # Connect to Starburst
    DSN = "your_starburst_dsn"  # Replace with your DSN name
    USERNAME = "your_username"   # Optional
    PASSWORD = "your_password"   # Optional
    
    if not finder.connect_to_starburst(DSN, USERNAME, PASSWORD):
        return
    
    # Load views from Starburst
    query = """
        SELECT 
            view_name,
            view_definition_json as view_json
        FROM your_catalog.your_schema.your_views_table
        LIMIT 3500
    """
    
    finder.load_views_from_query(
        query, 
        view_name_col='view_name',
        view_json_col='view_json'
    )
    
    # Option 1: Find similar views for a specific view
    print("\n" + "="*60)
    print("Finding similar views for a specific view...")
    print("="*60)
    
    similar_df = finder.find_similar_views(
        'your_view_name', 
        top_k=10,
        min_similarity=0.3
    )
    print(similar_df)
    
    # Option 2: Find all similarities (takes longer)
    print("\n" + "="*60)
    print("Finding all view similarities...")
    print("="*60)
    
    all_similarities = finder.find_all_similarities(
        top_k=5,
        min_similarity=0.4,
        output_file='all_view_similarities.csv'
    )
    
    print(f"\nTop 10 most similar view pairs:")
    print(all_similarities.head(10))
    
    # Option 3: Find clusters of highly similar views
    print("\n" + "="*60)
    print("Finding view clusters...")
    print("="*60)
    
    clusters = finder.get_view_clusters(min_similarity=0.7)
    
    print(f"\nFound {len(clusters)} clusters:")
    for cluster_head, cluster_views in list(clusters.items())[:5]:
        print(f"\nCluster led by '{cluster_head}':")
        print(f"  Members ({len(cluster_views)}): {', '.join(cluster_views[:5])}")
        if len(cluster_views) > 5:
            print(f"  ... and {len(cluster_views) - 5} more")
    
    # Export any errors
    finder.export_errors('view_load_errors.csv')
    
    # Close connection
    finder.close()


if __name__ == "__main__":
    main()