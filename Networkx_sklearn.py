import json
import networkx as nx
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class ViewSimilarityAnalyzer:
    """
    Analyzer specifically designed for VIEW objects to calculate similarity scores.
    """
    
    def __init__(self):
        self.views = {}
        self.graphs = {}
        self.feature_matrix = None
        self.view_names = []
    
    def add_view(self, view_data: Dict):
        """
        Add a VIEW object to the analyzer.
        
        Expected format:
        {
            "view_name": "my_view",
            "definition": {
                "select": [{"column": "col1", "source": "table1"}, ...],
                "from": [{"table": "table1", "alias": "t1"}, ...],
                "joins": [{"type": "INNER", "source_table": "t1", "target_table": "t2"}, ...],
                "where_clause": "optional"
            }
        }
        """
        view_name = view_data.get('view_name') or view_data.get('object_name')
        
        if not view_name:
            print("Warning: View missing 'view_name' field, skipping")
            return
        
        self.views[view_name] = view_data
        self.view_names.append(view_name)
        self.graphs[view_name] = self._build_view_graph(view_data)
    
    def _build_view_graph(self, view_data: Dict) -> nx.DiGraph:
        """Build dependency graph for a VIEW"""
        G = nx.DiGraph()
        definition = view_data.get('definition', {})
        
        # Add metadata
        G.graph['view_name'] = view_data.get('view_name') or view_data.get('object_name')
        
        # Add source tables from FROM clause
        for table in definition.get('from', []):
            table_name = table.get('table', table) if isinstance(table, dict) else table
            G.add_node(table_name, node_type='table', source='from')
        
        # Add columns
        for col in definition.get('select', []):
            if isinstance(col, dict):
                col_name = col.get('column')
                col_source = col.get('source')
            else:
                col_name = col
                col_source = None
            
            if col_name:
                col_node = f"col_{col_name}"
                G.add_node(col_node, node_type='column', name=col_name)
                
                if col_source:
                    G.add_edge(col_source, col_node, edge_type='provides')
        
        # Add joins
        for join in definition.get('joins', []):
            if isinstance(join, dict):
                target_table = join.get('target_table')
                source_table = join.get('source_table')
                join_type = join.get('type', 'INNER')
                
                if target_table:
                    G.add_node(target_table, node_type='table')
                    if source_table:
                        G.add_edge(source_table, target_table, edge_type='join', join_type=join_type)
        
        # Add where clause indicator
        if definition.get('where_clause'):
            G.add_node('where_filter', node_type='filter')
        
        return G
    
    def extract_features(self) -> np.ndarray:
        """
        Extract feature vectors for all views.
        Features capture structural complexity and characteristics.
        """
        features = []
        
        for view_name in self.view_names:
            G = self.graphs[view_name]
            definition = self.views[view_name].get('definition', {})
            
            # Calculate graph metrics
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            
            # Count different node types
            num_tables = sum(1 for n, d in G.nodes(data=True) if d.get('node_type') == 'table')
            num_columns = sum(1 for n, d in G.nodes(data=True) if d.get('node_type') == 'column')
            num_filters = sum(1 for n, d in G.nodes(data=True) if d.get('node_type') == 'filter')
            
            # Count edge types
            num_joins = sum(1 for u, v, d in G.edges(data=True) if d.get('edge_type') == 'join')
            num_provides = sum(1 for u, v, d in G.edges(data=True) if d.get('edge_type') == 'provides')
            
            # Graph structure metrics
            density = nx.density(G) if num_nodes > 0 else 0
            is_connected = 1 if nx.is_weakly_connected(G) and num_nodes > 0 else 0
            num_components = len(list(nx.weakly_connected_components(G))) if num_nodes > 0 else 0
            avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0
            
            # Definition-based metrics
            num_select_cols = len(definition.get('select', []))
            num_from_tables = len(definition.get('from', []))
            num_join_clauses = len(definition.get('joins', []))
            has_where = 1 if definition.get('where_clause') else 0
            
            # Join complexity
            join_types = set()
            for join in definition.get('joins', []):
                if isinstance(join, dict):
                    join_types.add(join.get('type', 'INNER'))
            num_join_types = len(join_types)
            
            # Build feature vector
            feature_vector = [
                num_nodes,
                num_edges,
                density,
                num_tables,
                num_columns,
                num_filters,
                num_joins,
                num_provides,
                is_connected,
                num_components,
                avg_degree,
                num_select_cols,
                num_from_tables,
                num_join_clauses,
                has_where,
                num_join_types
            ]
            
            features.append(feature_vector)
        
        self.feature_matrix = np.array(features)
        return self.feature_matrix
    
    def calculate_similarity(self, view1_name: str, view2_name: str) -> Dict:
        """
        Calculate comprehensive similarity between two views.
        Returns similarity scores with detailed breakdown.
        """
        if view1_name not in self.views or view2_name not in self.views:
            return {'error': 'One or both views not found'}
        
        g1 = self.graphs[view1_name]
        g2 = self.graphs[view2_name]
        def1 = self.views[view1_name].get('definition', {})
        def2 = self.views[view2_name].get('definition', {})
        
        # 1. Structural similarity (graph-based)
        nodes1 = set(g1.nodes())
        nodes2 = set(g2.nodes())
        node_similarity = len(nodes1 & nodes2) / len(nodes1 | nodes2) if nodes1 | nodes2 else 0
        
        edges1 = set(g1.edges())
        edges2 = set(g2.edges())
        edge_similarity = len(edges1 & edges2) / len(edges1 | edges2) if edges1 | edges2 else 0
        
        structural_sim = (node_similarity + edge_similarity) / 2
        
        # 2. Feature-based similarity (cosine similarity of feature vectors)
        if self.feature_matrix is None:
            self.extract_features()
        
        idx1 = self.view_names.index(view1_name)
        idx2 = self.view_names.index(view2_name)
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.feature_matrix)
        feature_sim = cosine_similarity([scaled_features[idx1]], [scaled_features[idx2]])[0][0]
        
        # 3. Table similarity (what tables are used)
        tables1 = self._get_tables(g1)
        tables2 = self._get_tables(g2)
        table_similarity = len(tables1 & tables2) / len(tables1 | tables2) if tables1 | tables2 else 0
        
        # 4. Column similarity (what columns are selected)
        columns1 = self._get_columns(g1)
        columns2 = self._get_columns(g2)
        column_similarity = len(columns1 & columns2) / len(columns1 | columns2) if columns1 | columns2 else 0
        
        # 5. Join pattern similarity
        joins1 = set(j.get('type', 'INNER') for j in def1.get('joins', []) if isinstance(j, dict))
        joins2 = set(j.get('type', 'INNER') for j in def2.get('joins', []) if isinstance(j, dict))
        join_similarity = len(joins1 & joins2) / len(joins1 | joins2) if joins1 | joins2 else 0
        
        # Weighted composite score
        weights = {
            'structural': 0.25,
            'feature': 0.25,
            'table': 0.20,
            'column': 0.20,
            'join': 0.10
        }
        
        overall_similarity = (
            structural_sim * weights['structural'] +
            feature_sim * weights['feature'] +
            table_similarity * weights['table'] +
            column_similarity * weights['column'] +
            join_similarity * weights['join']
        )
        
        return {
            'view1': view1_name,
            'view2': view2_name,
            'overall_similarity': round(overall_similarity, 4),
            'breakdown': {
                'structural': round(structural_sim, 4),
                'feature_based': round(feature_sim, 4),
                'table_overlap': round(table_similarity, 4),
                'column_overlap': round(column_similarity, 4),
                'join_pattern': round(join_similarity, 4)
            },
            'common_elements': {
                'tables': sorted(list(tables1 & tables2)),
                'columns': sorted(list(columns1 & columns2)),
                'join_types': sorted(list(joins1 & joins2))
            },
            'unique_elements': {
                'view1_only_tables': sorted(list(tables1 - tables2)),
                'view2_only_tables': sorted(list(tables2 - tables1)),
                'view1_only_columns': sorted(list(columns1 - columns2)),
                'view2_only_columns': sorted(list(columns2 - columns1))
            }
        }
    
    def _get_tables(self, G: nx.DiGraph) -> Set[str]:
        """Extract table names from graph"""
        return {node for node, data in G.nodes(data=True) 
                if data.get('node_type') == 'table'}
    
    def _get_columns(self, G: nx.DiGraph) -> Set[str]:
        """Extract column names from graph"""
        return {data.get('name', '') for node, data in G.nodes(data=True) 
                if data.get('node_type') == 'column' and data.get('name')}
    
    def find_all_similarities(self, threshold: float = 0.0) -> List[Dict]:
        """
        Calculate similarity scores for all view pairs.
        Returns sorted list of similarity results.
        """
        results = []
        
        for i in range(len(self.view_names)):
            for j in range(i + 1, len(self.view_names)):
                view1_name = self.view_names[i]
                view2_name = self.view_names[j]
                
                similarity = self.calculate_similarity(view1_name, view2_name)
                
                if similarity.get('overall_similarity', 0) >= threshold:
                    results.append(similarity)
        
        return sorted(results, key=lambda x: x['overall_similarity'], reverse=True)
    
    def get_view_summary(self, view_name: str) -> Dict:
        """Get summary statistics for a view"""
        if view_name not in self.views:
            return {'error': 'View not found'}
        
        G = self.graphs[view_name]
        definition = self.views[view_name].get('definition', {})
        
        return {
            'view_name': view_name,
            'num_tables': len(self._get_tables(G)),
            'num_columns': len(self._get_columns(G)),
            'num_joins': len(definition.get('joins', [])),
            'has_where_clause': bool(definition.get('where_clause')),
            'tables': sorted(list(self._get_tables(G))),
            'columns': sorted(list(self._get_columns(G))),
            'complexity_score': G.number_of_nodes() + G.number_of_edges()
        }


# Example usage
if __name__ == "__main__":
    # Sample VIEW definitions
    views = [
        {
            "view_name": "sales_summary",
            "definition": {
                "select": [
                    {"column": "customer_id", "source": "customers"},
                    {"column": "total_amount", "source": "orders"},
                    {"column": "order_count", "source": "orders"}
                ],
                "from": [{"table": "orders", "alias": "o"}],
                "joins": [
                    {"type": "INNER", "source_table": "orders", "target_table": "customers"}
                ],
                "where_clause": "status = 'completed'"
            }
        },
        {
            "view_name": "revenue_report",
            "definition": {
                "select": [
                    {"column": "customer_id", "source": "customers"},
                    {"column": "total_amount", "source": "orders"},
                    {"column": "order_date", "source": "orders"}
                ],
                "from": [{"table": "orders", "alias": "o"}],
                "joins": [
                    {"type": "INNER", "source_table": "orders", "target_table": "customers"}
                ],
                "where_clause": "year = 2024"
            }
        },
        {
            "view_name": "product_sales",
            "definition": {
                "select": [
                    {"column": "product_id", "source": "products"},
                    {"column": "quantity", "source": "order_items"},
                    {"column": "revenue", "source": "order_items"}
                ],
                "from": [{"table": "order_items", "alias": "oi"}],
                "joins": [
                    {"type": "LEFT", "source_table": "order_items", "target_table": "products"}
                ]
            }
        },
        {
            "view_name": "customer_metrics",
            "definition": {
                "select": [
                    {"column": "customer_id", "source": "customers"},
                    {"column": "total_spent", "source": "orders"}
                ],
                "from": [{"table": "customers", "alias": "c"}],
                "joins": [
                    {"type": "LEFT", "source_table": "customers", "target_table": "orders"}
                ]
            }
        }
    ]
    
    # Initialize analyzer
    analyzer = ViewSimilarityAnalyzer()
    
    # Add all views
    print("Adding views to analyzer...")
    for view in views:
        analyzer.add_view(view)
    
    print(f"\nTotal views added: {len(analyzer.view_names)}\n")
    
    # Show view summaries
    print("=" * 80)
    print("VIEW SUMMARIES")
    print("=" * 80)
    for view_name in analyzer.view_names:
        summary = analyzer.get_view_summary(view_name)
        print(f"\n{summary['view_name']}:")
        print(f"  Tables: {summary['num_tables']} - {summary['tables']}")
        print(f"  Columns: {summary['num_columns']} - {summary['columns']}")
        print(f"  Joins: {summary['num_joins']}")
        print(f"  Has WHERE: {summary['has_where_clause']}")
        print(f"  Complexity: {summary['complexity_score']}")
    
    # Calculate all similarity scores
    print("\n" + "=" * 80)
    print("SIMILARITY SCORES")
    print("=" * 80)
    
    all_similarities = analyzer.find_all_similarities(threshold=0.0)
    
    for i, result in enumerate(all_similarities, 1):
        print(f"\n{i}. {result['view1']} vs {result['view2']}")
        print(f"   Overall Similarity: {result['overall_similarity']:.2%}")
        print(f"   Breakdown:")
        print(f"     - Structural: {result['breakdown']['structural']:.2%}")
        print(f"     - Feature-based: {result['breakdown']['feature_based']:.2%}")
        print(f"     - Table overlap: {result['breakdown']['table_overlap']:.2%}")
        print(f"     - Column overlap: {result['breakdown']['column_overlap']:.2%}")
        print(f"     - Join pattern: {result['breakdown']['join_pattern']:.2%}")
        
        if result['common_elements']['tables']:
            print(f"   Common tables: {result['common_elements']['tables']}")
        if result['common_elements']['columns']:
            print(f"   Common columns: {result['common_elements']['columns']}")
    
    # Show high similarity pairs
    print("\n" + "=" * 80)
    print("HIGH SIMILARITY PAIRS (>= 30%)")
    print("=" * 80)
    
    high_similarity = [r for r in all_similarities if r['overall_similarity'] >= 0.3]
    
    if high_similarity:
        for result in high_similarity:
            print(f"\n{result['view1']} ↔ {result['view2']}: {result['overall_similarity']:.2%}")
    else:
        print("\nNo view pairs with similarity >= 30%")