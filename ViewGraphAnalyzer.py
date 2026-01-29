"""
SQL View Similarity Graph Analyzer
Uses NetworkX to create and analyze similarity graphs
Identifies clusters of similar views
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json
from collections import defaultdict


class ViewSimilarityGraphAnalyzer:
    """
    Creates and analyzes graph representations of view similarities
    Useful for identifying clusters and communities of similar views
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.views_data = {}
        
    def build_graph_from_pairs(self, similar_pairs: List[Tuple], views_data: List[Dict]):
        """
        Build a NetworkX graph from similar view pairs
        
        Args:
            similar_pairs: List of (view1_idx, view2_idx, tfidf_sim, struct_sim, combined_sim)
            views_data: List of view data dictionaries
        """
        print("Building similarity graph...")
        
        self.views_data = {i: view for i, view in enumerate(views_data)}
        
        # Add nodes
        for idx, view in enumerate(views_data):
            self.graph.add_node(
                idx,
                name=view['name'],
                id=view['id'],
                num_tables=len(view['structural_features']['tables']),
                num_columns=len(view['structural_features']['columns']),
                has_joins=len(view['structural_features']['joins']) > 0
            )
        
        # Add edges with similarity as weight
        for view1_idx, view2_idx, tfidf_sim, struct_sim, combined_sim in similar_pairs:
            self.graph.add_edge(
                view1_idx,
                view2_idx,
                weight=combined_sim,
                tfidf_similarity=tfidf_sim,
                structural_similarity=struct_sim
            )
        
        print(f"Graph created: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def find_communities(self) -> List[set]:
        """
        Find communities (clusters) of similar views using Louvain algorithm
        NetworkX supports this through the community module
        """
        print("\nFinding communities of similar views...")
        
        try:
            from networkx.algorithms import community
            
            # Use Louvain method for community detection
            communities = community.louvain_communities(self.graph, weight='weight')
            
            print(f"Found {len(communities)} communities")
            
            # Sort communities by size
            communities = sorted(communities, key=len, reverse=True)
            
            # Print community details
            for i, comm in enumerate(communities, 1):
                if len(comm) > 1:  # Only show communities with multiple views
                    view_names = [self.views_data[idx]['name'] for idx in comm]
                    print(f"\nCommunity {i} ({len(comm)} views):")
                    print(f"  Views: {', '.join(view_names[:5])}" + 
                          (f" ... and {len(view_names)-5} more" if len(view_names) > 5 else ""))
            
            return communities
            
        except ImportError:
            print("Warning: networkx.algorithms.community not available")
            print("Using connected components instead")
            return list(nx.connected_components(self.graph))
    
    def find_central_views(self, top_n: int = 10) -> List[Tuple]:
        """
        Find the most central views in the similarity network
        Central views are those most similar to many other views
        
        Args:
            top_n: Number of top central views to return
            
        Returns:
            List of (view_idx, view_name, centrality_score)
        """
        print(f"\nFinding top {top_n} central views...")
        
        # Calculate degree centrality (number of similar views)
        degree_centrality = nx.degree_centrality(self.graph)
        
        # Calculate weighted degree (sum of similarity scores)
        weighted_degree = {}
        for node in self.graph.nodes():
            weighted_degree[node] = sum(
                self.graph[node][neighbor]['weight'] 
                for neighbor in self.graph.neighbors(node)
            )
        
        # Combine both metrics
        central_views = []
        for node in self.graph.nodes():
            score = (degree_centrality[node] + weighted_degree[node]) / 2
            central_views.append((
                node,
                self.views_data[node]['name'],
                score
            ))
        
        # Sort by score
        central_views.sort(key=lambda x: x[2], reverse=True)
        
        print("\nMost central views (potential candidates for consolidation):")
        for rank, (idx, name, score) in enumerate(central_views[:top_n], 1):
            degree = self.graph.degree(idx)
            print(f"  {rank}. {name} (score: {score:.4f}, connected to {degree} views)")
        
        return central_views[:top_n]
    
    def identify_duplicate_candidates(self, similarity_threshold: float = 0.9) -> List[Tuple]:
        """
        Identify pairs of views that are very similar and might be duplicates
        
        Args:
            similarity_threshold: Minimum similarity to consider as duplicate
            
        Returns:
            List of (view1_idx, view2_idx, similarity_score)
        """
        print(f"\nIdentifying potential duplicates (similarity >= {similarity_threshold})...")
        
        duplicates = []
        for edge in self.graph.edges(data=True):
            view1_idx, view2_idx, data = edge
            similarity = data['weight']
            
            if similarity >= similarity_threshold:
                duplicates.append((
                    view1_idx,
                    self.views_data[view1_idx]['name'],
                    view2_idx,
                    self.views_data[view2_idx]['name'],
                    similarity
                ))
        
        duplicates.sort(key=lambda x: x[4], reverse=True)
        
        print(f"Found {len(duplicates)} potential duplicate pairs:")
        for view1_idx, name1, view2_idx, name2, sim in duplicates[:10]:
            print(f"  {name1} <-> {name2} (similarity: {sim:.4f})")
        
        return duplicates
    
    def export_graph_data(self, output_prefix: str = 'view_graph'):
        """
        Export graph data in various formats for further analysis
        """
        print(f"\nExporting graph data...")
        
        # Export edge list
        edge_list = []
        for edge in self.graph.edges(data=True):
            view1_idx, view2_idx, data = edge
            edge_list.append({
                'view1_name': self.views_data[view1_idx]['name'],
                'view2_name': self.views_data[view2_idx]['name'],
                'similarity': data['weight'],
                'tfidf_similarity': data['tfidf_similarity'],
                'structural_similarity': data['structural_similarity']
            })
        
        edge_df = pd.DataFrame(edge_list)
        edge_df.to_csv(f'{output_prefix}_edges.csv', index=False)
        print(f"  Saved edge list to: {output_prefix}_edges.csv")
        
        # Export node attributes
        node_list = []
        for node, attrs in self.graph.nodes(data=True):
            node_data = {
                'view_name': attrs['name'],
                'view_id': attrs['id'],
                'num_similar_views': self.graph.degree(node),
                'num_tables': attrs['num_tables'],
                'num_columns': attrs['num_columns'],
                'has_joins': attrs['has_joins']
            }
            node_list.append(node_data)
        
        node_df = pd.DataFrame(node_list)
        node_df.to_csv(f'{output_prefix}_nodes.csv', index=False)
        print(f"  Saved node attributes to: {output_prefix}_nodes.csv")
        
        # Export graph statistics
        stats = self.get_graph_statistics()
        with open(f'{output_prefix}_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved graph statistics to: {output_prefix}_statistics.json")
    
    def get_graph_statistics(self) -> Dict:
        """
        Calculate and return graph statistics
        """
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'num_connected_components': nx.number_connected_components(self.graph),
            'average_clustering': nx.average_clustering(self.graph),
        }
        
        # Degree distribution
        degrees = [self.graph.degree(node) for node in self.graph.nodes()]
        stats['average_degree'] = sum(degrees) / len(degrees) if degrees else 0
        stats['max_degree'] = max(degrees) if degrees else 0
        stats['min_degree'] = min(degrees) if degrees else 0
        
        # Connected components sizes
        components = list(nx.connected_components(self.graph))
        stats['largest_component_size'] = len(max(components, key=len)) if components else 0
        stats['num_isolated_nodes'] = sum(1 for comp in components if len(comp) == 1)
        
        return stats
    
    def visualize_graph(self, output_file: str = 'view_similarity_graph.png', 
                       max_nodes: int = 100):
        """
        Create a visualization of the similarity graph
        For large graphs (3000+ nodes), only visualizes the largest component
        
        Args:
            output_file: Output file path for the visualization
            max_nodes: Maximum number of nodes to visualize
        """
        print(f"\nCreating graph visualization...")
        
        # For large graphs, only visualize the largest connected component
        if self.graph.number_of_nodes() > max_nodes:
            print(f"Graph has {self.graph.number_of_nodes()} nodes. Visualizing largest component only.")
            components = list(nx.connected_components(self.graph))
            largest_component = max(components, key=len)
            subgraph = self.graph.subgraph(largest_component)
            
            # If still too large, take top connected nodes
            if len(largest_component) > max_nodes:
                # Get nodes with highest degree
                degree_dict = dict(subgraph.degree())
                top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
                top_node_ids = [node for node, _ in top_nodes]
                subgraph = self.graph.subgraph(top_node_ids)
        else:
            subgraph = self.graph
        
        # Create visualization
        plt.figure(figsize=(20, 16))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)
        
        # Node sizes based on degree
        node_sizes = [300 + subgraph.degree(node) * 50 for node in subgraph.nodes()]
        
        # Edge widths based on similarity
        edge_widths = [subgraph[u][v]['weight'] * 3 for u, v in subgraph.edges()]
        
        # Draw the graph
        nx.draw_networkx_nodes(
            subgraph, pos,
            node_size=node_sizes,
            node_color='lightblue',
            alpha=0.7
        )
        
        nx.draw_networkx_edges(
            subgraph, pos,
            width=edge_widths,
            alpha=0.3,
            edge_color='gray'
        )
        
        # Draw labels for nodes with high degree
        high_degree_nodes = {
            node: self.views_data[node]['name']
            for node in subgraph.nodes()
            if subgraph.degree(node) >= 3
        }
        
        nx.draw_networkx_labels(
            subgraph, pos,
            labels=high_degree_nodes,
            font_size=8
        )
        
        plt.title(f"View Similarity Graph\n({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges)", 
                 fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Graph visualization saved to: {output_file}")
        plt.close()


def integrate_with_similarity_analyzer():
    """
    Example showing how to integrate the graph analyzer with the main similarity analyzer
    """
    from view_similarity_analyzer import SQLViewSimilarityAnalyzer
    
    print("=" * 80)
    print("View Similarity Graph Analysis")
    print("=" * 80)
    
    # Step 1: Run the similarity analyzer
    # (Using sample data - replace with your actual data loading)
    sample_data = {
        'view_id': range(1, 21),
        'view_name': [f'view_{i}' for i in range(1, 21)],
        'view_definition': [
            '{"definition": "SELECT a, b FROM t1 JOIN t2 ON t1.id=t2.id"}',
            '{"definition": "SELECT a, b FROM t1 INNER JOIN t2 ON t1.id=t2.id"}',
            '{"definition": "SELECT c, d FROM t3 LEFT JOIN t4 ON t3.key=t4.key"}',
            '{"definition": "SELECT SUM(amount) FROM orders GROUP BY customer"}',
            '{"definition": "SELECT COUNT(*) FROM orders GROUP BY customer"}',
        ] * 4  # Repeat to get 20 views
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize and run similarity analyzer
    sim_analyzer = SQLViewSimilarityAnalyzer(similarity_threshold=0.5)
    sim_analyzer.load_views_from_dataframe(df, 'view_definition', 'view_id', 'view_name')
    sim_analyzer.build_tfidf_matrix()
    similar_pairs = sim_analyzer.find_similar_pairs_efficient(top_k=5)
    
    # Step 2: Create graph analysis
    graph_analyzer = ViewSimilarityGraphAnalyzer()
    graph_analyzer.build_graph_from_pairs(similar_pairs, sim_analyzer.views_data)
    
    # Step 3: Analyze the graph
    communities = graph_analyzer.find_communities()
    central_views = graph_analyzer.find_central_views(top_n=10)
    duplicates = graph_analyzer.identify_duplicate_candidates(similarity_threshold=0.85)
    
    # Step 4: Export results
    graph_analyzer.export_graph_data('view_similarity_graph')
    
    # Step 5: Visualize (if matplotlib is available)
    try:
        graph_analyzer.visualize_graph('view_similarity_graph.png', max_nodes=50)
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    # Step 6: Print summary statistics
    stats = graph_analyzer.get_graph_statistics()
    print("\n" + "=" * 80)
    print("GRAPH STATISTICS")
    print("=" * 80)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    integrate_with_similarity_analyzer()


    