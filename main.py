import os
import networkx as nx
import numpy as np
import random
import community.community_louvain as community_louvain
from scipy.stats import ttest_ind, f_oneway
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm  # For progress bar
import logging

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Directories for data input and output
DATA_DIR = "data"  # Directory containing .edges files
RESULTS_DIR = "results"  # Directory to save CSV results
GRAPHS_DIR = "graphs"  # Directory to save generated graphs
LOGS_DIR = "logs"  # Directory to save log files

# Create output directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging to record warnings and errors
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, 'network_analysis.log'),  # Log file path
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    level=logging.INFO  # Logging level
)


# =============================================================================
# 2. Load Data and Construct Graph
# =============================================================================

def load_edges(filename: str) -> List[Tuple[int, int]]:
    """
    Load edges from a .edges file.

    Each line in the file should contain two integers representing an edge between two nodes.

    Parameters:
    - filename (str): Path to the .edges file.

    Returns:
    - List[Tuple[int, int]]: A list of tuples where each tuple represents an edge (source, target).
    """
    edges = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    source, target = map(int, parts)
                    edges.append((source, target))
    except Exception as e:
        logging.error(f"Failed to load edges from {filename}: {e}")
    return edges


def construct_graph(edges: List[Tuple[int, int]], ego_node: int) -> nx.Graph:
    """
    Construct an undirected graph from a list of edges and add the ego node.

    Parameters:
    - edges (List[Tuple[int, int]]): List of edges as tuples (source, target).
    - ego_node (int): The ID of the ego node to be added to the graph.

    Returns:
    - nx.Graph: A NetworkX graph with the ego node and all provided edges.
    """
    G = nx.Graph()
    G.add_node(ego_node)  # Add ego node
    G.add_edges_from(edges)  # Add edges to the graph
    return G


# =============================================================================
# 5. Calculate Average Clustering Coefficient Within Communities
# =============================================================================

def calculate_community_clustering(graph: nx.Graph, partition: Dict[int, int]) -> Dict[int, float]:
    """
    Calculate the average clustering coefficient for each community.

    Parameters:
    - graph (nx.Graph): The NetworkX graph.
    - partition (Dict[int, int]): A dictionary mapping node IDs to community IDs.

    Returns:
    - Dict[int, float]: A dictionary mapping community IDs to their average clustering coefficient.
    """
    community_clustering = {}
    communities = {}
    # Group nodes by their community ID
    for node, community_id in partition.items():
        communities.setdefault(community_id, []).append(node)
    # Calculate average clustering coefficient for each community
    for community_id, nodes in communities.items():
        if len(nodes) == 0:
            community_clustering[community_id] = 0.0
            continue
        subgraph = graph.subgraph(nodes)
        if len(subgraph) == 0:
            community_clustering[community_id] = 0.0
            continue
        clustering_coeffs = nx.clustering(subgraph)
        if len(clustering_coeffs) == 0:
            avg_clustering = 0.0
        else:
            avg_clustering = sum(clustering_coeffs.values()) / len(clustering_coeffs)
        community_clustering[community_id] = avg_clustering
    return community_clustering


# =============================================================================
# 6. Identify Highly Clustered Communities
# =============================================================================

def get_highly_clustered_communities(community_clustering: Dict[int, float], network_avg_clustering: float) -> List[
    int]:
    """
    Identify communities with average clustering coefficient above the network's average.

    Parameters:
    - community_clustering (Dict[int, float]): Average clustering coefficient per community.
    - network_avg_clustering (float): The overall average clustering coefficient of the network.

    Returns:
    - List[int]: A list of community IDs that are highly clustered.
    """
    highly_clustered_communities = [
        community_id for community_id, avg_clustering in community_clustering.items()
        if avg_clustering > network_avg_clustering
    ]
    return highly_clustered_communities


# =============================================================================
# 7. Identify High-Degree Nodes (Top 10%)
# =============================================================================

def get_high_degree_nodes(degrees: Dict[int, int], percentile: float = 90) -> List[int]:
    """
    Identify high-degree nodes based on a percentile threshold.

    Parameters:
    - degrees (Dict[int, int]): A dictionary mapping node IDs to their degree.
    - percentile (float): The percentile threshold to define high-degree nodes (default is 90).

    Returns:
    - List[int]: A list of node IDs that are considered high-degree.
    """
    degree_values = list(degrees.values())
    if not degree_values:
        return []
    threshold = np.percentile(degree_values, percentile)
    high_degree_nodes = [node for node, degree in degrees.items() if degree >= threshold]
    return high_degree_nodes


# =============================================================================
# 8. Identify Bridge Nodes
# =============================================================================

def get_bridge_nodes(high_degree_nodes: List[int], partition: Dict[int, int],
                     highly_clustered_communities: List[int]) -> List[int]:
    """
    Identify bridge nodes which are high-degree nodes within highly clustered communities.

    Parameters:
    - high_degree_nodes (List[int]): List of high-degree node IDs.
    - partition (Dict[int, int]): Dictionary mapping node IDs to community IDs.
    - highly_clustered_communities (List[int]): List of community IDs that are highly clustered.

    Returns:
    - List[int]: List of bridge node IDs.
    """
    bridge_nodes = [
        node for node in high_degree_nodes
        if partition.get(node) in highly_clustered_communities
    ]
    return bridge_nodes


# =============================================================================
# 9. Identify High-Degree Nodes Not in Highly Clustered Communities
# =============================================================================

def get_high_degree_nodes_not_in_highly_clustered_communities(
        high_degree_nodes: List[int],
        partition: Dict[int, int],
        highly_clustered_communities: List[int]
) -> List[int]:
    """
    Identify high-degree nodes that are not part of highly clustered communities.

    Parameters:
    - high_degree_nodes (List[int]): List of high-degree node IDs.
    - partition (Dict[int, int]): Dictionary mapping node IDs to community IDs.
    - highly_clustered_communities (List[int]): List of community IDs that are highly clustered.

    Returns:
    - List[int]: List of high-degree node IDs not in highly clustered communities.
    """
    nodes = [
        node for node in high_degree_nodes
        if partition.get(node) not in highly_clustered_communities
    ]
    return nodes


# =============================================================================
# 10. Select Random Nodes
# =============================================================================

def get_random_nodes(graph: nx.Graph, num_nodes: int) -> List[int]:
    """
    Randomly select a specified number of nodes from the graph.

    Parameters:
    - graph (nx.Graph): The NetworkX graph.
    - num_nodes (int): Number of random nodes to select.

    Returns:
    - List[int]: List of randomly selected node IDs.
    """
    available_nodes = list(graph.nodes())
    if num_nodes > len(available_nodes):
        num_nodes = len(available_nodes)
    return random.sample(available_nodes, num_nodes)


# =============================================================================
# 11. Define the Independent Cascade Model Function
# =============================================================================

def independent_cascade(
        graph: nx.Graph,
        seeds: List[int],
        propagation_prob: float = 0.1,
        max_steps: int = None
) -> List[int]:
    """
    Simulate the Independent Cascade Model for information spread.

    Parameters:
    - graph (nx.Graph): The NetworkX graph representing the network.
    - seeds (List[int]): List of initial seed node IDs to start the spread.
    - propagation_prob (float): Probability of propagation along each edge (default is 0.1).
    - max_steps (int, optional): Maximum number of steps to run the simulation.

    Returns:
    - List[int]: List of node IDs that were activated during the spread.
    """
    activated_nodes = set(seeds)  # Nodes that have been activated
    newly_activated_nodes = set(seeds)  # Nodes activated in the current step
    steps = 0  # Step counter

    while newly_activated_nodes and (max_steps is None or steps < max_steps):
        steps += 1
        next_newly_activated = set()  # Nodes to be activated in the next step
        for node in newly_activated_nodes:
            neighbors = set(graph.neighbors(node)) - activated_nodes
            for neighbor in neighbors:
                if random.random() <= propagation_prob:
                    next_newly_activated.add(neighbor)
                    activated_nodes.add(neighbor)
        newly_activated_nodes = next_newly_activated

    return list(activated_nodes)


# =============================================================================
# 12. Define Simulation Function
# =============================================================================

def simulate_spread(
        graph: nx.Graph,
        seed_nodes: List[int],
        num_simulations: int = 100,
        propagation_prob: float = 0.1
) -> Dict[int, List[int]]:
    """
    Simulate the spread of information from multiple seed nodes.

    Parameters:
    - graph (nx.Graph): The NetworkX graph representing the network.
    - seed_nodes (List[int]): List of seed node IDs to simulate spread from.
    - num_simulations (int): Number of simulation runs per seed node (default is 100).
    - propagation_prob (float): Probability of propagation along each edge (default is 0.1).

    Returns:
    - Dict[int, List[int]]: Dictionary mapping seed node IDs to lists of spread sizes from simulations.
    """
    results = {}
    for seed in seed_nodes:
        spread_sizes = []
        for _ in range(num_simulations):
            activated_nodes = independent_cascade(
                graph, seeds=[seed], propagation_prob=propagation_prob
            )
            spread_sizes.append(len(activated_nodes))
        results[seed] = spread_sizes
    return results


# =============================================================================
# 14. Collect Spread Sizes
# =============================================================================

def collect_spread_sizes(results: Dict[int, List[int]]) -> List[int]:
    """
    Aggregate spread sizes from simulation results.

    Parameters:
    - results (Dict[int, List[int]]): Dictionary mapping seed node IDs to spread size lists.

    Returns:
    - List[int]: Combined list of all spread sizes across seed nodes and simulations.
    """
    spread_sizes = []
    for sizes in results.values():
        spread_sizes.extend(sizes)
    return spread_sizes


# =============================================================================
# 15. Perform Statistical Analysis
# =============================================================================

def perform_statistical_analysis(
        bridge_spread_sizes: List[int],
        high_degree_spread_sizes: List[int],
        random_spread_sizes: List[int]
) -> Dict[str, Tuple[Any, Any]]:
    """
    Perform ANOVA and T-tests on spread size data across different node groups.

    Parameters:
    - bridge_spread_sizes (List[int]): Spread sizes from bridge nodes.
    - high_degree_spread_sizes (List[int]): Spread sizes from high-degree nodes not in HCC.
    - random_spread_sizes (List[int]): Spread sizes from random nodes.

    Returns:
    - Dict[str, Tuple[Any, Any]]: Dictionary containing test statistics and p-values.
    """
    stats = {}
    # Ensure each group has at least 2 samples for meaningful statistical tests
    if (len(bridge_spread_sizes) >= 2 and
            len(high_degree_spread_sizes) >= 2 and
            len(random_spread_sizes) >= 2):

        # Perform one-way ANOVA to compare means across all groups
        try:
            F_statistic, p_value = f_oneway(
                bridge_spread_sizes,
                high_degree_spread_sizes,
                random_spread_sizes
            )
            stats["ANOVA"] = (F_statistic, p_value)
        except Exception as e:
            logging.warning(f"ANOVA failed: {e}")
            stats["ANOVA"] = (np.nan, np.nan)

        # Perform independent T-tests between pairs of groups
        try:
            t_statistic1, p_value1 = ttest_ind(
                bridge_spread_sizes,
                high_degree_spread_sizes,
                equal_var=False
            )
            stats["T-test Bridge vs High-Degree"] = (t_statistic1, p_value1)
        except Exception as e:
            logging.warning(f"T-test Bridge vs High-Degree failed: {e}")
            stats["T-test Bridge vs High-Degree"] = (np.nan, np.nan)

        try:
            t_statistic2, p_value2 = ttest_ind(
                bridge_spread_sizes,
                random_spread_sizes,
                equal_var=False
            )
            stats["T-test Bridge vs Random"] = (t_statistic2, p_value2)
        except Exception as e:
            logging.warning(f"T-test Bridge vs Random failed: {e}")
            stats["T-test Bridge vs Random"] = (np.nan, np.nan)

        try:
            t_statistic3, p_value3 = ttest_ind(
                high_degree_spread_sizes,
                random_spread_sizes,
                equal_var=False
            )
            stats["T-test High-Degree vs Random"] = (t_statistic3, p_value3)
        except Exception as e:
            logging.warning(f"T-test High-Degree vs Random failed: {e}")
            stats["T-test High-Degree vs Random"] = (np.nan, np.nan)
    else:
        # Not enough samples to perform statistical tests
        logging.warning("Not enough samples for statistical analysis.")
        stats["ANOVA"] = (np.nan, np.nan)
        stats["T-test Bridge vs High-Degree"] = (np.nan, np.nan)
        stats["T-test Bridge vs Random"] = (np.nan, np.nan)
        stats["T-test High-Degree vs Random"] = (np.nan, np.nan)
    return stats


# =============================================================================
# 16. Get Spread Statistics for Each Group
# =============================================================================

def get_spread_statistics(spread_sizes: List[int]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of spread sizes.

    Parameters:
    - spread_sizes (List[int]): List of spread sizes from simulations.

    Returns:
    - Dict[str, float]: Dictionary containing average, median, standard deviation, minimum, and maximum spread sizes.
    """
    if not spread_sizes:
        return {
            "Average Spread Size": np.nan,
            "Median Spread Size": np.nan,
            "Standard Deviation": np.nan,
            "Minimum Spread Size": np.nan,
            "Maximum Spread Size": np.nan
        }
    return {
        "Average Spread Size": np.mean(spread_sizes),
        "Median Spread Size": np.median(spread_sizes),
        "Standard Deviation": np.std(spread_sizes),
        "Minimum Spread Size": np.min(spread_sizes),
        "Maximum Spread Size": np.max(spread_sizes)
    }


# =============================================================================
# 17. Visualize the Results
# =============================================================================

def visualize_results(
        bridge_spread_sizes: List[int],
        high_degree_spread_sizes: List[int],
        random_spread_sizes: List[int],
        node_id: int
) -> None:
    """
    Generate and save a boxplot visualizing spread size distributions across different node groups.

    Parameters:
    - bridge_spread_sizes (List[int]): Spread sizes from bridge nodes.
    - high_degree_spread_sizes (List[int]): Spread sizes from high-degree nodes not in HCC.
    - random_spread_sizes (List[int]): Spread sizes from random nodes.
    - node_id (int): The current node ID being processed.

    Returns:
    - None
    """
    data = []
    labels = []
    # Append data and labels for each group if data is available
    if bridge_spread_sizes:
        data.append(bridge_spread_sizes)
        labels.append('Bridge Nodes')
    if high_degree_spread_sizes:
        data.append(high_degree_spread_sizes)
        labels.append('High-Degree Nodes Not in HCC')
    if random_spread_sizes:
        data.append(random_spread_sizes)
        labels.append('Random Nodes')

    # If no data is available for any group, skip plotting
    if not data:
        logging.warning(f"No data to plot for Node ID {node_id}. Skipping plot.")
        return

    # Create boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=labels, showfliers=False)  # Updated 'labels' to 'tick_labels'
    plt.ylabel('Spread Size')
    plt.title(f'Spread Size Distribution by Node Group for Node ID {node_id}')
    plt.grid(axis='y')
    plt.tight_layout()
    # Save the plot as a PNG file in the graphs directory
    plt.savefig(os.path.join(GRAPHS_DIR, f"{node_id}_boxplot.png"))
    plt.close()


# =============================================================================
# Function to process a single node_id
# =============================================================================

def process_node_id(node_id: int) -> None:
    """
    Perform the complete analysis pipeline for a single node ID.

    This includes loading data, constructing the graph, calculating metrics,
    simulating information spread, performing statistical analysis, and saving results.

    Parameters:
    - node_id (int): The node ID to process.

    Returns:
    - None
    """
    try:
        # Load edges from the .edges file
        edges_file = os.path.join(DATA_DIR, f"{node_id}.edges")
        if not os.path.isfile(edges_file):
            logging.warning(f"Edges file not found for Node ID {node_id}. Skipping.")
            return
        edges = load_edges(edges_file)
        # Construct the graph with the ego node
        G = construct_graph(edges, ego_node=node_id)

        # Check if the graph is empty
        if G.number_of_nodes() == 0:
            logging.warning(f"Graph for Node ID {node_id} is empty. Skipping.")
            return

        # Calculate node properties
        degrees = dict(G.degree())
        clustering_coeffs = nx.clustering(G)
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)

        # Perform community detection using Louvain algorithm
        partition = community_louvain.best_partition(G)

        # Calculate average clustering coefficient within each community
        community_clustering = calculate_community_clustering(G, partition)

        # Identify highly clustered communities based on network's average clustering
        network_avg_clustering = nx.average_clustering(G)
        highly_clustered_communities = get_highly_clustered_communities(
            community_clustering,
            network_avg_clustering
        )

        # Identify high-degree nodes (top 10%)
        high_degree_nodes = get_high_degree_nodes(degrees, percentile=90)

        if not high_degree_nodes:
            logging.warning(f"No high-degree nodes found for Node ID {node_id}.")

        # Identify bridge nodes: high-degree nodes within highly clustered communities
        bridge_nodes = get_bridge_nodes(
            high_degree_nodes,
            partition,
            highly_clustered_communities
        )

        # Identify high-degree nodes not in highly clustered communities
        high_degree_nodes_not_in_hcc = get_high_degree_nodes_not_in_highly_clustered_communities(
            high_degree_nodes,
            partition,
            highly_clustered_communities
        )

        # Select random nodes equal to the number of bridge nodes
        num_bridge_nodes = len(bridge_nodes)
        if num_bridge_nodes == 0:
            random_nodes = []
            logging.warning(f"No bridge nodes found for Node ID {node_id}. No random nodes will be selected.")
        else:
            random_nodes = get_random_nodes(G, num_bridge_nodes)

        # Simulate information spread using the Independent Cascade Model
        num_simulations = 100  # Number of simulation runs per seed node
        propagation_prob = 0.1  # Probability of propagation per edge

        bridge_results = simulate_spread(
            G,
            bridge_nodes,
            num_simulations=num_simulations,
            propagation_prob=propagation_prob
        ) if bridge_nodes else {}

        high_degree_results = simulate_spread(
            G,
            high_degree_nodes_not_in_hcc,
            num_simulations=num_simulations,
            propagation_prob=propagation_prob
        ) if high_degree_nodes_not_in_hcc else {}

        random_results = simulate_spread(
            G,
            random_nodes,
            num_simulations=num_simulations,
            propagation_prob=propagation_prob
        ) if random_nodes else {}

        # Aggregate spread sizes from simulations
        bridge_spread_sizes = collect_spread_sizes(bridge_results)
        high_degree_spread_sizes = collect_spread_sizes(high_degree_results)
        random_spread_sizes = collect_spread_sizes(random_results)

        # Perform statistical analysis on spread sizes
        stats = perform_statistical_analysis(
            bridge_spread_sizes,
            high_degree_spread_sizes,
            random_spread_sizes
        )

        # Calculate basic spread statistics for each group
        bridge_stats = get_spread_statistics(bridge_spread_sizes)
        high_degree_stats = get_spread_statistics(high_degree_spread_sizes)
        random_stats = get_spread_statistics(random_spread_sizes)

        # Generate and save visualization of spread size distributions
        visualize_results(
            bridge_spread_sizes,
            high_degree_spread_sizes,
            random_spread_sizes,
            node_id
        )

        # Prepare results data for CSV
        results_data = {
            "Group": [],
            "Average Spread Size": [],
            "Median Spread Size": [],
            "Standard Deviation": [],
            "Minimum Spread Size": [],
            "Maximum Spread Size": []
        }

        # Populate results data for Bridge Nodes
        if bridge_spread_sizes:
            results_data["Group"].append("Bridge Nodes")
            results_data["Average Spread Size"].append(bridge_stats["Average Spread Size"])
            results_data["Median Spread Size"].append(bridge_stats["Median Spread Size"])
            results_data["Standard Deviation"].append(bridge_stats["Standard Deviation"])
            results_data["Minimum Spread Size"].append(bridge_stats["Minimum Spread Size"])
            results_data["Maximum Spread Size"].append(bridge_stats["Maximum Spread Size"])
        else:
            results_data["Group"].append("Bridge Nodes")
            results_data["Average Spread Size"].append(np.nan)
            results_data["Median Spread Size"].append(np.nan)
            results_data["Standard Deviation"].append(np.nan)
            results_data["Minimum Spread Size"].append(np.nan)
            results_data["Maximum Spread Size"].append(np.nan)

        # Populate results data for High-Degree Nodes Not in HCC
        if high_degree_spread_sizes:
            results_data["Group"].append("High-Degree Nodes Not in HCC")
            results_data["Average Spread Size"].append(high_degree_stats["Average Spread Size"])
            results_data["Median Spread Size"].append(high_degree_stats["Median Spread Size"])
            results_data["Standard Deviation"].append(high_degree_stats["Standard Deviation"])
            results_data["Minimum Spread Size"].append(high_degree_stats["Minimum Spread Size"])
            results_data["Maximum Spread Size"].append(high_degree_stats["Maximum Spread Size"])
        else:
            results_data["Group"].append("High-Degree Nodes Not in HCC")
            results_data["Average Spread Size"].append(np.nan)
            results_data["Median Spread Size"].append(np.nan)
            results_data["Standard Deviation"].append(np.nan)
            results_data["Minimum Spread Size"].append(np.nan)
            results_data["Maximum Spread Size"].append(np.nan)

        # Populate results data for Random Nodes
        if random_spread_sizes:
            results_data["Group"].append("Random Nodes")
            results_data["Average Spread Size"].append(np.mean(random_spread_sizes))
            results_data["Median Spread Size"].append(np.median(random_spread_sizes))
            results_data["Standard Deviation"].append(np.std(random_spread_sizes))
            results_data["Minimum Spread Size"].append(np.min(random_spread_sizes))
            results_data["Maximum Spread Size"].append(np.max(random_spread_sizes))
        else:
            results_data["Group"].append("Random Nodes")
            results_data["Average Spread Size"].append(np.nan)
            results_data["Median Spread Size"].append(np.nan)
            results_data["Standard Deviation"].append(np.nan)
            results_data["Minimum Spread Size"].append(np.nan)
            results_data["Maximum Spread Size"].append(np.nan)

        # Convert results data to a pandas DataFrame
        results_df = pd.DataFrame(results_data)

        # Prepare statistical test results for CSV
        stats_df = pd.DataFrame([
            {
                "Test": "ANOVA",
                "Statistic": stats["ANOVA"][0],
                "p-value": stats["ANOVA"][1]
            },
            {
                "Test": "T-test Bridge vs High-Degree",
                "Statistic": stats["T-test Bridge vs High-Degree"][0],
                "p-value": stats["T-test Bridge vs High-Degree"][1]
            },
            {
                "Test": "T-test Bridge vs Random",
                "Statistic": stats["T-test Bridge vs Random"][0],
                "p-value": stats["T-test Bridge vs Random"][1]
            },
            {
                "Test": "T-test High-Degree vs Random",
                "Statistic": stats["T-test High-Degree vs Random"][0],
                "p-value": stats["T-test High-Degree vs Random"][1]
            },
        ])

        # Define paths for saving results
        results_csv_path = os.path.join(RESULTS_DIR, f"{node_id}_spread_statistics.csv")
        stats_csv_path = os.path.join(RESULTS_DIR, f"{node_id}_statistical_tests.csv")

        # Save spread statistics and statistical test results to CSV files
        results_df.to_csv(results_csv_path, index=False)
        stats_df.to_csv(stats_csv_path, index=False)

        # Log completion of analysis for the current node ID
        logging.info(f"Completed analysis for Node ID {node_id}")
    except Exception as e:
        logging.error(f"Error processing Node ID {node_id}: {e}")
        print(f"Error processing Node ID {node_id}: {e}")

# =============================================================================
# Main Function to Process All Node IDs
# =============================================================================


def main() -> None:
    """
    Main function to detect all node IDs in the data directory and perform analysis on each.

    This function scans the DATA_DIR for all .edges files, extracts node IDs, and processes each
    node ID by running the analysis pipeline. Results are saved to the RESULTS_DIR and GRAPHS_DIR,
    and logs are recorded in the LOGS_DIR.

    Returns:
    - None
    """
    node_ids = []
    # Iterate over all files in the data directory
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".edges"):
            try:
                # Extract node_id by splitting the filename
                node_id = int(filename.split('.')[0])
                node_ids.append(node_id)
            except ValueError:
                # Log a warning if the filename does not start with an integer node_id
                logging.warning(f"Skipping file with invalid node_id: {filename}")

    # Check if any node_ids were detected
    if not node_ids:
        logging.error("No .edges files found in the data directory.")
        print("No .edges files found in the data directory.")
        return

    # Log and print the detected node IDs
    logging.info(f"Detected Node IDs: {node_ids}")
    print(f"Detected Node IDs: {node_ids}")

    # Process each node_id with a progress bar
    for node_id in tqdm(node_ids, desc="Processing Node IDs"):
        process_node_id(node_id)


# =============================================================================
# Execute the Script
# =============================================================================

if __name__ == "__main__":
    main()
