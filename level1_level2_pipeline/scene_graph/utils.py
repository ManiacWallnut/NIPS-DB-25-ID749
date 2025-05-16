import networkx as nx
import matplotlib.pyplot as plt


# Function to build a graph from the Tree structure
def build_graph_from_tree(tree_node, graph=None, parent_name=None):
    if graph is None:
        graph = nx.DiGraph()  # Create a directed graph

    # Add the current node to the graph
    graph.add_node(tree_node.name)

    # If the node has a parent, add an edge between the parent and the current node
    if parent_name:
        graph.add_edge(parent_name, tree_node.name)

    # Recursively add children to the graph
    for child in tree_node.children:
        build_graph_from_tree(child, graph, tree_node.name)

    return graph


visited_node = set()


def show_tree_like_dir(tree_node, depth=0, file=None):
    if tree_node.name in visited_node:
        return
    if depth == 0:
        print(f"root:[{tree_node.name}]", file=file)

    for child in tree_node.children:
        print("|       " * (1 + depth) + "+--" + child.name, file=file)
        show_tree_like_dir(child, depth + 1, file=file)
    visited_node.add(tree_node.name)


# Function to visualize the tree structure
def visualize_tree(tree, file=None):

    for root_node in tree.nodes.values():
        # Only plot the root nodes (those without a parent)
        show_tree_like_dir(root_node, 0, file=file)

    for root_node in tree.nodes.values():
        if (
            root_node.parent is None
        ):  # Only plot the root nodes (those without a parent)
            G = build_graph_from_tree(root_node)  # Build the graph for the root node

            # Positioning the nodes in a hierarchical layout
            pos = nx.spring_layout(G, k=0.5, iterations=50)

            # Draw the nodes and edges
            plt.figure(figsize=(12, 8))
            nx.draw(
                G,
                pos,
                with_labels=True,
                node_size=3000,
                node_color="skyblue",
                font_size=10,
                font_weight="bold",
                arrows=True,
            )
            plt.title(f"Tree Visualization for Root: {root_node.name}")
            plt.show()
