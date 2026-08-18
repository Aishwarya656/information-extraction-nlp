import networkx as nx
from pyvis.network import Network

def generate_knowledge_graph(relations):

    graph = nx.DiGraph()

    for subject, relation, obj in relations:

        graph.add_node(subject)

        graph.add_node(obj)

        graph.add_edge(subject, obj, title=relation, label=relation)

    network = Network(
        height="700px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True
    )

    network.from_nx(graph)

    network.repulsion(
        node_distance=220,
        spring_length=200
    )

    network.save_graph("static/knowledge_graph.html")