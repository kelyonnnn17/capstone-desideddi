import networkx as nx

def build_reasoning_graph(drug_a, drug_b, predictions, genes, rag_docs):
    G = nx.DiGraph()

    G.add_node(drug_a, type="drug")
    G.add_node(drug_b, type="drug")

    for pred in predictions[:5]:
        G.add_node(pred['name'], type="side_effect")
        G.add_edge(drug_a, pred['name'])
        G.add_edge(drug_b, pred['name'])

    for gene in genes[:10]:
        G.add_node(gene['gene_name'], type="gene")

    for i, doc in enumerate(rag_docs):
        G.add_node(f"Doc{i}", type="evidence")

    return G