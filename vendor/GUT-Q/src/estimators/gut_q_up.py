import networkx as nx
from networkx.readwrite import json_graph

from .estimator import Estimator


class GUTQUP(Estimator):
    def __init__(self, aggregation, **kwargs):
        super().__init__(**kwargs)
        if aggregation != "sum":
            raise ValueError("GUT-UP only supports aggregation='sum'")
        self.aggregation = aggregation
        self.alpha = kwargs.get("alpha", 0.2)

    def __str__(self):
        if self.weight:
            return f"GUT-UP_{self.aggregation}_w"
        return f"GUT-UP_{self.aggregation}"

    def __call__(self, g_data):
        u_dict = {}
        for item in g_data:
            G = json_graph.node_link_graph(item["G"])
            u_dict[item["id"]] = self._calculate_g(G, item["answers_id"])
        return u_dict

    def _calculate_g(self, G, answers_id):
        self._gut_up(G)
        return sum(G.nodes[u]['gut_up'] for u in answers_id)

    def _gut_up(self, G):
        for u in nx.topological_sort(G):
            score = self._get_weight(G.nodes[u])
            for v in G.predecessors(u):
                score += self.alpha * G.nodes[v]['gut_up']
            G.nodes[u]['gut_up'] = score
