import json

import networkx as nx
import numpy as np
from math_verify import parse, verify
from networkx.readwrite import json_graph
from tqdm import tqdm

from src.utils.nli import NLIScorer


class GraphConstructor:
    def __init__(self):
        self.scorer = NLIScorer()

    def __call__(self, responses_data, output_dir, merge_way='aov', sim_way="2entail"):
        if merge_way != 'aov':
            raise ValueError("Only merge_way='aov' is supported")
        self.sim_way = sim_way
        g_data = []
        with open(f"{output_dir}/{merge_way}_{sim_way}/g_data.jsonl", "w", encoding="utf-8") as f:
            for item in tqdm(responses_data):
                G, answers_id = self._construct(item["responses"])
                g_item = {
                    "id": item["id"],
                    "G": json_graph.node_link_data(G),
                    "answers_id": answers_id,
                }
                g_data.append(g_item)
                f.write(json.dumps(g_item, ensure_ascii=False) + "\n")
        return g_data

    def _construct(self, responses):
        G = nx.DiGraph()
        G.add_node(0, num_nodes=1)
        answers = []
        node_id = 1
        for i, response in enumerate(responses):
            cot = response["step-wise info"]
            nodeset = set(G.nodes) - {a_id for _, a_id in answers} - {0}
            prev_id = 0

            nodelist = list(nodeset)
            text_list = [G.nodes[u]['text'] for u in nodelist]
            newtext_list = [step['text'] for step in cot]
            flags = self._node_sim2(text_list, newtext_list)
            step_u_list = [
                [nodelist[j] for j in range(len(nodelist)) if flags[h][j]]
                for h in range(len(cot))
            ]
            for h, step in enumerate(cot):
                u = next((u for u in step_u_list[h] if u in nodeset), None)
                if u is not None:
                    nodeset -= nx.ancestors(G, u)
                    nodeset -= {u}
                    if prev_id != u:
                        G.add_edge(prev_id, u, chain=i)
                        G.nodes[u]['prob_info_all'].append(step['prob info'])
                        G.nodes[u]['num_nodes'] += 1
                    prev_id = u
                else:
                    G.add_node(node_id, text=step['text'], num_nodes=1, prob_info_all=[step['prob info']])
                    G.add_edge(prev_id, node_id, chain=i)
                    prev_id = node_id
                node_id += 1

            answer = response["answer info"]
            answer_parse = parse(answer["text"])
            for parsed_answer, answer_id in answers:
                if verify(parsed_answer, answer_parse):
                    G.add_edge(prev_id, answer_id, chain=i)
                    G.nodes[answer_id]['prob_info_all'].append(answer['prob info'])
                    G.nodes[answer_id]['num_nodes'] += 1
                    break
            else:
                G.add_node(node_id, text=answer['text'], num_nodes=1, prob_info_all=[answer['prob info']])
                answers.append((answer_parse, node_id))
                G.add_edge(prev_id, node_id, chain=i)
            node_id += 1
        return G, [a_id for _, a_id in answers]

    def _node_sim2(self, text_list, newtext_list):
        flags = self.scorer.predict(
            text_list * len(newtext_list),
            sum([[text] * len(text_list) for text in newtext_list], []),
            judge=self.sim_way,
        )
        return np.array(flags).reshape(len(newtext_list), len(text_list))
