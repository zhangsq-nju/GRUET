from __future__ import annotations
import re
from collections import defaultdict
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment

ANSWER_FORMAT_EXAMPLE = "\nExample1 \n\nQ: Where is Australia located on Earth?\nA: Australia is in the Southern Hemisphere, southeast of Asia, between the Indian and Pacific Oceans.\n\nExample2\nQ: How does the tilt of the Earth affect seasons?\nA: Earth's 23.5° axial tilt causes different hemispheres to receive varying amounts of sunlight throughout the year, resulting in opposite seasons between the Northern and Southern Hemispheres.\n"

KNOWLEDGE_FORMAT_EXAMPLE = "\nExample1\n\nQuestion: If it is currently summer in Australia, what season is it in Canada?\nExpected Response (For required knowledge): \n1. Where is Australia located on Earth?\n2. Where is Canada located on Earth?\n3. What is the geographical relationship between Australia and Canada?\n4. How does the tilt of the Earth affect seasons?\n\nExample2\n\nQuestion: If a lunar eclipse is occurring at midnight on the vernal equinox in Greenwich, England, what date and time is it in Tokyo, Japan, is the eclipse visible there, and what season is it in Sydney, Australia?\nExpected Response (For required knowledge): \n1. What is the time difference between Greenwich, England (GMT), and Tokyo, Japan?\n2. When it's midnight on the vernal equinox in Greenwich, what is the local date and time in Tokyo?\n3. Is a lunar eclipse visible from Tokyo at that specific date and time?\n4. What is the significance of the vernal equinox in terms of seasons?\n5. How do seasons differ between the Northern and Southern Hemispheres?\n6. What season is it in Sydney, Australia, during the vernal equinox?\n7. What is the geographical location of Sydney, Australia, in relation to hemispheres?\n8. How does Earth's tilt and orbit affect the occurrence of seasons?\n9. What are the conditions required for a lunar eclipse to be visible from a particular location?\n10. Does the International Date Line affect the date when moving from Greenwich to Tokyo?\n"

REASON_PATH_EXAMPLE = "\nExample1\n\nQuestion: \nIf it is currently summer in Australia, what season is it in Canada?\n\nDict: \nEdeg0: Where is Australia located on Earth?, Node0: Australia in the Southern Hemisphere.;\nEdge1:Where is Canada located on Earth?, Node1: Canada is located in the Northern Hemisphere.;\nEdge2: What is the geographical relationship between Australia and Canada?, Node2: Australia and Canada are in the opposite hemisphere.;\nEdge3: How does the tilt of the Earth affect seasons?, Node3: Opposite hemispheres experience opposite seasons beacuse of the Earth's tilt\n\nYou should reason first, and then get the final edge, which is a conclusion sentence: In this example E.g., It is summer in Canada. Then there should be an added: ResultNode and ResultEdge as: [Node xxx, NodeResult, ResultEdge], replace xxx with actual number.\n\nAn example output: \nStructure: {[NodeRaw, Node0, Edge0], [NodeRaw, Node1, Edge1], [Node0, Node2, Edge2], [Node1, Node2, Edge2], [Node2, Node3, Edge3], [Node3, NodeResult, ResultEdge]}; ResultEdge: It is summer in Canada.\n"

STRUCT_RE = re.compile('Structure:\\s*\\{.*?\\}', re.DOTALL | re.IGNORECASE)

EDGE_RESULT_RE = re.compile('\\b(?:EdgeResult|ResultEdge):\\s*(.*?)(?:\\n|\\.|$)', re.IGNORECASE | re.DOTALL)

def extract_structure_and_edge(text: str) -> Tuple[str, str]:
    text = text.strip()
    m1 = STRUCT_RE.search(text)
    structure = m1.group(0) if m1 else ''
    m2 = EDGE_RESULT_RE.search(text)
    edge = f'EdgeResult: {m2.group(1).strip()}' if m2 else ''
    return (structure, edge)

def extract_numbered_questions(text: str) -> List[str]:
    return re.findall('^\\d+\\.\\s*(.+)', text, flags=re.MULTILINE)

def add_qa_from_block(dst: Dict[str, str], qa_block: str) -> Dict[str, str]:
    q, a = ('', '')
    for line in qa_block.splitlines():
        if line.startswith('Q:'):
            q = line[2:].strip()
        elif line.startswith('A:'):
            a = line[2:].strip()
    if q and a:
        dst[q] = a
    return dst

def dict_to_pairs_str(d: Dict[str, str]) -> str:
    lines = []
    for i, (q, a) in enumerate(d.items()):
        lines.append(f'Edge{i}: {q}, Node{i}: {a};')
    return '\n'.join(lines)

def get_knowledge_points(engine: Engine, question: str, example: str) -> str:
    system_prompt = 'You are a helpful assistant to do the following: Given a question, you should reflect and come up with the sufficient knowledge that you need to solve this question. Two standards: sufficient and concise. And you should respond with numbered points.'
    user_prompt = f'Given a question: {question} Please give response by learning the format from the example: {example}.'
    return engine.chat(system_prompt, user_prompt)

def get_knowledge_answers(engine: Engine, question: str, example: str) -> str:
    system_prompt = 'You are an answering assistant to provide concise answers to a question based on your knowledge.'
    user_prompt = f'Given a question: {question} Please give response following the format in the example: {example}.'
    return engine.chat(system_prompt, user_prompt)

def get_reasoning_structure(engine: Engine, qa_pairs: Dict[str, str], question: str) -> Tuple[str, str]:
    system_prompt = 'You are a reasoning assistant. Given some Edge-Node pairs (representing Q-A pairs), find a reasoning path that solves the given question. Your response should follow a structured format.'
    pair_str = dict_to_pairs_str(qa_pairs)
    user_prompt = 'Given a question: ' + question + ' Please learn how it is reasoned from the example: ' + REASON_PATH_EXAMPLE + '. Now give the reasoning path for ' + pair_str
    resp = engine.chat(system_prompt, user_prompt)
    structure, edge = extract_structure_and_edge(resp)
    if not structure:
        structure = resp
    return (structure, edge)

def extract_structure_paths(structure_str):
    pattern = '\\[([^\\]]+)\\]'
    matches = re.findall(pattern, structure_str)
    edges = set()
    nodes = set()
    for match in matches:
        parts = match.split(', ')
        if len(parts) != 3:
            continue
        source = parts[0].strip()
        target = parts[1].strip()
        edge_label = parts[2].strip()
        nodes.add(source)
        nodes.add(target)
        edges.add((source, target, edge_label))
    graph = defaultdict(list)
    for src, tgt, edge in edges:
        graph[src].append((tgt, edge))

    if 'NodeRaw' not in nodes or 'NodeResult' not in nodes:
        return 'Structure: {}'
    reachability_graph = nx.DiGraph((src, tgt) for src, tgt, _ in edges)
    if not nx.has_path(reachability_graph, 'NodeRaw', 'NodeResult'):
        return 'Structure: {}'

    def find_paths(current_node, target, path, edge_path, all_paths):
        if current_node == target:
            all_paths.append((list(path), list(edge_path)))
            return
        for neighbor, edge in graph.get(current_node, []):
            if neighbor not in path:
                path.append(neighbor)
                edge_path.append(edge)
                find_paths(neighbor, target, path, edge_path, all_paths)
                path.pop()
                edge_path.pop()
    all_paths = []
    if 'NodeRaw' in graph:
        find_paths('NodeRaw', 'NodeResult', ['NodeRaw'], [], all_paths)
    if not all_paths:
        return 'Structure: {}'
    unique_edges = set()
    for path, edge_path in all_paths:
        for i in range(len(path) - 1):
            unique_edges.add((path[i], path[i + 1], edge_path[i]))
    formatted_edges = [f'[{src}, {tgt}, {edge}]' for src, tgt, edge in unique_edges]
    return 'Structure: {' + ', '.join(formatted_edges) + '}'

def parse_structure(structure_str):
    import re
    items = re.findall('\\[([^]]+)\\]', structure_str)
    parsed_items = []
    nodes = set()
    edges = []
    for item in items:
        parts = [x.strip() for x in item.split(',')]
        if len(parts) == 3:
            node1, node2, edge = parts
            edges.append(('edge', node1, node2, edge))
            nodes.add(node1)
            nodes.add(node2)
        else:
            raise ValueError(f'Invalid format in structure_str: {item}')
    for node in nodes:
        parsed_items.append(('node', node))
    parsed_items.extend(edges)
    return parsed_items

def build_graph(structure_str, embeddings):
    G = nx.Graph()
    parsed_items = parse_structure(structure_str)
    for item in parsed_items:
        if item[0] == 'edge':
            node1, node2, edge = (item[1], item[2], item[3])
            if edge in embeddings:
                G.add_edge(node1, node2, embedding=embeddings[edge])
            else:
                raise KeyError(f"Edge '{edge}' not found in embeddings.")
        elif item[0] == 'node':
            node = item[1]
            if node in embeddings:
                G.add_node(node, embedding=embeddings[node])
            else:
                raise KeyError(f"Node '{node}' not found in embeddings.")
    return G

def average_cosine_similarity(embedding, embeddings_list):
    similarities = [1 - cosine(embedding, emb) for emb in embeddings_list]
    return np.mean(similarities)

class HausdorffEditDistance:

    def __init__(self):
        pass

    def node_deletion(self, G, node):
        node_embedding = G.nodes[node]['embedding']
        other_embeddings = [G.nodes[n]['embedding'] for n in G.nodes if n != node]
        return 1 - average_cosine_similarity(node_embedding, other_embeddings)

    def node_insertion(self, G, node):
        node_embedding = G.nodes[node]['embedding']
        other_embeddings = [G.nodes[n]['embedding'] for n in G.nodes]
        return 1 - average_cosine_similarity(node_embedding, other_embeddings)

    def node_substitution(self, G1, G2, node1, node2):
        embedding1 = G1.nodes[node1]['embedding']
        embedding2 = G2.nodes[node2]['embedding']
        return 1 - (1 - cosine(embedding1, embedding2))

    def edge_deletion(self, G, edge):
        edge_embedding = G.edges[edge]['embedding']
        other_embeddings = [G.edges[e]['embedding'] for e in G.edges if e != edge]
        return 1 - average_cosine_similarity(edge_embedding, other_embeddings)

    def edge_insertion(self, G, edge):
        edge_embedding = G.edges[edge]['embedding']
        other_embeddings = [G.edges[e]['embedding'] for e in G.edges]
        return 1 - average_cosine_similarity(edge_embedding, other_embeddings)

    def edge_substitution(self, G1, G2, edge1, edge2):
        embedding1 = G1.edges[edge1]['embedding']
        embedding2 = G2.edges[edge2]['embedding']
        return 1 - (1 - cosine(embedding1, embedding2))

    def ged(self, G1, G2):
        nodes1 = list(G1.nodes)
        nodes2 = list(G2.nodes)
        edges1 = list(G1.edges)
        edges2 = list(G2.edges)
        node_cost_matrix = np.zeros((len(nodes1), len(nodes2)))
        for i, node1 in enumerate(nodes1):
            for j, node2 in enumerate(nodes2):
                node_cost_matrix[i, j] = self.node_substitution(G1, G2, node1, node2)
        edge_cost_matrix = np.zeros((len(edges1), len(edges2)))
        for i, edge1 in enumerate(edges1):
            for j, edge2 in enumerate(edges2):
                edge_cost_matrix[i, j] = self.edge_substitution(G1, G2, edge1, edge2)
        row_ind, col_ind = linear_sum_assignment(node_cost_matrix)
        node_cost = node_cost_matrix[row_ind, col_ind].sum()
        row_ind, col_ind = linear_sum_assignment(edge_cost_matrix)
        edge_cost = edge_cost_matrix[row_ind, col_ind].sum()
        delete_cost = sum((self.node_deletion(G1, node) for node in nodes1)) + sum((self.edge_deletion(G1, edge) for edge in edges1))
        insert_cost = sum((self.node_insertion(G2, node) for node in nodes2)) + sum((self.edge_insertion(G2, edge) for edge in edges2))
        total_cost = node_cost + edge_cost + delete_cost + insert_cost
        similarity = 1 / (1 + total_cost)
        return similarity


def source_embedding_record(row, embed):
    structure_str = row['structure_raw']
    record = {'embeddings': {}, 'structure_str': structure_str,
              'sub_structure_str': extract_structure_paths(structure_str)}
    for line in row['string_q_a'].split(';'):
        if ':' in line:
            key, value = line.split(':', 1)
            if value.strip():
                record['embeddings'][key.strip()] = embed(value.strip())
    for pair in structure_str.split(','):
        elements = pair.replace('{', '').replace('}', '').replace('[', '')\
                       .replace(']', '').split()
        for element in elements:
            if element.startswith('Node') or element.startswith('Edge'):
                if element not in record['embeddings']:
                    record['embeddings'][element] = embed(element)
    if row['response_str']:
        record['embeddings']['ResultEdge'] = embed(row['response_str'])
    return record


def source_topouq(rows, embed):
    graphs, invalid = [], []
    for sample_index, row in enumerate(rows):
        try:
            record = source_embedding_record(row, embed)
            graph = build_graph(record['sub_structure_str'], record['embeddings'])
            if not graph.number_of_nodes() or not graph.number_of_edges():
                raise ValueError('no_valid_raw_to_result_path')
            graphs.append((sample_index, graph))
        except (ValueError, KeyError) as error:
            invalid.append({'sample_index': sample_index, 'reason': str(error)})
    diagnostics = {'attempted_samples': len(rows), 'valid_samples': len(graphs),
                   'invalid_samples': invalid, 'pairs': []}
    if len(graphs) < 2:
        return float('nan'), {**diagnostics, 'reason': 'insufficient_valid_graphs'}
    distances = []
    scorer = HausdorffEditDistance()
    for left in range(len(graphs)):
        for right in range(left + 1, len(graphs)):
            distance = float(1 - scorer.ged(graphs[left][1], graphs[right][1]))
            distances.append(distance)
            diagnostics['pairs'].append({'left': graphs[left][0],
                                          'right': graphs[right][0],
                                          'distance': distance})
    if not np.isfinite(distances).all():
        return float('nan'), {**diagnostics, 'reason': 'source_nonfinite_distance'}
    return float(np.var(distances)), diagnostics
