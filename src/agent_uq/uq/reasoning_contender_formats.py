import json
import re

from agent_uq.uq import topouq_source as topo

ANNOTATION_TEMPLATE = ' \nYou will be provided with a question and a multi-step response containing reasoning steps. \nFor each long reasoning step labeled "Step i:", extract the keywords, only the relevant tokens for that specific reasoning step.\nYou also need to evaluate the importance of each keyword to the final answer. Please evaluate the importance score following with the keyword by (/<importance score>/) on a scale of 1 to 10, where 1 is the least critical and 10 is the most critical.\nIf you find more than one keyword in a specific step, separate them with “;”.\nIf a specific step does not contribute meaningfully to deriving the final answer (e.g., repeating information already provided in the question, introducing irrelevant assumptions or speculations), return "Step i: NO ANSWER" for that step. For example:\nQ: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?\nA: Step 1: Identify the amount of blue fiber needed. The robe requires 2 bolts of blue fiber.  \nStep 2: Determine the amount of white fiber needed. It is half the amount of blue fiber, which is 2 ÷ 2 = 1 bolt.  \nStep 3: Compute the total number of bolts. Add the bolts of blue fiber (2) and white fiber (1) to get 3 bolts.\nFinal Answer: 3\nKeywords for Each Reasoning Step: Step 1: 2 bolts (/3/)\nStep 2: 1 bolt (/10/)\nStep 3: 3 bolts (/7/)\n\nThe following is your task:\nQ: <QUESTION>\nA: <RESPONSE>\nKeywords for Each Reasoning Step:\n'

_STEP = re.compile(r'^\s*(?:[-*]\s*)?Step\s+(\d+)\s*[:;]\s*', re.M | re.I)
_SCORE = re.compile(
    r'(?P<keyword>.*?)\s*(?P<score>'
    r'\(\s*/?\s*<?\s*(?:importance\s+score\s*:?\s*)?\d+\s*>?\s*/?\s*\)'
    r'|/\s*\(\s*\d+\s*\)\s*/|/\s*\d+\s*/)\s*[.,]?\s*$', re.I)


def _parse_keyword_block(text):
    text = text.replace('**', '').replace('```', '')
    matches = list(_STEP.finditer(text))
    if not matches or [int(m.group(1)) for m in matches] != list(range(1, len(matches) + 1)):
        return False
    keywords, scores, normalized = [], [], []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = re.sub(r'\s*\n\s*', ' ', text[match.end():end]).strip()
        body = re.sub(r'^Keywords(?:\s+for\s+Step\s+\d+)?\s*:\s*', '', body, flags=re.I)
        words, weights = [], []
        for piece in (p.strip() for p in body.split(';') if p.strip()):
            score = _SCORE.fullmatch(piece)
            if score is None:
                return False
            word = score.group('keyword').strip().strip('"\'`')
            weight = int(re.search(r'\d+', score.group('score')).group())
            if not word or not 1 <= weight <= 10 or re.search(r'\(/|/\)', word):
                return False
            words.append(word)
            weights.append(weight)
        if not words:
            return False
        keywords.append(words)
        scores.append(weights)
        normalized.append(f'Step {index + 1}: ' + '; '.join(
            f'{word}(/{weight}/)' for word, weight in zip(words, weights)))
    return '\n'.join(normalized), keywords, scores


def parse_confidence(text):
    text = text.replace('**', '').replace('```', '')
    fields = list(re.finditer(
        r'(?:Confidence(?:\s+score)?|Probability)\s*[:=]\s*\[?\s*'
        r'(\d+(?:\.\d+)?)\s*%?\s*\]?\s*%?', text, flags=re.I))
    match = fields[-1] if fields else re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*%?\s*', text)
    if match is None:
        return None
    value = float(match.group(1))
    if '%' in match.group() or value > 1:
        value /= 100
    return value if 0 <= value <= 1 else None


def parse_keywords(text):
    text = text.replace('**', '').replace('```', '')
    text = re.sub(r'Keywords\s+for\s+Each\s+Reasoning\s+Step\s*:\s*', '', text, flags=re.I)
    text = re.sub(r';\s*(?=Step\s+\d+\s*:)', '\n', text, flags=re.I)
    matches = list(_STEP.finditer(text))
    steps = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        parsed = _parse_keyword_block('Step 1: ' + text[match.end():end])
        if not parsed:
            return False
        number = int(match.group(1))
        values = parsed[1][0], parsed[2][0]
        if number in steps and steps[number] != values:
            return False
        steps[number] = values
    normalized = '\n'.join(f'Step {number}: ' + '; '.join(
        f'{word}(/{weight}/)' for word, weight in zip(*steps[number])) for number in sorted(steps))
    return _parse_keyword_block(normalized)

def canonicalize_graph(text, qa_pairs):

    for marker in ('**', '__', '`'):
        text = re.sub(re.escape(marker) + r'((?:Structure|ResultEdge|EdgeResult|ResultNode)\s*:?)'
                      + re.escape(marker), r'\1', text, flags=re.IGNORECASE)
    source_structure, result_edge = topo.extract_structure_and_edge(text)
    if not result_edge or not result_edge.split(':', 1)[1].strip():
        raise ValueError('explicit conclusion missing')

    graph_text = source_structure or text
    if not source_structure:
        for block in re.findall(r'```[^\n]*\n(.*?)```', text, flags=re.DOTALL):
            if re.search(r'\[\s*"?Node(?:Raw|Result|\d+)"?\s*,', block):
                graph_text = block
                break
    triples = []
    for bracket in re.findall(r'\[([^\[\]]*)\]', graph_text):
        parts = [part.strip() for part in bracket.split(',')]
        if len(parts) != 3:
            raise ValueError('malformed bracket triple')
        for index, part in enumerate(parts):
            if part.startswith('"'):
                try:
                    parts[index] = json.loads(part)
                except ValueError as error:
                    raise ValueError('invalid quoted identifier') from error
        triples.append(parts)
    if not triples:
        raise ValueError('explicit bracket triples missing')

    nodes = {f'Node{i}' for i in range(len(qa_pairs))}
    declarations = re.findall(r'(?im)^\s*(?:[-*]\s*)?ResultNode\s*:\s*([^\n]+)', text)
    terminals = set()
    for declaration in declarations:
        match = re.fullmatch(r'(NodeResult|Node\d+)(?:\s+\([^()\n]*\))?\s*', declaration)
        if not match:
            raise ValueError('ambiguous terminal declaration')
        terminals.add(match.group(1))
    if len(terminals) > 1:
        raise ValueError('conflicting terminal declarations')
    alias = next(iter(terminals), 'NodeResult')
    if alias != 'NodeResult':
        if (alias in nodes or any(src == alias for src, _, _ in triples)
                or any('NodeResult' in (src, dst) for src, dst, _ in triples)
                or not any(dst == alias and edge == 'ResultEdge' for _, dst, edge in triples)):
            raise ValueError('unsafe terminal alias')
        triples = [[src, 'NodeResult' if dst == alias else dst, edge] for src, dst, edge in triples]

    for src, dst, edge in triples:
        if (not re.fullmatch(r'Node(?:Raw|Result|\d+)', src)
                or not re.fullmatch(r'Node(?:Raw|Result|\d+)', dst)
                or not re.fullmatch(r'(?:Edge\d+|ResultEdge)', edge)):
            raise ValueError('unknown node or edge identifier')
        if edge == 'ResultEdge' and dst != 'NodeResult':
            raise ValueError('ResultEdge must target terminal')

    reachable = {'NodeRaw'}
    while True:
        added = {dst for src, dst, _ in triples if src in reachable and src != 'NodeResult'} - reachable
        if not added:
            break
        reachable.update(added)
    if not any(src in reachable and src != 'NodeResult' and dst == 'NodeResult' and edge == 'ResultEdge'
               for src, dst, edge in triples):
        raise ValueError('directed root-to-terminal path missing')
    structure = 'Structure: {' + ', '.join('[' + ', '.join(parts) + ']' for parts in triples) + '}'
    return structure, result_edge


def require_structured_turn(turn):
    for name in ('task_question', 'observation_text'):
        if not isinstance(turn.get(name), str) or (name == 'task_question' and not turn[name].strip()):
            raise ValueError(f'prepared turn requires structured {name}')


def build_topo_input(turn):
    require_structured_turn(turn)
    return (f'Dataset question: {turn["task_question"]}\n\nCurrent turn:\n'
            f'Thought: {turn["thought_text"]}\nAction: {turn["action_text"]}\n'
            f'Observation: {turn["observation_text"]}')


def graph_format_clause(qa_count):
    nodes = ', '.join(f'Node{i}' for i in range(qa_count))
    edges = ', '.join(f'Edge{i}' for i in range(qa_count))
    return ('\n\nOutput format requirements (use the same graph representation as the example):\n'
            'Use NodeRaw as the initial question node and NodeResult as the separate conclusion node.\n'
            f'The supplied Q-A pairs define only these internal nodes: {nodes}; and these edges: {edges}.'
            'Write each directed connection as [source_node, target_node, edge_label].'
            'Include the connections that form a directed path from NodeRaw to NodeResult.'
            ' Use ResultEdge for the final connection into NodeResult.\n'
            'Output only Structure: {the connection triples}; ResultEdge: the conclusion sentence. ')


def graph_row(text, qa_pairs):


    structure, edge = topo.extract_structure_and_edge(text)
    try:
        structure, edge = canonicalize_graph(text, qa_pairs)
    except ValueError:
        structure = structure or text
    return dict(structure_raw=structure, response_str=edge,
                string_q_a=topo.dict_to_pairs_str(qa_pairs))
