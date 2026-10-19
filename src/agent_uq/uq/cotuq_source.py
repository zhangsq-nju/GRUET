from __future__ import annotations
import math
import re
import torch

def is_effectively_empty(obj):
    if obj is None:
        return True
    if isinstance(obj, (int, float)) and obj == 0:
        return True
    if obj == '':
        return True
    if isinstance(obj, list):
        return all((is_effectively_empty(item) for item in obj))
    if isinstance(obj, dict):
        if len(obj) == 0:
            return True
        return all((is_effectively_empty(value) for value in obj.values()))
    return False

def clean_words(word):
    return word.replace(' ', '').replace('.', '').replace('"', '').replace('\n', '').replace('_', '').replace('Ġ', '').lower()

def find_subsequence_position(sub_sequence, long_sequence):
    len_long = long_sequence.size(0)
    len_sub = len(sub_sequence)
    sub_sequence_tensor = torch.tensor(sub_sequence, device=long_sequence.device)
    for i in range(len_long - len_sub + 1):
        if torch.equal(long_sequence[i:i + len_sub], sub_sequence_tensor):
            return i
    return -1

def step_exacts_2_list(response):
    lines = response.splitlines()
    lines = [line for line in lines if line.strip()]
    keywords_by_step = []
    contributions_by_step = []
    valid_response_text = []
    for line in lines:
        match = re.search('Step \\d+: (.+)', line)
        if match:
            if '(/' not in line or '/)' not in line:
                continue
            keywords_w_contribution = match.group(1).split('; ')
            if any(('(/' not in key_w_c or '/)' not in key_w_c for key_w_c in keywords_w_contribution)):
                continue
            try:
                keywords = [key_w_c.split('(/')[0].strip() for key_w_c in keywords_w_contribution]
                contributions = [int(key_w_c.split('(/')[1].split('/)')[0].strip()) for key_w_c in keywords_w_contribution]
            except ValueError:
                return False
            for i in contributions:
                if i > 10:
                    return False
            keywords_by_step.append(keywords)
            contributions_by_step.append(contributions)
            valid_response_text.append(line)
    if not valid_response_text:
        return False
    return ('\n'.join(valid_response_text), keywords_by_step, contributions_by_step)

def parse_response_to_dict(response):
    steps = {}
    final_answer = None
    match = re.search('Final Answer:\\s*(.+?)\\s*(?=(\\n|$))', response, re.DOTALL)
    if match:
        final_answer = match.group(1).strip()
        response_before_final_answer = response[:match.end()].strip()
    else:
        return (None, None, None)
    matches = list(re.finditer('(Step \\d+):', response_before_final_answer))
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(response_before_final_answer)
        segment = response[start:end].strip()
        steps[match.group(1)] = segment
    return_response = response_before_final_answer
    return (final_answer, steps, return_response)

def find_token_indices(tokens, word):
    word_len = len(word.replace(' ', ''))
    for start_index in range(len(tokens)):
        combined_text = ''
        end_index = start_index
        while end_index < len(tokens) and len(combined_text) < word_len:
            combined_text += tokens[end_index]
            if clean_words(combined_text) == clean_words(word):
                return (start_index, end_index)
            end_index += 1
    return (-1, -1)

def is_word_in_sentence(sentence, word):
    pattern = re.escape(word)
    match = re.search(pattern, sentence, re.IGNORECASE)
    return True if match else False

def extract_p_mean(keyword_token_probability, contribution_scores=None):
    if contribution_scores == None:
        return_dict = {}
        for step, inner_dict in keyword_token_probability.items():
            for key, values in inner_dict.items():
                if len(values) == 0:
                    continue
                value_to_add = sum(values) / len(values)
                if key in return_dict:
                    return_dict[key].append(value_to_add)
                else:
                    return_dict[key] = [value_to_add]
        return return_dict
    else:
        return_keyword_dict = {}
        return_contribution_dict = {}
        for step, inner_dict in keyword_token_probability.items():
            for key, values in inner_dict.items():
                if len(values) == 0:
                    continue
                value_to_add = sum(values) / len(values)
                if key in return_keyword_dict:
                    return_keyword_dict[key].append(value_to_add)
                    return_contribution_dict[key].append(contribution_scores[step][key])
                else:
                    return_keyword_dict[key] = [value_to_add]
                    return_contribution_dict[key] = [contribution_scores[step][key]]
        return (return_keyword_dict, return_contribution_dict)

def extract_keystep(llm_response, contribution_scores=None):
    if contribution_scores == None:
        return ''
    else:
        step_avg_cons = []
        for step, inner_dict in contribution_scores.items():
            con_list = []
            for key, value in inner_dict.items():
                if value == None:
                    print('##### Contribution Score has None Value #####')
                    continue
                con_list.append(value)
            if len(con_list) == 0:
                avg_con = 0
            else:
                avg_con = sum(con_list) / len(con_list)
            step_avg_cons.append(avg_con)
        max_con = max(step_avg_cons)
        step_idx = len(step_avg_cons) - 1 - step_avg_cons[::-1].index(max_con)
        steps = re.split('\\s*Step \\d+:\\s*', llm_response)
        try:
            return_text = steps[step_idx + 1]
            return_text = re.split('\\s*Final Answer', return_text, maxsplit=1, flags=re.IGNORECASE)[0]
        except IndexError:
            print(f'Warning: Could not extract step {step_idx + 1} from response. The format might be unexpected.')
            return ''
        return return_text.strip()

def weighted_sum(values):
    if len(values) == 1:
        return values[0]
    weights = [math.exp(-c) for c in values]
    sum_weights = sum(weights)
    normalized_weights = [w / sum_weights for w in weights]
    result = sum((w * c for w, c in zip(normalized_weights, values)))
    return result

def extract_probing_confidence(response):
    match = re.search('(\\d+(\\.\\d+)?)%', response)
    if match:
        return float(match.group(1)) / 100
    first_line = response.strip().split('\n')[0]
    try:
        confidence = float(first_line)
        if confidence > 1:
            confidence /= 100
        return confidence
    except ValueError:
        pass
    match = re.search('(\\d+(\\.\\d+)?)', first_line)
    if match:
        confidence = float(match.group(1))
        if confidence > 1:
            confidence /= 100
        return confidence
    return None

instruction = ' \nYou will be provided with a question and a multi-step response containing reasoning steps. \nFor each long reasoning step labeled "Step i:", extract the keywords, only the relevant tokens for that specific reasoning step.\nYou also need to evaluate the importance of each keyword to the final answer. Please evaluate the importance score following with the keyword by (/<importance score>/) on a scale of 1 to 10, where 1 is the least critical and 10 is the most critical.\nIf you find more than one keyword in a specific step, separate them with “;”.\nIf a specific step does not contribute meaningfully to deriving the final answer (e.g., repeating information already provided in the question, introducing irrelevant assumptions or speculations), return "Step i: NO ANSWER" for that step. For example:\n\nQuestion:\n<QUESTION>\nMulti-Step Response:\n<RESPONSE>\nKeywords for Each Reasoning Step:\n'


def mean_confidence(keyword_probabilities, contribution_scores):
    probabilities, contribution_lists = extract_p_mean(
        keyword_probabilities, contribution_scores)
    probabilities = {key: weighted_sum(values)
                     for key, values in probabilities.items()}
    contributions = {key: sum(values) / len(values)
                     for key, values in contribution_lists.items()}
    if not probabilities:
        return float('nan')
    total_weight = sum(contributions.values())
    if total_weight == 0:
        return sum(probabilities.values()) / len(probabilities)
    return sum(probabilities[key] * contributions[key]
               for key in probabilities) / total_weight


def format_stored_cot(thought, action):
    if not re.search(r'Step \d+:', thought):
        raise ValueError('accepted_thought_missing_source_step_markers')
    return thought + '\nFinal Answer: ' + action


def keystep_prompt(question, action, formatted_cot, contribution_scores):
    key_step = extract_keystep(formatted_cot, contribution_scores)
    return (
        f'Question: {question}\nPossible Answer: {action}\n'
        f'The most critical step in reasoning to the possible answer: {key_step}\n'
        'Q: Considering this critical reasoning step as additional information, '
        'how likely is the above answer to be correct? Please first show your '
        'reasoning concisely and then answer with the following format:\n'
        f'```Confidence: [the probability of answer {action} to be correct, '
        'not the one you think correct, please only include the numerical '
        'number]%```\nConfidence: ')


def stored_thought_steps(thought):
    matches = list(re.finditer(r'(Step \d+):', thought))
    if not matches:
        raise ValueError('accepted_thought_missing_source_step_markers')
    steps = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(thought)
        steps[match.group(1)] = thought[match.start():end].strip()
    return steps


def keyword_probability_artifact(steps, annotation, token_ids, logprobs, tokenizer):
    parsed = step_exacts_2_list(annotation)
    if not parsed or 'NO ANSWER' in annotation:
        raise ValueError('source_annotation_invalid')
    _, keyword_lists, contribution_lists = parsed
    if len(steps) > len(keyword_lists):
        raise ValueError('source_annotation_missing_steps')
    generated_ids = torch.tensor(token_ids)
    probabilities = [{token_id: math.exp(logprob)}
                     for token_id, logprob in zip(token_ids, logprobs)]
    keyword_probabilities, keyword_contributions, mapping_diagnostics = {}, {}, {}
    for step_index, (step_name, step_text) in enumerate(steps.items()):
        step_tokens = tokenizer.tokenize(step_text)
        processed = [token[1:] if token.startswith('Ġ') or token.startswith('▁')
                     else token for token in step_tokens]
        step_token_ids = tokenizer.convert_tokens_to_ids(step_tokens)
        start = find_subsequence_position(step_token_ids[1:-2], generated_ids) - 1
        if start < 0 or start + len(step_tokens) > len(token_ids):
            raise ValueError('source_keyword_alignment_unrecoverable')
        keyword_probabilities[step_name] = {}
        keyword_contributions[step_name] = {}
        mapping_diagnostics[step_name] = {}
        for keyword_index, keyword in enumerate(keyword_lists[step_index]):
            if not is_word_in_sentence(step_text, keyword):
                continue
            first, last = find_token_indices(processed, keyword)
            if first < 0 or last < first:
                raise ValueError('source_keyword_alignment_unrecoverable')
            original_ids = generated_ids[start + first:start + last + 1].tolist()
            if original_ids != step_token_ids[first:last + 1]:
                raise ValueError('source_keyword_alignment_unrecoverable')
            values = [probabilities[start + first + index].get(token_id, 0.)
                      for index, token_id in enumerate(original_ids)]
            keyword_probabilities[step_name][keyword] = values
            keyword_contributions[step_name][keyword] = int(
                contribution_lists[step_index][keyword_index])
            mapping_diagnostics[step_name][keyword] = {
                'accepted_token_indices': list(range(start + first, start + last + 1)),
                'token_ids': original_ids}
    if is_effectively_empty(keyword_probabilities):
        raise ValueError('source_keyword_probabilities_empty')
    return keyword_probabilities, keyword_contributions, mapping_diagnostics
