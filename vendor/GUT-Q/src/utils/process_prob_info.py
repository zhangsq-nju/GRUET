import numpy as np
import re

def group_prob_info(probinfo, window_size):
    if len(probinfo)>=window_size:
        return [sum(probinfo[i:i+window_size])/window_size for i in range(len(probinfo)-window_size+1)]
    return [sum(probinfo)/len(probinfo)]

def step_level_aggregate(probinfo, way, window_size):
    if way=='average':
        return sum(probinfo) / len(probinfo)

    pattern = r'^(bottom|tail|head)-(\d+)(%)?$'
    m = re.match(pattern, way)
    if not m:
        raise ValueError(f"Invalid format: {way}")
    method = m.group(1)
    number = int(m.group(2))
    is_percent = m.group(3) == '%'

    if is_percent:
        if method == 'bottom':
            group_scores = group_prob_info(probinfo, window_size)
            k = max(1, int(len(group_scores) * number / 100))
            bottom_scores = np.partition(group_scores, k-1)[:k]
            return sum(bottom_scores) / len(bottom_scores)
        elif method == 'tail':
            k = max(1, int(len(probinfo) * number / 100))
            tail_scores = probinfo[-k:]
            return sum(tail_scores) / len(tail_scores)
        elif method == 'head':
            k = max(1, int(len(probinfo) * number / 100))
            head_scores = probinfo[:k]
            return sum(head_scores) / len(head_scores)
    else:
        if method == 'bottom':
            group_scores = group_prob_info(probinfo, window_size)
            if len(group_scores)>number:
                bottom_scores = np.partition(group_scores, number-1)[:number]
                return sum(bottom_scores) / len(bottom_scores)
            else:
                return sum(group_scores) / len(group_scores)
        elif method == 'tail':
            if len(probinfo)>number:
                tail_scores = probinfo[-number:]
                return sum(tail_scores) / len(tail_scores)
            else:
                return sum(probinfo) / len(probinfo)
        elif method == 'head':
            if len(probinfo)>number:
                head_scores = probinfo[:number]
                return sum(head_scores) / len(head_scores)
            else:
                return sum(probinfo) / len(probinfo)
    raise ValueError(f"Invalid format: {way}")


def standard_probinfo_weight(probinfo_dict, token_level, way, window_size=6):
    if not probinfo_dict[token_level]:
        return 0.
    value = step_level_aggregate(probinfo_dict[token_level], way, window_size)
    if token_level=='max logprob':
        return -value
    elif token_level =='token logprob':
        return -value
    elif token_level =='avg neg logprob' or token_level == 'avg neg logprob nor':
        return 1./value
    elif token_level =='neg entropy' or token_level == 'neg entropy nor':
        return -value
    else:
        raise ValueError
