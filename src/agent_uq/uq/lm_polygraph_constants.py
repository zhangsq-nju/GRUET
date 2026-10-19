from __future__ import annotations

import pathlib

_DEFAULT_NLI_MODEL_NAME = "microsoft/deberta-large-mnli"
_DEFAULT_CROSS_ENCODER_NAME = "cross-encoder/stsb-roberta-large"
_LM_POLYGRAPH_ADAPTER_CONTRACT = "whole_raw_candidate_v2"
_RAW_TOKEN_ARRAYS_MISALIGNED = "raw_token_arrays_misaligned"
_MODEL_CONF_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "conf" / "model"
).resolve()

DEFAULT_P_TRUE_PROMPT = (
    "Question: {initial_prompt}\n"
    "Possible answer: {candidate}\n"
    "Is the possible answer:\n"
    "(A) True\n"
    "(B) False\n"
    "The possible answer is:"
)

DEFAULT_VERBALIZED_PROMPT = (
    "Question: {initial_prompt}\n"
    "Guess: {candidate}\n"
    "Provide the probability that your guess is correct. Give ONLY the probability, no other words or explanation.\n\n"
    "For example:\n\n"
    "Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n"
    "Probability:"
)
DEFAULT_VERBALIZED_CONFIDENCE_REGEX = r"Probability:\s*(\d+\.\d+)"
_P_TRUE_SCORING_MODE = "generated_token_logprob_v1"
