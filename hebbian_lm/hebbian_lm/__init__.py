"""Hebbian language learner: local-learning-only model of words, grammar and facts."""

from .model import Config, HebbianLM
from .tokenizer import BOS, tokenize

__all__ = ["Config", "HebbianLM", "BOS", "tokenize"]
