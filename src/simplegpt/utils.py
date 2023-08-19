import random

import numpy as np
import tiktoken
import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_tokenizer(name):
    class CharTokenizer:
        def encode(self, s):
            return [ord(c) for c in s]

        def decode(self, ids):
            return "".join([chr(i) for i in ids])

    if name == "char":
        return CharTokenizer()
    return tiktoken.get_encoding(name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
