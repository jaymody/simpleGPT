import os

import numpy as np
from tqdm import tqdm

from .utils import get_tokenizer


def count_lines(filename):
    """Count the number of lines in a file fast.

    Stolen from: https://stackoverflow.com/a/27518377/11070463
    """
    f = open(filename, "rb")
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b"\n")
        buf = read_f(buf_size)

    return lines


def main(text_file: str, output_file: str, tokenizer_name: str):
    assert not os.path.exists(output_file), f"Output file already exists: {output_file}"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    tokenizer = get_tokenizer(tokenizer_name)

    # TODO: maybe this should be done in chunks of characters instead of line by line
    # in case a line has a crap ton of characters that may not fit in memory
    ids = []
    num_lines = count_lines(text_file)
    with open(text_file, "r") as f:
        for line in tqdm(f, total=num_lines):
            ids.extend(tokenizer.encode(line))

    print(f"number of tokens = {len(ids)}")
    print(f"saving to {output_file}")
    # TODO: np.uint16 saves space, but this only works if the ids are < 65536, which
    # is true for the "gpt2" tokenizer, but other tokenizers may have a larger vocab
    np.array(ids, dtype=np.uint16).tofile(output_file)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
