import os
from pathlib import Path
import pickle
import sys
from typing import List, Union
from bpe import BPETokenizer
from utils import text_to_byte_tokens, text_to_bytes
from constants import (
    SPECIAL_TOKENS,
    COLOR_CYAN,
    COLOR_RED,
    COLOR_RESET,
    DATA_DIREDCTORY,
)


def build_corpus(path: Union[Path, str], limit: int = 1000) -> List[str]:
    path = Path(path)
    if not path.exists():
        raise OSError(f"Directory not found! path={path.resolve()}")
    if not path.is_dir():
        raise OSError(f"Not a directory: {path.resolve()}")

    count = 0
    sentences = []
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            if count > limit:
                break
            # print(f"{dirpath = }, {dirnames = }, {file_path }")
            file_path = (Path(dirpath) / filename).resolve()
            s = file_path.read_text("utf-8")
            # s += "<|endoftext|>"  # to inform that text is over
            sentences.append(s)
            count += 1
    return sentences


def main():
    print("Hello from tokenizer-hi!")
    try:
        corpus = build_corpus(
            DATA_DIREDCTORY,
        )
    except OSError as e:
        print(f"{COLOR_RED}[ERROR]{COLOR_RESET} {e}")
        sys.exit(1)

    # create a tokenizer()
    # initialize a vocabulary
    # train the tokenizer with the corpus and required config
    # save the tokenizer object in a pickle or something else
    # store the vocab and special tokens

    tokenizer = BPETokenizer(
        num_merges=100,
        # vocab=vocab,
    )

    sequences = [text_to_byte_tokens(text, tokenizer.byte_encoder) for text in corpus]
    vocab, merges = tokenizer.train(sequences)

    print(f"{COLOR_CYAN}Vocab size:{COLOR_RESET} {len(vocab)}")

    # Save tokenizer
    with open("tokenizer_hi.pkl", "wb") as f:
        pickle.dump(
            {
                "vocab": vocab,
                "merges": merges,
                "special_tokens": SPECIAL_TOKENS,
            },
            f,
        )


if __name__ == "__main__":
    main()
