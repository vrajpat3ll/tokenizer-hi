from typing import Dict, List, Tuple, Union
from collections import defaultdict

from utils import bytes_to_unicode, text_to_byte_tokens, unicode_to_bytes
from constants import SPECIAL_TOKENS


def init_byte_vocab():
    byte_encoder = bytes_to_unicode()
    vocab = {ch: i for i, ch in enumerate(byte_encoder.values())}
    vocab.update(SPECIAL_TOKENS)
    return vocab, byte_encoder


class BPETokenizer:
    def __init__(
        self,
        num_merges: int = 10_000,
        # vocab: Union[Dict[str, int], None] = None,
    ) -> None:
        self.num_merges = num_merges
        self.merges = []

        self.vocab, self.byte_encoder = init_byte_vocab()
        self.byte_decoder = unicode_to_bytes(self.byte_encoder)

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def get_pair_frequencies(
        self,
        sequences: List[List[str]],
    ) -> Dict[Tuple[str, str], int]:
        pair_freq = defaultdict(int)

        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_freq[pair] += 1

        return pair_freq

    def merge_pair(self, pair, sequences):
        merged = []
        a, b = pair
        ab = a + b

        for seq in sequences:
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    new_seq.append(ab)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            merged.append(new_seq)

        return merged

    def train(self, sequences):
        for i in range(self.num_merges):
            pair_freq = self.get_pair_frequencies(sequences)
            if not pair_freq:
                break

            best_pair: Tuple[str, str] = max(pair_freq, key=pair_freq.get)
            sequences = self.merge_pair(best_pair, sequences)

            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)
            self.inverse_vocab[len(self.inverse_vocab)] = new_token
            self.merges.append(best_pair)

            # if i % 500 == 0:
            print(f"Merge {i}: {best_pair}, ", self._test())

        return self.vocab, self.merges

    def apply_merge(self, tokens: List[str], pair: tuple[str, str]) -> List[str]:
        merged = []
        i = 0
        a, b = pair
        ab = a + b

        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                merged.append(ab)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1

        return merged

    def encode(self, text: str) -> List[int]:
        """
        Encode text into a list of token IDs using byte-level BPE.
        """

        # 1. Text → byte-level symbols
        tokens = text_to_byte_tokens(text, self.byte_encoder)
        # 2. Apply BPE merges IN TRAINING ORDER
        for pair in self.merges:
            tokens = self.apply_merge(tokens, pair)

        # 3. Tokens → token IDs
        token_ids = []
        for tok in tokens:
            if tok in self.vocab:
                token_ids.append(self.vocab[tok])
            else:
                # Should be rare if training was correct
                token_ids.append(self.vocab.get("<unk>"))

        return token_ids

    def _test(self):
        text = "भारत एक देश है"

        ids = self.encode(text)
        decoded = self.decode(ids)

        return decoded == text

    def decode(self, token_ids: List[int]) -> str:
        # 1. Token IDs → merged token strings
        tokens = [self.inverse_vocab[i] for i in token_ids]

        # 2. Flatten merged tokens into byte symbols
        byte_symbols = []
        for tok in tokens:
            for ch in tok:
                if ch in SPECIAL_TOKENS:
                    continue
                byte_symbols.append(ch)

        # 3. Byte symbols → bytes
        byte_values = [self.byte_decoder[ch] for ch in byte_symbols]

        # 4. Bytes → UTF-8 string
        return bytes(byte_values).decode("utf-8", errors="replace")
