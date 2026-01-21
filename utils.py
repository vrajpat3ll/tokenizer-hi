from typing import List


def text_to_bytes(text: str) -> List[str]:
    """
    Converts text to a list of byte tokens.
    Each byte is represented as a string (e.g. '65').
    """
    return [str(b) for b in text.encode("utf-8")]


def bytes_to_unicode():
    """
    GPT-2 style reversible byte-to-unicode mapping.
    Ensures all bytes map to readable characters.
    """
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0

    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    return dict(zip(bs, map(chr, cs)))


def unicode_to_bytes(byte_encoder):
    return {v: k for k, v in byte_encoder.items()}


def text_to_byte_tokens(text: str, byte_encoder) -> List[str]:
    return [byte_encoder[b] for b in text.encode("utf-8")]
