from tokenizers import (
    decoders,
    models,
    processors,
    Tokenizer,
    pre_tokenizers,
)
import numpy as np


def make_tokenizer(
    vocab_words=[],
    encoding="xval",
    save_file="./tokenizer.json",
    special_tokens=["[END]", "[MASK]", "[PAD]"],
    efficient_json=True,
    sample_keys=None,
):
    if encoding == "xval":
        special_tokens += ["[NUM]"]
        full_vocab = {}

    else:
        vocab = ["{", "}", "[", "]", ",", "-", "+", "#"]
        full_vocab = {el: i for i, el in enumerate(vocab)}

        if encoding == "fp15":
            vocab_words += (
                [
                    "]]",
                    "[[",
                    "]]]",
                    "[[[",
                    "],[",
                    "]],[[",
                    "]]],[[[",
                    "'data':",
                    "'description':",
                    "+0.00e+0",
                ]
                + [
                    f"{s}{n:.2f}e+{i}"
                    for n in np.arange(1, 10, 0.01)
                    for i in range(0, 8)
                    for s in ["+", "-"]
                ]
                + [
                    f"{s}{n:.2f}e-{i}"
                    for n in np.arange(1, 10, 0.01)
                    for i in range(1, 8)
                    for s in ["+", "-"]
                ]
            )

        elif encoding == "p10":
            vocab_words += (
                [
                    "]]",
                    "[[",
                    "]]]",
                    "[[[",
                    "],[",
                    "]],[[",
                    "]]],[[[",
                    "'data':",
                    "'description':",
                    "+0.00e+0",
                    "+",
                    "-",
                ]
                + [str(d) for d in range(10)]
                + [f"e+{i}" for i in range(0, 8)]
                + [f"e-{i}" for i in range(1, 8)]
            )

        elif encoding == "p1000":
            vocab_words = (
                [
                    "]]",
                    "[[",
                    "]]]",
                    "[[[",
                    "],[",
                    "]],[[",
                    "]]],[[[",
                    "'data':",
                    "'description':",
                    "0.00",
                ]
                + [f"{n:.2f}" for n in np.arange(1, 10, 0.01)]
                + [f"e+{i}" for i in range(0, 8)]
                + [f"e-{i}" for i in range(1, 8)]
            )

        elif encoding == "b1999":
            vocab_words = (
                [
                    "]]",
                    "[[",
                    "]]]",
                    "[[[",
                    "],[",
                    "]],[[",
                    "]]],[[[",
                    "'data':",
                    "'description':",
                    "+0.00",
                ]
                + [f"{s}{n:.2f}" for n in np.arange(1, 10, 0.01) for s in ["+", "-"]]
                + [f"e+{i}" for i in range(0, 8)]
                + [f"e-{i}" for i in range(1, 8)]
            )

    if efficient_json:
        efficient_vocab = [
            "{",
            "}",
            "[",
            "]",
            ",",
            "]]]",
            "[[[",
            "]]",
            "[[",
            "]]],[[[",
            "]],[[",
            "],[",
        ]
        vocab_words += efficient_vocab

    if sample_keys is not None:
        vocab_words += [f"'{el}':" for el in sample_keys]

    tokenizer = Tokenizer(models.BPE(vocab=full_vocab, merges=[]))
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.add_tokens(vocab_words)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(save_file)
