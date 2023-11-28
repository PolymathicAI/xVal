# Defining the regular expression replacement to replace numbers with ¬s
# and add the numbers to a list
import re
import numpy as np

# Defining the regular expression replacement to replace numbers with ¬s
# and add the numbers to a list

def replace(text, numbers, num_token="[NUM]"):
    text = text.replace(num_token, "¬").replace("¬¬", "¬, ¬").replace("¬¬", "¬, ¬")
    for number in numbers:
        text = text.replace("¬", str(number), 1)
    return text

def compress_matrix(text):
    text = (
        text.replace("¬, ¬", "¬¬")
        .replace("¬, ¬", "¬¬")
        .replace("¬,¬", "¬¬")
        .replace("¬,¬", "¬¬")
    )
    return text

def extract(text, num_token="[NUM]"):
    import re

    # this regular expression is intended to match numerical values in various forms
    # like integers, floating-point numbers, or scientific notation, while avoiding
    # matching numbers that are part of strings.
    pattern = r"(?<!\')-?\d+(\.\d+)?([eE][-+]?\d+)?(?!\'|\d)"

    numbers = []

    def replace(match):
        numbers.append(match.group())
        return "¬"

    nonum_text = re.sub(pattern, replace, text)
    return compress_matrix(nonum_text).replace("¬", num_token), numbers

def tokenize_fnc(sample, tokenizer, num_token="[NUM]"):
    if type(sample) != str:
        sample = sample["text"]

    sample = sample.replace(" ", "")

    num_token_id = tokenizer.convert_tokens_to_ids(num_token)

    nonum_text, numbers = extract(sample, num_token=num_token)
    out = tokenizer(
        nonum_text, return_attention_mask=False, return_token_type_ids=False
    )
    ids = np.array(out["input_ids"])
    locs = ids == num_token_id
    num_embed = np.ones(len(ids)).astype(np.float16)
    num_embed[locs] = numbers
    out["numbers"] = num_embed
    out["len"] = len(ids)
    return out

def decode(sample, tokenizer, num_token="[NUM]"):
    num_token_id = tokenizer.encode(num_token)[0]
    text = tokenizer.decode(sample["input_ids"])
    numbers = [
        num
        for num, id in zip(sample["numbers"], sample["input_ids"])
        if id == num_token_id
    ]

    return replace(text, numbers, num_token=num_token)

def extract_all_keys(input_string):
    if type(input_string) == dict:
        input_string = str(input_string)

    assert type(input_string) == str, "input_string must be a string or a dictionary"

    pattern = r"'(.*?)'"
    keys = re.findall(pattern, input_string)
    return keys

def scientific_notation(match, sigfigs):
    number = float(match.group())
    if np.abs(number) <= 1e-8:
        return "+0.00e+0"

    # get the first three significant figures
    s = (
        ("{:+." + f"{sigfigs-1}" + "e}")
        .format(number)
        .replace("e-0", "e-")
        .replace("e+0", "e+")
    )
    return s

def convert_num_string(txt, sigfigs=3, remove_comma=True, remove_space=True):
    # p1000 encoding but ignoring numbers within single quotes
    out = re.sub(
        r"(?<!\')-?\d+(\.\d+)?([eE][-+]?\d+)?(?!\'|\d)",
        lambda x: scientific_notation(x, sigfigs),
        txt,
    )

    if remove_comma:
        out = re.sub(r"(\d),\s*([+-])", r"\1\2", out)

    if remove_space:
        out = out.replace(" ", "")

    return out
