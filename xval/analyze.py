import torch
from tqdm import tqdm
import pandas as pd
import numpy as np


def token_structure(sample, tokenizer, start=0, tokens=30):
    import textwrap

    def colored_text(text, color_code):
        return f"\033[{color_code}m{text}\033[0m"

    if start < 0:
        start = len(sample["input_ids"]) + start

    seq = sample["input_ids"][start : start + tokens]

    print(tokenizer.decode(seq))

    text_parts = []
    for i, el in enumerate(seq):
        index_part = f"{start+i}"  # Red color for index part
        element_part = colored_text(f"[{el}]", "32")  # Green color for element part
        decoded_part = colored_text(
            f"{tokenizer.decode(el)}", "34"
        )  # Blue color for decoded text part
        to_append = f"{index_part},{element_part},{decoded_part}"
        if "numbers" in sample.keys():
            to_append += colored_text(f",{sample['numbers'][start+i]:.2g}", "31")
        text_parts.append(to_append)

    text = "   ".join(text_parts)

    print()
    print("\n".join(textwrap.wrap(text, 240, break_long_words=False)))


def mask_numbers(sample, tokenizer, n_list):
    import copy

    mask_token = tokenizer.encode("[MASK]")[0]
    masked_sample = copy.deepcopy(sample)
    len_ = len(masked_sample["input_ids"])
    masked_sample["masked_numbers"] = copy.deepcopy(sample["numbers"])[:len_]
    masked_sample["numbers"] = masked_sample["numbers"][:len_]
    masked_sample["labels"] = sample["input_ids"]
    for n in n_list:
        masked_sample["input_ids"][n] = mask_token
        masked_sample["masked_numbers"][n] = 1.0
        # Next two lines are for calculating the correct mlm loss
        # tells the model to only look at the masked token for calculating x-entropy
        masked_sample["labels"] = list(0 * np.array(masked_sample["labels"]) - 100)
        masked_sample["labels"][n] = sample["input_ids"][n]
        masked_sample["ans"] = masked_sample["numbers"][n]
    masked_sample["text"] = tokenizer.decode(sample["input_ids"])
    masked_sample["masked_text"] = tokenizer.decode(masked_sample["input_ids"])
    return masked_sample


def mask_nth_number(sample, tokenizer, n):
    import copy

    mask_token = tokenizer.encode("[MASK]")[0]
    masked_sample = copy.deepcopy(sample)
    masked_sample["input_ids"][n] = mask_token
    len_ = len(masked_sample["input_ids"])
    masked_sample["masked_numbers"] = copy.deepcopy(sample["numbers"])[:len_]
    masked_sample["numbers"] = masked_sample["numbers"][:len_]
    masked_sample["labels"] = sample["input_ids"]
    masked_sample["masked_numbers"][n] = 1.0
    # Next two lines are for calculating the correct mlm loss
    # tells the model to only look at the masked token for calculating x-entropy
    masked_sample["labels"] = list(0 * np.array(masked_sample["labels"]) - 100)
    masked_sample["labels"][n] = sample["input_ids"][n]
    masked_sample["text"] = tokenizer.decode(sample["input_ids"])
    masked_sample["masked_text"] = tokenizer.decode(masked_sample["input_ids"])
    masked_sample["ans"] = masked_sample["numbers"][n]
    return masked_sample


### Each number
def predict(model, masked_sample, device="cuda"):
    model.eval()
    model.to(device)
    input = {
        "x": torch.tensor(masked_sample["input_ids"]).view(1, -1).to(device),
        # "y": torch.tensor(masked_sample["labels"]).view(1, -1).to(device),
        "x_num": torch.tensor(masked_sample["masked_numbers"]).view(1, -1).to(device),
        # "y_num": torch.tensor(masked_sample["masked_numbers"]).view(1, -1).to(device),
    }
    out = model(**input)
    return out


### Each row
def predict_numbers(model, sample, tokenizer, n_list, device, all_at_once=False):
    num_pred_list = []
    num_true_list = []
    if all_at_once:
        masked_sample = mask_numbers(sample, tokenizer, n_list)
        out = predict(model, masked_sample, device)
        for n in n_list:
            num_pred_list.append(out[1][0][n].item())
            num_true_list.append(masked_sample["numbers"][n])
    else:
        for n in n_list:
            masked_sample = mask_nth_number(sample, tokenizer, n)
            out = predict(model, masked_sample, device)
            num_pred_list.append(out[1][0][n].item())
            num_true_list.append(masked_sample["numbers"][n])

    return {
        "num_pred_list": num_pred_list,
        "num_true_list": num_true_list,
    }


### Run on whole dataset
def slow_eval_numbers(
    model, dataset, tokenizer, n_list, device, num_samples=None, all_at_once=False
):
    model.eval()
    model.to(device)

    if num_samples is None:
        num_samples = len(dataset)

    with torch.no_grad():
        out = []
        for i in tqdm(range(num_samples)):
            sample = dataset[i]
            out.append(
                predict_numbers(model, sample, tokenizer, n_list, device, all_at_once)
            )

    pd_out = pd.DataFrame(out)
    return pd_out
