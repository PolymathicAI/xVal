import os, argparse
import numpy as np
from xval.preprocess import extract_all_keys, tokenize_fnc, convert_num_string
from xval.tokenizer import make_tokenizer
from datasets import DatasetDict
from transformers import PreTrainedTokenizerFast
import yaml 

parser = argparse.ArgumentParser()
parser.add_argument("--datapath", type=str, help="Path to data")
parser.add_argument("--savepath", type=str, help="Save path for tokenizer and tokenized dataset.")
parser.add_argument("--encoding", type=str, choices=['xval', 'fp15', 'p10', 'p1000', 'b1999'], default='xval', help="Choose your encoding scheme. (xVal is ours.)")
args = parser.parse_args()

data_path = args.datapath
save_path = args.savepath

print(f"Using encoding: {args.encoding}") 

if save_path is None:
    save_path = data_path # use same path by default

files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and "test" in f or "train" in f or "val" in f and "json" not in f and "token" not in f and "yaml" not in f]

print(f"\nLoading dataset from {data_path} with splits: {files}")
ds = DatasetDict.from_text({file: data_path + "/" + file for file in files})

print("\nExtracting keys from the train set...")
ds_keys = ds["train"].map(
    lambda x: {"keys": extract_all_keys(x["text"])},
    num_proc=30,
    remove_columns=["text"],
    load_from_cache_file=False,
)
sample_keys = list(set([item for sublist in ds_keys["keys"] for item in sublist]))
print(f"\nExtracted keys from dataset:  {sample_keys}\n")

tokenizer_path = save_path + "tokenizer_{}.json".format(args.encoding)
os.makedirs(save_path, exist_ok=True)
make_tokenizer(
    encoding=args.encoding,
    save_file=tokenizer_path, 
    efficient_json=True, 
    sample_keys=sample_keys
)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=tokenizer_path,
    bos_token="[END]",
    eos_token="[END]",
    mask_token="[MASK]",
    pad_token="[PAD]",
)

print("Tokenizer saved to: ", tokenizer_path)
print("Vocab size: ", len(tokenizer.vocab))

print("\nStarting tokenization...")
if args.encoding == 'xval':
    tokenize_lambda = lambda x: tokenize_fnc(x, tokenizer)
    batched = False
    batch_size = None
    
else: 
    def tokenize_lambda(samples):
        out = []
        for sample in samples["text"]:
            out.append(tokenizer.encode(convert_num_string(sample)))
        return {"input_ids": out}
    batched = True
    batch_size = 100

tokenized_ds = ds.map(
    tokenize_lambda,
    batched=batched,
    num_proc=30,
    batch_size=batch_size,
    remove_columns=["text"],
    load_from_cache_file=False,
)

if args.encoding == "xval":
    max_len = max([max(tokenized_ds[key]["len"]) for key in tokenized_ds.keys()])
    tokenized_ds = DatasetDict(
    {key: val.remove_columns(["len"]) for key, val in tokenized_ds.items()}
)
else:
    lens = tokenized_ds["train"].map(
    lambda x: {"len": len(x["input_ids"])},
    num_proc=30,
    remove_columns="input_ids",
    load_from_cache_file=False,
    )["len"]
    max_len = max(lens)

print(f"Longest sequence length: {max_len}")

print("\nTokenization finished. Saving...")
full_save_path = save_path + "/tokenized_ds_"+str(args.encoding)
tokenized_ds.save_to_disk(full_save_path)
print("Tokenized dataset saved to: ", full_save_path)

config = {
    "vocab_size": len(tokenizer.vocab),
    "block_size": max_len,
    "tokenizer": tokenizer_path,
    "dataset": full_save_path,
    "dataset_type": "hf_fullsample_numbers_mlm",
    "mask_token": "[MASK]",
}

config_path = save_path + f"/config_{args.encoding}.yaml"
with open(config_path, "w") as file:
    yaml.dump(config, file)

print("\nConfiguration saved to: ", config_path)
