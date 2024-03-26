# %%
import sys, os

sys.path.append("..")
from rebound_sim import construct_example
import numpy as np
import uuid
from tqdm import tqdm

path = "."
os.makedirs(path, exist_ok=True)


def reformat_data(data):
    data["description"].pop("seed")
    stepsize = data["description"].pop("stepsize")
    keys_list = data["description"].keys()

    rescale = {"m": 1e5, "a": 1, "e": 20}

    desc_dict = {
        "planet" + str(i): {key: val * rescale[key] for key, val in zip(keys_list, el)}
        for i, el in enumerate(zip(*data["description"].values()))
    }

    desc_dict["stepsize"] = stepsize

    data = [[[l2["x"], l2["y"]] for l2 in l1] for l1 in data["data"]]

    return str({"description": desc_dict, "data": data}).replace(" ", "")


if __name__ == "__main__":
    import sys

    num_draws = int(sys.argv[1])

    filename = str(uuid.uuid4())

    samples = []

    num_prints = 10
    print_every = num_draws // num_prints

    print("\nStarting sample generation...\n")
    drawn_samples = 0
    while drawn_samples < num_draws:
        sample = reformat_data(construct_example())
        max_value = np.abs(np.array(eval(sample)["data"])).max()
        if max_value < 10:
            drawn_samples += 1
            samples.append(reformat_data(construct_example()))
        else:
            print("Sample rejected, max value: {}".format(max_value))

        if drawn_samples % print_every == 0:
            print("Sample {}/{}".format(drawn_samples, num_draws))

    print("\nWriting to file: {}{}\n".format(path, filename))

    with open(
        r"{}{}".format(path, filename),
        "w",
    ) as f:
        for el in samples:
            # write each item on a new line
            f.write("{}\n".format(el))

    print("Done!")

# %%
