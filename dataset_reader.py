import os

DATASET_FOLDER = "dataset/old_monk"

def load_dataset():

    text = ""

    for filename in sorted(os.listdir(DATASET_FOLDER)):

        if filename.endswith(".txt"):

            filepath = os.path.join(DATASET_FOLDER, filename)

            with open(filepath, "r", encoding="utf-8") as file:

                text += file.read() + "\n\n"

    return text


def get_total_files():

    return len([
        file
        for file in os.listdir(DATASET_FOLDER)
        if file.endswith(".txt")
    ])