from datasets import load_dataset


def write_dataset_to_file(dataset, path, label_dict):
    with open(path, encoding="utf8", mode="w") as file:
        for sample in dataset:
            if len(sample['tokens']) != len(sample['tags']):
                raise Exception("Different number of tokens and tags.")
            for token, tag in zip(sample['tokens'], sample['tags']):
                file.write(token + "\t" + label_dict[tag] + "\n")
            file.write("\n")


if __name__ == "__main__":
    # ds = load_dataset("tner/tweetner7")

    # recommended splits for general use
    train_dataset = load_dataset("tner/tweetner7", split="train_all")
    valid_dataset = load_dataset("tner/tweetner7", split="validation_2021")
    test_dataset = load_dataset("tner/tweetner7", split="test_2021")

    label_dict = {"B-corporation": 0, "B-creative_work": 1, "B-event": 2, "B-group": 3, "B-location": 4, "B-person": 5,
                  "B-product": 6, "I-corporation": 7, "I-creative_work": 8, "I-event": 9, "I-group": 10,
                  "I-location": 11, "I-person": 12, "I-product": 13, "O": 14}

    label_dict = {value: key for key, value in label_dict.items()}

    write_dataset_to_file(train_dataset, "tweetner7/train.txt", label_dict)
    write_dataset_to_file(valid_dataset, "tweetner7/dev.txt", label_dict)
    write_dataset_to_file(test_dataset, "tweetner7/test.txt", label_dict)