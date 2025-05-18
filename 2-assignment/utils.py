import torch

def apply_kaggle_model(model, mapper, x_val, y_val):

    result = []
    valid_indices = []
    for idx, text in x_val.items():
        try:
            result.append(mapper[model(text)[0]['label']])
            valid_indices.append(idx)  # Only keep successful ones
        except Exception as e:
            print(f"Error processing text at index {idx}: {text}")
            print(e)

    # Update x_val and y_val to only keep successful entries
    x_val = x_val.loc[valid_indices].reset_index(drop=True)
    y_val = y_val.loc[valid_indices].reset_index(drop=True)

    # comparing the results
    from sklearn.metrics import classification_report, accuracy_score
    print(classification_report(y_val, result))
    print(accuracy_score(y_val, result))

def tokenize_function(examples, tokenizer):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


def tokenize_data(texts, tokenizer):
    return tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=512,
    )


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    


class CustomDataset1(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        # Assuming labels are 0 or 1. If you have more classes,
        # you'll need to adjust num_classes
        self.num_classes = 2 # Based on your model's config.id2label

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        
        # Get the original integer label (0 or 1)
        original_label = self.labels[idx]
        
        # Create a one-hot encoded tensor
        # Example: if original_label is 0, one_hot_label will be [1., 0.]
        #          if original_label is 1, one_hot_label will be [0., 1.]
        one_hot_label = torch.zeros(self.num_classes)
        one_hot_label[original_label] = 1.0
        
        # Assign the one-hot encoded label (must be float for BCEWithLogitsLoss)
        item["labels"] = one_hot_label.float() # Ensure it's float

        return item

    def __len__(self):
        return len(self.labels)