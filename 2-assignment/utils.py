
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
