
def apply_kaggle_model(model, mapper, x_val, y_val):

    result = []

    for text in x_val:
        try:
            result.append(mapper[model(text)[0]['label']])
        except Exception as e:
            print(f"Error processing text: {text}")
            print(e)
            result.append(0)

    # comparing the results
    from sklearn.metrics import classification_report, accuracy_score
    print(classification_report(y_val, result))
    print(accuracy_score(y_val, result))
