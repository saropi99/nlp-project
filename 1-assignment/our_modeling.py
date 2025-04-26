from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from our_feature_extraction import basic_bag, tf_idf
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def support_vector_machine(x_train_vec, x_val_vec, y_train, y_val):
    svm_model = SVC()
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }
    grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Fit the model to training data
    grid_search.fit(x_train_vec, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Accuracy:", grid_search.best_score_)

    # Make predictions
    y_pred = grid_search.predict(x_val_vec)
    
    # Evaluate the model
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    print("Accuracy:", accuracy_score(y_val, y_pred))

def nb(x_train_vec, x_val_vec, y_train, y_val):
    # Train the Naive Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(x_train_vec, y_train)

    # Make predictions
    y_pred = nb_model.predict(x_val_vec)

    # Evaluate the model
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

def random_forest(x_train_vec, x_val_vec, y_train, y_val, random_state=42):
    # Train the Random Forest model
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees
        'max_depth': [None, 10, 20],  # Depth of each tree
        'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
        'min_samples_leaf': [1, 2, 4],  # Minimum samples required in a leaf node
    }

    # Create RandomForest model
    rf = RandomForestClassifier(random_state=random_state)

    # Grid search with cross-validation
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(x_train_vec, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Accuracy:", grid_search.best_score_)

    # Make predictions
    y_pred = grid_search.predict(x_val_vec)
    
    # Evaluate the model
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    print("Accuracy:", accuracy_score(y_val, y_pred))