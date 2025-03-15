from sklearn.svm import SVC
from our_feature_extraction import basic_bag, tf_idf, ClassAwareVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def support_vector_machine(x_train_vec, x_val_vec, y_train, y_val):

    # Initialize the SVM model
    model = SVC(kernel='linear', random_state=42)
    
    # Train the model
    model.fit(x_train_vec, y_train)
    
    # Make predictions
    y_pred = model.predict(x_val_vec)
    
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