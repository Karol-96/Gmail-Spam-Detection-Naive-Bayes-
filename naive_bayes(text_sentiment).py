from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("Libraries imported successfully")  # Debug print

# Initial dataset
texts = [
    "I love this product",
    "This is the worst purchase",
    "Best deal ever",
    "I hate it"
]
labels = [1, 0, 1, 0]  # 1 = Positive, 0 = Negative

print("Dataset initialized") 
# Additional data for training and testing
new_texts = [
    "Absolutely fantastic experience",
    "Terrible quality, never buying again",
    "Good value for the price",
    "Awful service and rude staff",
    "Excellent customer service",
    "Not worth the money"
]
new_labels = [1, 0, 1, 0, 1, 0]  # Corresponding labels

# Combine datasets
all_texts = texts + new_texts
all_labels = labels + new_labels

# Convert text to features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(all_texts)

# Split combined data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, all_labels, test_size=0.25, random_state=42)

# Train Naive Bayes model
model = MultinomialNB(alpha=1.0)  # Default smoothing
model.fit(X_train, y_train)

# Predict and evaluate on the test set
y_pred = model.predict(X_test)

# Output results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Example: Predicting on new unseen data
new_inputs = [
    "Amazing quality and great support",
    "Terrible, I want a refund",
    "Good but could be better",
    "I absolutely love it"
]
new_features = vectorizer.transform(new_inputs)
predictions = model.predict(new_features)

print("\nPredictions on new data:")
for text, pred in zip(new_inputs, predictions):
    print(f"'{text}' => {'Positive' if pred == 1 else 'Negative'}")
