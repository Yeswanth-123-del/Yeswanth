# train_reddit_model.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
data = pd.read_csv('Reddit_Data.csv')

# Remove rows where 'clean_comment' or 'category' is NaN
data = data.dropna(subset=['clean_comment', 'category'])

print("Sample Data:")
print(data.head())
# Step 3: Separate features and labels
X = data['clean_comment']
y = data['category']

X = X.fillna("")  # Important step to handle missing text


# Step 4: Convert text into numbers (vectorization)
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Step 5: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Step 6: Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
with open('reddit_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the vectorizer
with open('reddit_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Step 7: Predict on test set
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Step 9: Manual Prediction
print("\nEnter your own text for prediction (type 'exit' to stop):")
while True:
    user_input = input("Your text: ")
    if user_input.lower() == 'exit':
        break
    user_vector = vectorizer.transform([user_input])  # transform into same vector shape
    prediction = model.predict(user_vector)
    print(f"Prediction: {prediction[0]}")
