# predict_text.py

import pickle

# 1. Load model
with open('reddit_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Load vectorizer
with open('reddit_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# 3. Take input
text = input("Enter your text: ")

# 4. Transform and Predict
text_vectorized = vectorizer.transform([text])
prediction = model.predict(text_vectorized)

print("Prediction:", prediction[0])
