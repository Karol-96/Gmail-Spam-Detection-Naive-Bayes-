import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Note: First, let's load and prepare our email dataset
def prepare_data(csv_file):
    """
    Hey! This function helps us load and clean our email data.
    I'm combining subject and body to get more context for classification.
    """
    df = pd.read_csv(csv_file)
    
    # Combine subject and body for better context
    df['full_text'] = df['subject'] + ' ' + df['body'].fillna('')
    
    # For now, let's manually label some emails as spam (1) or not spam (0)
    # We'll use some common spam indicators
    df['is_spam'] = df['full_text'].apply(lambda x: 1 if any(indicator in x.lower() for indicator in 
        ['unsubscribe', 'click here', 'limited time', 'act now', 'free trial', 'winner']) else 0)
    
    return df

def clean_text(text):
    """
    This is where we clean up our text data - removing special characters, 
    converting to lowercase, etc. Makes the data more consistent!
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

def train_spam_detector():
    """
    This is where the magic happens! We're using Naive Bayes because it works 
    really well with text classification and is surprisingly effective for spam detection.
    """
    # Load and prepare our data
    print("Loading and preparing email data...")
    df = prepare_data('Emails.csv')
    
    # Clean the text
    print("Cleaning email text...")
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    
    # Convert text to numerical features using bag of words
    print("Converting text to features...")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['is_spam']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    print("Training Naive Bayes classifier...")
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    
    # Make predictions
    print("Testing the model...")
    y_pred = clf.predict(X_test)
    
    # Print performance metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and vectorizer for future use
    return clf, vectorizer

def predict_spam(text, clf, vectorizer):
    """
    Use this function to classify new emails as they come in!
    """
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = clf.predict(features)
    probability = clf.predict_proba(features)
    
    return {
        'is_spam': bool(prediction[0]),
        'confidence': f"{max(probability[0]) * 100:.2f}%"
    }

def analyze_emails():
    """
    Analyze all emails and display which ones are classified as spam
    """
    # Load the data
    df = pd.read_csv('Emails.csv')
    
    # Train the model first
    classifier, vectorizer = train_spam_detector()
    
    # Analyze each email
    results = []
    for index, row in df.iterrows():
        full_text = f"{row['subject']} {row['body']}"
        prediction = predict_spam(full_text, classifier, vectorizer)
        
        results.append({
            'subject': row['subject'],
            'sender': row['sender'],
            'is_spam': prediction['is_spam'],
            'confidence': prediction['confidence']
        })
    
    # Create a DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Display spam emails
    print("\n=== SPAM EMAILS ===")
    spam_emails = results_df[results_df['is_spam']]
    for idx, email in spam_emails.iterrows():
        print(f"\nEmail {idx + 1}:")
        print(f"Subject: {email['subject']}")
        print(f"From: {email['sender']}")
        print(f"Confidence: {email['confidence']}")
        print("-" * 50)
    
    # Display statistics
    print(f"\nTotal emails analyzed: {len(results_df)}")
    print(f"Spam emails detected: {len(spam_emails)}")
    print(f"Spam ratio: {(len(spam_emails)/len(results_df))*100:.1f}%")

if __name__ == "__main__":
    analyze_emails()