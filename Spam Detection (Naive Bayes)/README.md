# Email Spam Detection using Naive Bayes

A Python-based email spam detection system that uses the Naive Bayes algorithm to classify emails as spam or non-spam. The system integrates with Gmail API to fetch emails and applies machine learning to identify potential spam messages.

## Features
- Gmail API integration for email fetching
- Text preprocessing and cleaning
- Naive Bayes classification
- Detailed spam analysis with confidence scores
- Support for both subject and body text analysis

## Setup and Installation

1. Install required packages:
```bash
pip install pandas numpy scikit-learn nltk google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

2. Download NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```

3. Set up Google Cloud Project and enable Gmail API:
   - Create a project in Google Cloud Console
   - Enable Gmail API
   - Download credentials as `key.json`

## Usage

1. First, fetch emails using the Gmail API:
```bash
python defender.py
```

2. Then run the spam detection:
```bash
python detector.py
```

## Sample Output

```
Loading and preparing email data...
Cleaning email text...
Converting text to features...
Training Naive Bayes classifier...
Testing the model...

Classification Report:
              precision    recall  f1-score   support

           0       0.60      0.67      0.63         9
           1       0.70      0.64      0.67        11

    accuracy                           0.65        20
   macro avg       0.65      0.65      0.65        20
weighted avg       0.65      0.65      0.65        20

=== SPAM EMAILS ===

Email 92:
Subject: The best tools to streamline your email marketing efforts
From: Neil Patel <np@neilpatel.com>
Confidence: 100.00%
--------------------------------------------------

[... other spam emails ...]

Total emails analyzed: 100
Spam emails detected: 67
Spam ratio: 67.0%
```

## How It Works

1. **Data Collection**: Uses Gmail API to fetch recent emails
2. **Preprocessing**: 
   - Combines subject and body text
   - Removes special characters and stopwords
   - Converts text to lowercase
3. **Feature Extraction**: Uses bag-of-words approach
4. **Classification**: Implements Naive Bayes algorithm
5. **Analysis**: Provides detailed report of spam detection results

## Project Structure
```
Spam Detection (Naive Bayes)/
├── defender.py        # Gmail API integration
├── detector.py        # Spam detection logic
├── key.json          # Google API credentials
├── token.json        # OAuth token
├── Emails.csv        # Stored email data
└── README.md         # Project documentation
```

## Performance Metrics
- Precision: 0.70 for spam detection
- Recall: 0.64 for spam detection
- Overall Accuracy: 65%

## Future Improvements
- Implement more sophisticated text preprocessing
- Add support for attachment analysis
- Include sender reputation scoring
- Add regular model retraining
- Improve classification accuracy

## License
MIT License

## Author
Karol Bhandari

## Acknowledgments
- Google Gmail API
- scikit-learn
- NLTK