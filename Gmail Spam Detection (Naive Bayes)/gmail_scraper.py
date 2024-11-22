import os
import base64
import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
from google.auth.transport.requests import Request


#defining the scope of the application
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

#Authentication & Service creation
def authenticate_gmail():
    creds = None
    token_file = 'token.json'
    client_secrets_file = 'key.json'

    if os.path.exists(token_file):
        print("Token found and available.")
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_file, SCOPES)
            # Specify the exact redirect URI that matches your Google Cloud Console
            creds = flow.run_local_server(
                port=8080,
                redirect_uri_port=8080,
                authorization_prompt_message='Please visit this URL: '
            )
            with open(token_file, 'w') as token:
                token.write(creds.to_json())

def get_emails(service, label_ids=['INBOX'], query=''):
    try:
        # Get the current date and calculate the date 10 days ago
        ten_days_ago = (datetime.datetime.now() - datetime.timedelta(days=10)).strftime('%Y/%m/%d')

        # Use the 'after' parameter in the query to get emails from the last 10 days
        query = f"after:{ten_days_ago}"

        # Call the Gmail API to fetch emails
        results = service.users().messages().list(userId='me', labelIds=label_ids, q=query).execute()
        messages = results.get('messages', [])

        if not messages:
            print('No messages found.')
            return []

        # Fetch details of each message
        email_data = []
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            payload = msg['payload']
            headers = payload['headers']
            
            # Extract relevant details like subject, sender, date, and body
            subject = next((item['value'] for item in headers if item['name'] == 'Subject'), 'No Subject')
            sender = next((item['value'] for item in headers if item['name'] == 'From'), 'Unknown Sender')
            date = next((item['value'] for item in headers if item['name'] == 'Date'), 'No Date')

            # Extract the email body (plain text)
            body = ''
            if 'parts' in payload:
                for part in payload['parts']:
                    if part['mimeType'] == 'text/plain':
                        body = base64.urlsafe_b64decode(part['body']['data']).decode()

            email_data.append({
                'subject': subject,
                'sender': sender,
                'date': date,
                'body': body
            })

        return email_data

    except HttpError as error:
        print(f'An error occurred: {error}')
        return []
    

def main():
    service = authenticate_gmail()
    if service:
        emails = get_emails(service)
        for email in emails:
            print(f"Subject: {email['subject']}")
            print(f"Sender: {email['sender']}")
            print(f"Date: {email['date']}")
            print(f"Body: {email['body'][:200]}...")  # Preview first 200 characters of body
            print('-' * 50)
        df = pd.DataFrame(emails)
        df.to_csv('Emails.csv')
if __name__ == '__main__':
    main()