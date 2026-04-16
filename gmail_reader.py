# gmail_reader.py
import imaplib
import email
import requests
import time
import getpass
import json
import os
from email.header import decode_header
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GmailSentimentReader:
    def __init__(self, email_address, app_password, api_url="http://localhost:5000"):
        """
        Initialize Gmail reader with app password
        
        Args:
            email_address: Your Gmail address
            app_password: 16-character app password from Google
            api_url: URL of your Flask sentiment API
        """
        self.email = email_address
        self.password = app_password
        self.api_url = api_url
        self.imap_server = "imap.gmail.com"
        self.mail = None
        
    def connect(self):
        """Connect to Gmail IMAP server"""
        try:
            logger.info(f"Connecting to Gmail as {self.email}...")
            self.mail = imaplib.IMAP4_SSL(self.imap_server)
            self.mail.login(self.email, self.password)
            logger.info("✓ Connected successfully!")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            logger.error("\nTroubleshooting tips:")
            logger.error("1. Make sure 2FA is enabled on your Google account")
            logger.error("2. Generate an App Password at: https://myaccount.google.com/apppasswords")
            logger.error("3. Use the 16-character app password, not your regular password")
            logger.error("4. Make sure IMAP is enabled in Gmail settings")
            return False
    
    def fetch_unread_emails(self, mark_as_read=True):
        """
        Fetch unread emails from inbox
        
        Args:
            mark_as_read: Whether to mark emails as read after fetching
        """
        if not self.mail:
            if not self.connect():
                return []
        
        try:
            # Select inbox
            self.mail.select("INBOX")
            
            # Search for unread emails
            status, messages = self.mail.search(None, 'UNSEEN')
            email_ids = messages[0].split()
            
            if not email_ids:
                logger.info("No new emails found")
                return []
            
            logger.info(f"Found {len(email_ids)} new email(s)")
            emails = []
            
            for email_id in email_ids:
                try:
                    # Fetch email
                    status, msg_data = self.mail.fetch(email_id, '(RFC822)')
                    
                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])
                            
                            # Parse email headers
                            subject = self.decode_header(msg.get("Subject", "No Subject"))
                            from_addr = msg.get("From", "Unknown")
                            date = msg.get("Date", "")
                            
                            # Parse email body
                            body = self.get_email_body(msg)
                            
                            emails.append({
                                'id': email_id,
                                'subject': subject,
                                'from': from_addr,
                                'date': date,
                                'body': body,
                                'raw_msg': msg
                            })
                            
                            # Mark as read if requested
                            if mark_as_read:
                                self.mail.store(email_id, '+FLAGS', '\\Seen')
                                
                except Exception as e:
                    logger.error(f"Error processing email {email_id}: {e}")
                    continue
            
            return emails
            
        except Exception as e:
            logger.error(f"Error fetching emails: {e}")
            return []
    
    def decode_header(self, header):
        """Decode email header (handles encodings)"""
        try:
            decoded, encoding = decode_header(header)[0]
            if isinstance(decoded, bytes):
                decoded = decoded.decode(encoding if encoding else 'utf-8')
            return decoded
        except:
            return header
    
    def get_email_body(self, msg):
        """Extract plain text body from email"""
        body = ""
        
        try:
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    
                    # Skip attachments
                    if "attachment" in content_disposition:
                        continue
                    
                    if content_type == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body = payload.decode('utf-8', errors='ignore')
                            break
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Error extracting body: {e}")
            body = ""
        
        return body.strip()
    
    def analyze_sentiment(self, text):
        """Send text to Flask API for sentiment analysis"""
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json={'sentence': text[:1000]},  # Limit text length
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error calling API: {e}")
            return None
    
    def process_emails(self, analyze_body=True, analyze_subject=True):
        """
        Process unread emails and analyze sentiment
        
        Args:
            analyze_body: Include email body in analysis
            analyze_subject: Include email subject in analysis
        """
        emails = self.fetch_unread_emails()
        
        if not emails:
            return []
        
        results = []
        
        for email_data in emails:
            # Combine subject and body for analysis
            text_to_analyze = ""
            if analyze_subject:
                text_to_analyze += email_data['subject'] + ". "
            if analyze_body:
                text_to_analyze += email_data['body'][:500]  # Limit to 500 chars
            
            if not text_to_analyze.strip():
                logger.warning(f"Empty email content from {email_data['from']}")
                continue
            
            # Analyze sentiment
            result = self.analyze_sentiment(text_to_analyze)
            
            if result and result.get('success'):
                email_result = {
                    'from': email_data['from'],
                    'subject': email_data['subject'],
                    'date': email_data['date'],
                    'sentiment': result['sentiment'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities'],
                    'analyzed_text': text_to_analyze[:200]  # Preview
                }
                results.append(email_result)
                
                # Display result
                self.display_email_result(email_result)
        
        return results
    
    def display_email_result(self, result):
        """Pretty print email analysis result"""
        print("\n" + "="*80)
        print(f"📧 From: {result['from']}")
        print(f"📝 Subject: {result['subject']}")
        print(f"📅 Date: {result['date']}")
        print("-"*80)
        
        # Sentiment with emoji
        sentiment = result['sentiment']
        emoji = "😊" if sentiment == "Positive" else "😞" if sentiment == "Negative" else "😐"
        print(f"🎯 Sentiment: {emoji} {sentiment}")
        print(f"📊 Confidence: {result['confidence']:.1f}%")
        
        # Probability bars
        print("\n📈 Confidence Levels:")
        probs = result['probabilities']
        max_len = max(len(k) for k in probs.keys())
        for label, prob in probs.items():
            bar_length = int(prob / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"  {label:>{max_len}}: {bar} {prob:.1f}%")
        
        print("="*80 + "\n")
    
    def save_results(self, results, filename="sentiment_results.json"):
        """Save analysis results to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {filename}")
    
    def close(self):
        """Close IMAP connection"""
        if self.mail:
            self.mail.close()
            self.mail.logout()
            logger.info("Disconnected from Gmail")


def main():
    """Main function to run email sentiment analyzer"""
    print("\n" + "="*80)
    print("📧 GMAIL SENTIMENT ANALYZER")
    print("="*80)
    print("\n⚠️  IMPORTANT: You need an App Password (not your regular password)")
    print("Generate one at: https://myaccount.google.com/apppasswords\n")
    
    # Get credentials
    email = input("Enter your Gmail address: ").strip()
    app_password = getpass.getpass("Enter your App Password (16 characters): ").strip()
    
    # Check if Flask API is running
    api_url = "http://localhost:5000"
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code != 200:
            print("\n❌ Flask API is not responding. Please run: python app.py")
            return
        print("✓ Flask API is running")
    except:
        print("\n❌ Cannot connect to Flask API. Please run: python app.py")
        return
    
    # Initialize reader
    reader = GmailSentimentReader(email, app_password, api_url)
    
    try:
        # Connect to Gmail
        if not reader.connect():
            return
        
        print("\n✓ Ready! Waiting for emails...")
        print("Press Ctrl+C to stop\n")
        
        # Continuous monitoring
        while True:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking for new emails...")
            
            # Process new emails
            results = reader.process_emails()
            
            # Save results if any
            if results:
                reader.save_results(results)
            
            # Wait before next check (60 seconds)
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n\n👋 Stopping email monitor...")
    finally:
        reader.close()


if __name__ == "__main__":
    main()