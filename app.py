import imaplib
import email
from email.header import decode_header
import os
from dotenv import load_dotenv
from inference import predict_sentiment

load_dotenv()

EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")

def read_latest_email():
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(EMAIL, PASSWORD)
    mail.select("inbox")

    _, messages = mail.search(None, "UNSEEN")
    mail_ids = messages[0].split()

    if not mail_ids:
        print("❌ No new emails.")
        return None

    latest_id = mail_ids[-1]

    _, msg_data = mail.fetch(latest_id, "(RFC822)")
    raw_email = msg_data[0][1]

    msg = email.message_from_bytes(raw_email)

    subject, encoding = decode_header(msg["Subject"])[0]
    if isinstance(subject, bytes):
        subject = subject.decode(encoding or "utf-8")

    body = ""

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_payload(decode=True).decode()
    else:
        body = msg.get_payload(decode=True).decode()

    return subject + " " + body


def main():
    print("\n📩 Checking email...")
    text = read_latest_email()

    if text:
        print("\n📨 Email Content:\n", text[:200])

        label, confidence = predict_sentiment(text)

        print("\n🧠 Prediction Result:")
        print(f"Sentiment: {label}")
        print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    main()