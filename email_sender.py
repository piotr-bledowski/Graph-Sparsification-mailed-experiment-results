import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from config import *
from credentials import *


def send_email(recipients: list, subject: str, body: str, attachment: str = None):
    for recipient in recipients:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = recipient
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        if attachment:
            file = open(attachment, 'rb')
            # Encode as base 64
            attach = MIMEBase('application', 'octet-stream')
            attach.set_payload((file).read())
            encoders.encode_base64(attach)
            attach.add_header('Content-Disposition', 'attachment; filename= data.csv')
            msg.attach(attach)

        text = msg.as_string()

        try:
            print('Connecting to server...')
            # Create a connection with smtplib
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(EMAIL_ADDRESS, PASSWORD)
            print('Connection established!')
            print(f'Sending email to {recipient}')
            server.sendmail(EMAIL_ADDRESS, recipient, text)
            print('Email successfully sent!')
        except Exception as e:
            print(e)

    server.quit() # Close the thing once done
