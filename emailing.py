import smtplib
import imghdr
from email.message import EmailMessage

PASSWORD = "..."
SENDER = "app8flask@gmail.com"
RECEIVER = "app8flask@gmail.com"
def send_email(image_path):

    email_message = EmailMessage()
    email_message["Subject"] = "New customer showed up!"
    email_message.set_content("Hey, we just saw a new customer!")

    # rb = reading binary mode
    with open(image_path, "rb") as file:
        content = file.read()
    email_message.add_attachment(content, mainType="image", subtype=imghdr.what(None, content))

    gmail = smtplib.SMTP("smtp.gmail.com", 587)
    gmail.ehlo()
    gmail.starttls()
    gmail.login(SENDER, PASSWORD)
    gmail.sendmail(SENDER, RECEIVER, email_message.as_string())

if __name__ == "__main__":
    send_email(image_path="images/19.png")