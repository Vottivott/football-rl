from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import smtplib

import email

import sys


def send_mail_message(subject, message, tag = None):

    if tag is None:
        tag_str = ""
    else:
        tag_str = "+" + tag


    # create message object instance
    msg = MIMEMultipart()

    # setup the parameters of the message
    password = "BeepBoop01"
    msg['From'] = "hannesprogram@gmail.com"
    msg['To'] = "hannes.von.essen" + tag_str + "@gmail.com"
    msg['Subject'] = subject

    # add in the message body
    msg.attach(MIMEText(message, 'plain'))

    try:
        # create server
        server = smtplib.SMTP('smtp.gmail.com: 587')

        server.starttls()

        # Login Credentials for sending the mail
        server.login(msg['From'], password)

        # send the message via the server.
        server.sendmail(msg['From'], msg['To'], msg.as_string())

        server.quit()

        print("Successfully sent email to %s" % (msg['To']))
    except:
        print("Unexpected error when sending email: " + str(sys.exc_info()[0]))


def send_mail_message_with_image(subject, message, image, tag = None, image_title=None):

    if tag is None:
        tag_str = ""
    else:
        tag_str = "+" + tag


    # create message object instance
    msg = MIMEMultipart()

    # setup the parameters of the message
    password = "BeepBoop01"
    msg['From'] = "hannesprogram@gmail.com"
    msg['To'] = "hannes.von.essen" + tag_str + "@gmail.com"
    msg['Subject'] = subject

    # add in the message body
    msg.attach(MIMEText(message, 'plain'))

    try:
        fp = open(image, 'rb')
        msgImage = MIMEImage(fp.read())
        fp.close()
        if image_title is not None:
            msgImage.add_header("Content-Disposition", "attachment", filename=image_title)
        msg.attach(msgImage)
    except IOError:
        print("Couldn't include image '" + image + "'.")
        msg.attach(MIMEText("\n\nCouldn't include image '" + image + "'.", 'plain'))

    try:
        # create server
        server = smtplib.SMTP('smtp.gmail.com: 587')

        server.starttls()

        # Login Credentials for sending the mail
        server.login(msg['From'], password)

        # send the message via the server.
        server.sendmail(msg['From'], msg['To'], msg.as_string())

        server.quit()

        print("Successfully sent email to %s" % (msg['To']))
    except:
        print("Unexpected error when sending email: " + str(sys.exc_info()[0]))

def send_mail_message_with_attachment(subject, message, filename, tag = None, image_title=None):

    if tag is None:
        tag_str = ""
    else:
        tag_str = "+" + tag


    # create message object instance
    msg = MIMEMultipart()

    # setup the parameters of the message
    password = "BeepBoop01"
    msg['From'] = "hannesprogram@gmail.com"
    msg['To'] = "hannes.von.essen" + tag_str + "@gmail.com"
    msg['Subject'] = subject

    # add in the message body
    msg.attach(MIMEText(message, 'plain'))

    # Open PDF file in binary mode
    with open(filename, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email
    email.encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename=" + image_title + filename[-4:],
    )

    # Add attachment to message and convert message to string
    msg.attach(part)

    try:
        # create server
        server = smtplib.SMTP('smtp.gmail.com: 587')

        server.starttls()

        # Login Credentials for sending the mail
        server.login(msg['From'], password)

        # send the message via the server.
        server.sendmail(msg['From'], msg['To'], msg.as_string())

        server.quit()

        print("Successfully sent email to %s" % (msg['To']))
    except:
        print("Unexpected error when sending email: " + str(sys.exc_info()[0]))



if __name__ == "__main__":
    #send_mail_message_with_image("HEJ", "message with image", "../../frames/frame000000.bmp", image_title="Gen: 43  Score: 435")
    send_mail_message_with_attachment("HEJ", "message with video", "../../videos/test_video.mp4", image_title="Gen: 43  Score: 435")