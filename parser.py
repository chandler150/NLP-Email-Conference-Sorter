import sys, os
from os import listdir
import warnings
import eml_parser
from bs4 import BeautifulSoup as soup



warnings.filterwarnings("ignore", category=UserWarning, module='bs4')


def parse_email(file_name):
    try:
        with open(file_name, 'rb') as email_file:
            raw_email = email_file.read()
            parser = eml_parser.EmlParser(include_raw_body=True)
            parsed_email = parser.decode_email_bytes(raw_email)

            return parsed_email
    except IOError as err:
        print("Error attempting to read {}!".format(file_name))
        print(err)
        sys.exit(1)


def parse_emails(directory_path):
    file_names = sorted([file_name for file_name in listdir(directory_path)])

    return [parse_email(directory_path + '/' + name) for name in file_names]


# Content of the email
def parse_content(email):
    return soup(
            email['body'][0]['content'],
            features="html5lib"
            ).get_text().strip()


# Subject: <subject>
def parse_subject(email):
    return email['header']['subject']


# All emails in CC:, From:, and To:
def parse_email_addresses(email):
    if 'email' in email['body'][0]:
        return email['body'][0]['email']
    if 'domain' in email['body'][0]:
        return email['body'][0]['domain']

    if 'email' in email['body'][1]:
        return email['body'][1]['email']
    if 'domain' in email['body'][1]:
        return email['body'][1]['domain']

    return []


def main():
    if len(sys.argv) != 2:
        print("Usage: python3.7+ parser.py [file_name]")
        sys.exit(1)

    email = parse_email(sys.argv[1])
    print("Addresses:\n", parse_email_addresses(email), "\n")
    print("Subject:\n", parse_subject(email), "\n")
    print("Content:\n", parse_content(email), "\n")


if __name__ == "__main__":
    main()
