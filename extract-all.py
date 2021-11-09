import sys, importlib
from parser import parse_emails


def toCSV(cfps):
    with open('output.csv', 'w') as csvfile:
        fieldnames = ['Event', 'Location', 'Date', 'Submission deadline', 'Notification deadline']
        csvfile.write(', '.join(fieldnames) + "\n")
        for cfp in cfps:
            csvfile.write(str(cfp) + "\n")


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 extract-all.py [dir_name]")
        sys.exit(1)

    extract_cfp = importlib.import_module("extract-CFP")
    emails = parse_emails(sys.argv[1])
    cfps = [extract_cfp.get_CFP(emails[i]) for i in range(len(emails))]

    toCSV(cfps)

if __name__ == "__main__":
    main()
