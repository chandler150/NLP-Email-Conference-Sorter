import sys, nltk, re, spacy
from parser import parse_email, parse_content
from nltk import word_tokenize
from spacy import displacy


SPACY_NLP = spacy.load("en_core_web_sm")

MONTHS = {'january', 'february', 'march', 'april', 'may', 'june', 'july',
        'august', 'september', 'october', 'november', 'december'}

MONTH_ABBREVS = {month[:3]: month[0].upper() + month[1:] for month in MONTHS}


class CFP:
    def __init__(self, event=None, location=None, event_date=None,
            submission_deadline=None, notification_deadline=None):
        self.event = event
        self.location = location
        self.event_date = event_date
        self.submission_deadline = submission_deadline
        self.notification_deadline = notification_deadline

    def __str__(self):
        return "{}, {}, {}, {}, {}".format(
                self.event, self.location, self.event_date,
                self.submission_deadline, self.notification_deadline
                )


# Also returns the index - n where needles were found
def get_n_previous_words(needles, haystack, n):
    prev_words = []
    found = False
    i = 0

    while i < len(haystack) and not found:
        j = 0
        while j < len(needles) and needles[j] == haystack[i + j]:
            j += 1

        if j == len(needles):
            found = True
        else:
            i += 1

    if found:
        while i > 0 and n > 0:
            i -= 1
            n -= 1
            prev_words.append(haystack[i])

    prev_words.reverse()

    return prev_words, i


def classify_date(context_words, order):
    key_words = {'submis': 'submission', 'submit': 'submission',
            'notifi': 'notification', 'notify': 'notification'}
    context_ents = SPACY_NLP(' '.join(context_words)).ents
    context_labels = [ent.label_ for ent in context_ents]
    context_words = [word.lower() for word in context_words]

    # If a word in the context words matches one of the
    # key words, it's likely the classification of this date.
    iterator = reversed if order == 'reverse' else iter
    for word in iterator(context_words):
        for key_word in key_words:
            if key_word in word:
                return key_words[key_word]

    # If no word matched a key word, it may still be an important
    # deadline that we're not interested in gathering but still
    # benefit from knowing.
    for word in iterator(context_words):
        if word == 'deadline':
            return 'deadline'

    # If we're looking at labels, we're assuming it's a date
    # referring to the label and we take the org, event, or gpe
    # to be the most relevant.
    if len(context_labels) > 0:
        event_labels = [label for label in context_labels
                if label in ('ORG', 'EVENT', 'GPE')]
        if len(event_labels) > 0:
            return event_labels[0].lower()
        else:
            return context_labels[0].lower()

    return 'unknown'


def normalize_date(date):
    month, day, year = None, None, None

    for token in date:
        token = token.lower()

        if not is_numeric(token) and month is None: # Assume month
            if token in MONTH_ABBREVS:
                month = MONTH_ABBREVS[token]
            elif token[:-1] in MONTH_ABBREVS:
                month = MONTH_ABBREVS[token[:-1]]
            elif token in MONTHS:
                month = token[0].upper() + token[1:]
            elif token[:-1] in MONTHS:
                month = token[0].upper() + token[1:-1]
        elif is_numeric(token):
            seperators = [char for char in token if char in ('-', '/')]
            if len(seperators) == 0:
                if len(token) == 4: # Assume year or day like '31st'
                    if is_numeric(token[-1]) and year is None:
                        year = token
                    elif not is_numeric(token[-1]) and day is None:
                        day = ''.join([char for char in token
                            if is_strictly_numeric(char)])
                elif 1 <= len(token) <= 2 and day is None: # Assume day
                    day = token
                elif len(token) == 3 and day is None: # Possible day like '1st'
                    if is_strictly_numeric(token[:1]):
                        day = token[:1]
            elif len(seperators) == 1: # Most likely a range of days, pick first
                token = re.split('(-|/)+', token)[0]
                if 1 <= len(token) <= 2 and is_strictly_numeric(token) and day is None:
                    day = token
            elif len(seperators) == 2: # Most likely mm/dd/yyyy or similar
                left, middle, right = (part for part in re.split('(-|/)', token)
                        if part not in ('/', '-'))

                if len(left) == 4: # Assume year, month, day
                    year, month, day = left, middle, right
                elif len(right) == 4:
                    if int(middle) > 12: # Assume month, day, year
                        month, day, year = left, middle, right
                    elif int(left) > 12: # Assume day, month, year
                        day, month, year = left, middle, right
                    else: # Assume month, day, year
                        month, day, year = left, middle, right
                else: # Assume month, day, year
                    month, day, year = left, middle, right

    if day is not None:
        day = ''.join([char for char in day if is_strictly_numeric(char)])

    return [month, int(day) if day is not None else None, year]


def update_conf_dates(date, classification, conf_dates):
    indices = {
            'event': 0, 'org': 0, 'gpe': 0, 'unknown': 0,
            'submission': 1,
            'notification': 2
            }

    if classification in indices:
        index = indices[classification]
        if (conf_dates[index] is None or
                missing_component_count(conf_dates[index]) >
                missing_component_count(date)):
            conf_dates[index] = date


# State refers the which direction we should look
# for context surrounding what a particular date
# is referring to.
# ie.
#   Submission date: Jan. 01 2001
#   vs.
#   Jan. 01 2001: Submission date
def get_state(left, right):
    if left in ('submission', 'notification', 'deadline'):
        return 'left'

    if right in ('submission', 'notification', 'deadline'):
        return 'right'

    return 'unknown'


def parse_date(date, conf_dates, alphanumerics, state):
    n = 10
    previous_words, index = get_n_previous_words(date, alphanumerics, n)
    later_words = []
    index = index + n + len(date)
    if index < len(alphanumerics):
        later_words = alphanumerics[index:index + n]

    left_classification = classify_date(previous_words, 'reverse')
    right_classification = classify_date(later_words, 'forward')

    if state == 'unknown':
        state = get_state(left_classification, right_classification)

    # Potentially moved to a different section of the email which follows a
    # different format, so we should update the state.
    if ((state == 'left' and left_classification not in
            ('submission', 'notification', 'deadline')) or
            (state == 'right' and right_classification not in
            ('submission', 'notification', 'deadline'))):
        state = get_state(left_classification, right_classification)

    if state == 'right':
        classification = right_classification
    else:
        classification = left_classification

    date = normalize_date(date)
    update_conf_dates(date, classification, conf_dates)

    return state


def get_conf_dates(content):
    conf_dates = [None, None, None] # event, submission, notification

    sentences = get_sentences(content)
    words = [word for sentence in sentences for word in sentence]
    alphanumerics = [word for word in words if is_alphanumeric(word)]
    state = 'unknown'

    for date in get_dates(content):
        date = [token for token in nltk.word_tokenize(date) if
            is_alphanumeric(token)]
        state = parse_date(date, conf_dates, alphanumerics, state)

    # If we're still missing some dates, use fallback primitive method
    if (conf_dates[0] is None or
            conf_dates[1] is None or
            conf_dates[2] is None):
        for date in get_dates_primitive(alphanumerics):
            state = parse_date(date, conf_dates, alphanumerics, state)

    update_years(conf_dates, alphanumerics)

    return [format_date(date) if date is not None else None
            for date in conf_dates]


# If we didn't find a year for a date, it's a pretty safe bet to assign it
# the most recent year found in the email.
def update_years(conf_dates, alphanumerics):
    years = [date[2] for date in conf_dates if date is not None and date[2] is not None]
    if len(years) == 0:
        years = [year for year in alphanumerics if is_year(year)]
    most_recent_year = max(years) if len(years) > 0 else None

    for date in conf_dates:
        if date is not None and date[2] is None:
            date[2] = most_recent_year


def classify_location(text):
    return

def classify_name(text):
    previous_ents = SPACY_NLP(' '.join(text)).ents
    previous_labels = {ent.label_ for ent in previous_ents}
    previous_words = text


    if 'EVENT' in previous_labels:
        return 'event'
    if 'international' in previous_words:
        return 'event'
    if 'conference' in previous_words:
        return 'event'
    return 'unknown'

def get_name_method1(alphanumerics):
    splash = 15                                    # this variable is number of words for/back of
                                                    # abbreviation we will look for name initially
    pattern = re.compile('''
            [A-Z^=:]{2,}[" "'^=:]?\d{2,}          # LL'NN, LLNN, LL NN
            | \d{2,}[" "'^=:]?[A-z^=:]{2,}        # NN'LL, NNLL, NN LL #Problem, 2019, APCEAS is two separate tokens
            | [A-Z^:=]{2,}['-^=:]+[A-Z^:=]{2,}''', re.VERBOSE)
    for i in alphanumerics[0:100]:
        x = re.search(pattern, i)
        if x is not None:
            ind = alphanumerics.index(i)
            hm = splash+1
            for l in alphanumerics[(ind-splash):ind]:
                hm -=1
                y = re.search("^[Tt]{1}he$", l)             #Perhaps replace with findall
                if y is not None:
                    num = (ind-hm)
                    return alphanumerics[num:ind], 12       #HIGH PRIORITY,have start & end
            hm = -1
            for m in alphanumerics[ind:ind+splash]:
                hm += 1
                z = re.search("[Tt]he", m)
                if z is not None:
                    num = (ind+hm)
                    return alphanumerics[num:num+splash], 10    #MID PRIORITY, have start, ukn end

    bigrams = []
    for i in nltk.bigrams(alphanumerics[0:100]):
        bigrams.append(" ".join(i))

    #If no key pattern matches are found in single parse of alphanumerics, it performs a bigram search
    for i in bigrams:
        x = re.search(pattern, i)
        if x is not None:
            ind = bigrams.index(i)

            #This block searches for "The prior to a pattern match of bigrams
            hm = splash + 1
            for l in alphanumerics[(ind-splash):ind]:
                hm -=1
                y = re.search("^[Tt]{1}he$", l)
                if y is not None:
                    num = (ind-hm)
                    return alphanumerics[num:ind], 12       #HIGH PRIORITY,have start & end

            hm = -1
            for m in alphanumerics[ind:ind+splash]:
                hm += 1
                z = re.search("[Tt]he", m)
                if z is not None:
                    num = (ind+hm)
                    return alphanumerics[num:num+splash], 10    #MID PRIORITY, have start, ukn end
            return alphanumerics[ind:ind+splash], 8           #low priority, have end, no start

    return [], 0

def get_dates(text):
    return [ent.text for ent in SPACY_NLP(text).ents if ent.label_ == 'DATE']


def get_dates_primitive(words):
    dates = []

    for i in range(len(words)):
        date = []
        word = words[i].lower()
        shortened_word = word[:-1]
        if (is_year(word) or
                word in MONTHS or
                word in MONTH_ABBREVS or
                shortened_word in MONTHS or
                shortened_word in MONTH_ABBREVS):
            if i > 0:
                date.append(words[i - 1])
            date.append(words[i])
            if i < len(words) - 1:
                date.append(words[i + 1])

            dates.append(date)

    return dates

def get_everything_else(text):
    tokens = SPACY_NLP(text)
    # This gives "Entities" Which are like core concepts, The title is one, so is the Acronym, and city.
    return [ent.text for ent in tokens.ents]

def get_ents(text):
    tokens = SPACY_NLP(text)
    return [ent for ent in tokens.ents]


def get_sentences(text):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    return [[token for token in nltk.word_tokenize(sentence)]
            for sentence in sent_tokenizer.tokenize(text)]


def missing_component_count(date):
    return len([part for part in date if part is None])


def format_date(date):
    return '{} {}, {}'.format(*date)


def is_alphanumeric(text):
    return re.match("[a-zA-Z0-9]+", text) is not None


def is_numeric(text):
    return re.match("[0-9]+(st|nd|rd|th)*", text) is not None


def is_strictly_numeric(text):
    return re.match('^([0-9]+)$', text) is not None


def is_year(text):
    return re.match('^([1-2][0-9]{3})$', text) is not None


def prioritizer(Pos_Name, content, alphanumerics):
    #Priority of Names, Assesses if first & last are known, or just first or just last or neither,
    #Continues assessment of name based on such.
    Name = Pos_Name[0]
    Priority = Pos_Name[1]

    if 12 > Priority > 8:
        Pos_Name = MidHighPri(Name, Priority, content, alphanumerics)
    if Priority < 9:
        Pos_Name = LowPri(Name, Priority)
    return Pos_Name

def MidHighPri(Name, Priority, content, alphanumerics):
    # Some count variables to use later
    i = -1
    count = 0
    dict = {}

    for entity in get_everything_else(content)[0:100]:
        i += 1
        label = [token for token in nltk.word_tokenize(entity) if
                 is_alphanumeric(token)]
        previous_words = get_n_previous_words(label, alphanumerics, 15)

        dict[i] = 0
    for n in Name:
        for l in label:
            if n == l:
                dict[i] += 1
    if max(dict, key=dict.get) > 4:
        Name = [get_everything_else(content)[max(dict, key=dict.get)]]
        Priority = 7
    return [Name, Priority]

def LowPri(Name, Priority):
    # Implementation for keyword search, more work needed
    any = ['International', 'World', 'Symposium', 'Annual', 'Congress']
    other = ['Science', 'Applied', 'Computing', 'Computers', 'Engineering', ]
    for n in any:
        if n in Name:
            yup = Name.index(n)
            del Name[:yup - 1]
            Priority = 11
    return Name, Priority

def get_conf_name(alphanumerics, content):
    Pos_Name = get_name_method1(alphanumerics)
    Pos_Name = prioritizer(Pos_Name, content, alphanumerics)
    Name = Pos_Name[0]
    Priority = Pos_Name[1]
    #print("Name:", Name)
    saved_Name = ' '.join(Name)
    #print("saved",saved_Name)
    return [saved_Name]

def get_location(alphanumerics, content):
    pos_locs = []
    doc = SPACY_NLP(content).ents
    count = 0
    for ent in doc:
        if ent.label_ == "GPE":
            if count <3:
                count +=1
                pos_locs.append(ent.text)
    #Arbitrary large number, look to reformat
    first = 999999999999
    if len(pos_locs) > 0:
        tryloc = pos_locs[0].split()
        for i in alphanumerics:
            for l in tryloc:
                if i == l:
                    first = alphanumerics.index(i)
            count += 1
        if len(alphanumerics) > first+2:
            return [' '.join(alphanumerics[(first-1):(first+2)])]
        return [pos_locs[0]]
    return ["Las Vegas"]


def get_CFP(email):
    content = parse_content(email)
    sentences = get_sentences(content)
    words = [word for sentence in sentences for word in sentence]
    alphanumerics = [word for word in words if is_alphanumeric(word)]

    return CFP(*get_conf_name(alphanumerics,content), *get_location(alphanumerics, content), *get_conf_dates(content))


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 extract-CFP.py [file_name]")
        sys.exit(1)

    email = parse_email(sys.argv[1])
    cfp = get_CFP(email)

    print("\n" + str(cfp.event))
    print("\n" + str(cfp.location))
    print("\n" + str(cfp.event_date))
    print("\n" + str(cfp.submission_deadline))
    print("\n" + str(cfp.notification_deadline) + "\n")


if __name__ == "__main__":
    main()
