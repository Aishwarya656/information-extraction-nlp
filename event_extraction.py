import spacy

nlp = spacy.load("en_core_web_sm")

EVENT_KEYWORDS = {
    "travel": "Travel",
    "visit": "Visit",
    "meet": "Meeting",
    "battle": "Battle",
    "research": "Research",
    "study": "Study",
    "celebrate": "Celebration",
    "sign": "Agreement",
    "discover": "Discovery",
    "write": "Writing",
    "record": "Recording",
    "live": "Residence",
    "fight": "Battle",
    "return": "Return"
}

def extract_events(text):

    doc = nlp(text)

    events = []

    for sent in doc.sents:

        event = ""
        person = ""
        place = ""
        date = ""

        for token in sent:
            if token.lemma_.lower() in EVENT_KEYWORDS:
                event = EVENT_KEYWORDS[token.lemma_.lower()]
                break

        for ent in sent.ents:

            if ent.label_ == "PERSON":
                person = ent.text

            elif ent.label_ in ["GPE", "LOC", "ORG", "FAC"]:
                if place == "":
                    place = ent.text

            elif ent.label_ in ["DATE", "TIME"]:
                date = ent.text

        if event:
            events.append((event, person, place, date))

    return events