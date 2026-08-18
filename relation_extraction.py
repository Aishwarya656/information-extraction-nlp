import spacy

nlp = spacy.load("en_core_web_sm")

def extract_relations(text):
    doc = nlp(text)

    relations = []

    for sent in doc.sents:

        subject = ""
        relation = ""
        obj = ""

        for token in sent:

            if token.dep_ in ("nsubj", "nsubjpass"):
                subject = token.text

            elif token.pos_ == "VERB":
                relation = token.lemma_

            elif token.dep_ in ("dobj", "pobj", "attr", "dative"):
                obj = token.text

        if subject and relation and obj:
            relations.append((subject, relation, obj))

    return relations