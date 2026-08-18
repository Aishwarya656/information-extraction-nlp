import spacy

nlp = spacy.load("en_core_web_sm")

def extract_temporal_information(text):

    doc = nlp(text)

    temporal = []

    for sent in doc.sents:

        for ent in sent.ents:

            if ent.label_ in ["DATE", "TIME"]:

                temporal.append(
                    (
                        ent.text,
                        sent.text
                    )
                )

    return temporal