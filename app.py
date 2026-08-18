from flask import Flask, render_template

from dataset_reader import load_dataset, get_total_files
from nlp_pipeline import named_entity_recognition, pos_tagging
from relation_extraction import extract_relations
from event_extraction import extract_events
from temporal_analysis import extract_temporal_information
from knowledge_graph import generate_knowledge_graph

app = Flask(__name__)


@app.route("/")
def home():

    text = load_dataset()

    entities = named_entity_recognition(text)

    pos_tags = pos_tagging(text)

    relations = extract_relations(text)

    events = extract_events(text)

    temporal = extract_temporal_information(text)

    generate_knowledge_graph(relations)

    total_files = get_total_files()

    total_words = len(text.split())

    total_entities = len(entities)

    total_pos = len(pos_tags)

    total_relations = len(relations)

    total_events = len(events)

    total_temporal = len(temporal)

    total_sentences = len([
        s for s in text.split(".")
        if s.strip()
    ])

    return render_template(

        "result.html",

        text=text,

        entities=entities,

        pos_tags=pos_tags,

        relations=relations,

        events=events,

        temporal=temporal,

        total_files=total_files,

        total_words=total_words,

        total_entities=total_entities,

        total_pos=total_pos,

        total_relations=total_relations,

        total_events=total_events,

        total_temporal=total_temporal,

        total_sentences=total_sentences

    )


if __name__ == "__main__":
    app.run(debug=True)