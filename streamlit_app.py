import os
import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components

from dataset_reader import load_dataset, get_total_files
from nlp_pipeline import named_entity_recognition, pos_tagging
from relation_extraction import extract_relations
from event_extraction import extract_events
from temporal_analysis import extract_temporal_information


st.set_page_config(
    page_title="Information Extraction System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 5% 5%, rgba(99,102,241,0.15), transparent 28%),
        radial-gradient(circle at 95% 15%, rgba(168,85,247,0.13), transparent 30%),
        linear-gradient(135deg, #f8fafc, #eef2ff, #faf5ff);
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #111827 0%,
        #1e1b4b 45%,
        #312e81 100%
    );
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] .stRadio label {
    padding: 9px 12px;
    border-radius: 10px;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.12);
}

.main-title {
    text-align: center;
    font-size: 45px;
    font-weight: 800;
    background: linear-gradient(90deg,#1e3a8a,#4f46e5,#7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 5px;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 16px;
    margin-bottom: 30px;
}

.hero {
    padding: 28px;
    border-radius: 22px;
    background: linear-gradient(135deg,#1e3a8a,#4f46e5,#7c3aed);
    color: white;
    box-shadow: 0 15px 35px rgba(79,70,229,0.22);
    margin-bottom: 25px;
}

.hero h2 {
    color: white !important;
}

.hero p {
    color: #e0e7ff;
    line-height: 1.7;
}

.stat-card {
    position: relative;
    overflow: hidden;
    border-radius: 20px;
    padding: 20px;
    color: white;
    min-height: 135px;
    box-shadow: 0 12px 30px rgba(15,23,42,0.14);
    transition: 0.3s;
}

.stat-card:hover {
    transform: translateY(-6px);
}

.stat-card h1 {
    color: white !important;
    font-size: 34px;
    margin: 7px 0;
}

.stat-card p {
    color: white;
    margin: 0;
    font-size: 14px;
}

.blue {
    background: linear-gradient(135deg,#2563eb,#1d4ed8);
}

.purple {
    background: linear-gradient(135deg,#7c3aed,#6d28d9);
}

.pink {
    background: linear-gradient(135deg,#db2777,#be185d);
}

.green {
    background: linear-gradient(135deg,#059669,#047857);
}

.orange {
    background: linear-gradient(135deg,#ea580c,#c2410c);
}

.red {
    background: linear-gradient(135deg,#dc2626,#b91c1c);
}

.teal {
    background: linear-gradient(135deg,#0f766e,#115e59);
}

.indigo {
    background: linear-gradient(135deg,#4338ca,#3730a3);
}

.section-card {
    background: rgba(255,255,255,0.78);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 25px;
    margin-top: 25px;
    border: 1px solid rgba(99,102,241,0.12);
    box-shadow: 0 10px 30px rgba(15,23,42,0.07);
}

[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
    border: 1px solid #c7d2fe;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 10px;
    border: none;
    font-weight: 600;
    background: linear-gradient(135deg,#4f46e5,#7c3aed);
    color: white;
}

.stTextInput input,
.stTextArea textarea {
    border-radius: 12px;
    border: 1px solid #c7d2fe;
}

.footer {
    margin-top: 50px;
    padding: 28px;
    text-align: center;
    border-radius: 20px;
    background: rgba(255,255,255,0.7);
    color: #64748b;
}

.footer h3 {
    color: #3730a3;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA PROCESSING FUNCTION
# =========================================================

def process_text(input_text):

    extracted_entities = named_entity_recognition(input_text)

    extracted_pos = pos_tagging(input_text)

    extracted_relations = extract_relations(input_text)

    extracted_events = extract_events(input_text)

    extracted_temporal = extract_temporal_information(input_text)

    return (
        extracted_entities,
        extracted_pos,
        extracted_relations,
        extracted_events,
        extracted_temporal
    )


# =========================================================
# DEFAULT DATASET
# =========================================================

default_text = load_dataset()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    """
    <div style="
        text-align:center;
        padding:15px;
        font-size:25px;
        font-weight:800;
    ">
        🤖 NLP SYSTEM
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div style="
        text-align:center;
        color:#c7d2fe;
        margin-bottom:20px;
    ">
        Old Monk Dataset
    </div>
    """,
    unsafe_allow_html=True
)


uploaded_file = st.sidebar.file_uploader(
    "📤 Upload TXT File",
    type=["txt"]
)


if uploaded_file is not None:

    try:

        uploaded_text = uploaded_file.read().decode(
            "utf-8",
            errors="ignore"
        )

        text = uploaded_text

        st.sidebar.success(
            "Uploaded file loaded!"
        )

    except Exception:

        text = default_text

        st.sidebar.error(
            "Unable to read uploaded file."
        )

else:

    text = default_text


(
    entities,
    pos_tags,
    relations,
    events,
    temporal
) = process_text(text)


total_files = get_total_files()

total_words = len(text.split())

total_sentences = len([
    s for s in text.split(".")
    if s.strip()
])


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📚 Dataset",
        "🔍 Entity Explorer",
        "🏷 Named Entities",
        "🔤 POS Tags",
        "🔗 Relations",
        "📅 Events",
        "⏰ Temporal Analysis",
        "🌐 Knowledge Graph",
        "📊 Statistical Dashboard"
    ]
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    "<div class='main-title'>🤖 Information Extraction System</div>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='subtitle'>
    NLP Based Named Entity Recognition • POS Tagging •
    Relation Extraction • Event Extraction • Temporal Analysis
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.markdown(
        """
        <div class="hero">
            <h2>🧠 Intelligent NLP Analytics</h2>
            <p>
            Extract and analyze meaningful information from
            the Old Monk dataset using multiple NLP techniques.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    cards = [
        ("📁", total_files, "Dataset Files", "blue"),
        ("📝", total_words, "Total Words", "purple"),
        ("🏷️", len(entities), "Named Entities", "pink"),
        ("🔤", len(pos_tags), "POS Tags", "green")
    ]

    for col, data in zip([c1,c2,c3,c4], cards):

        icon, value, label, color = data

        with col:

            st.markdown(
                f"""
                <div class="stat-card {color}">
                    <div style="font-size:25px;">{icon}</div>
                    <h1>{value}</h1>
                    <p>{label}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    c5, c6, c7, c8 = st.columns(4)

    cards2 = [
        ("🔗", len(relations), "Relations", "orange"),
        ("📅", len(events), "Events", "red"),
        ("⏰", len(temporal), "Temporal Data", "teal"),
        ("📖", total_sentences, "Sentences", "indigo")
    ]

    for col, data in zip([c5,c6,c7,c8], cards2):

        icon, value, label, color = data

        with col:

            st.markdown(
                f"""
                <div class="stat-card {color}">
                    <div style="font-size:25px;">{icon}</div>
                    <h1>{value}</h1>
                    <p>{label}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    entity_df = pd.DataFrame(
        entities,
        columns=["Entity", "Type"]
    )

    pos_df = pd.DataFrame(
        pos_tags,
        columns=["Word", "POS"]
    )

    col1, col2 = st.columns(2)

    with col1:

        if not entity_df.empty:

            counts = entity_df["Type"].value_counts()

            fig = px.pie(
                values=counts.values,
                names=counts.index,
                hole=0.5,
                title="🏷 Named Entity Distribution",
                color_discrete_sequence=px.colors.qualitative.Bold
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col2:

        if not pos_df.empty:

            counts = pos_df["POS"].value_counts().head(10)

            fig = px.bar(
                x=counts.index,
                y=counts.values,
                color=counts.index,
                title="🔤 Top POS Tags",
                labels={
                    "x": "POS Tag",
                    "y": "Frequency"
                },
                color_discrete_sequence=px.colors.qualitative.Vivid
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# DATASET
# =========================================================

elif page == "📚 Dataset":

    st.header("📚 Old Monk Dataset")

    st.success(
        "Dataset loaded successfully."
    )

    search_text = st.text_input(
        "🔎 Search inside dataset"
    )

    if search_text:

        lower_text = text.lower()
        lower_search = search_text.lower()

        if lower_search in lower_text:

            st.success(
                f"'{search_text}' was found in the dataset."
            )

            position = lower_text.find(
                lower_search
            )

            start = max(
                0,
                position - 100
            )

            end = min(
                len(text),
                position + len(search_text) + 150
            )

            st.code(
                text[start:end]
            )

        else:

            st.warning(
                f"'{search_text}' was not found."
            )

    st.text_area(
        "📄 Dataset Text",
        text,
        height=500
    )


# =========================================================
# ENTITY EXPLORER
# =========================================================

elif page == "🔍 Entity Explorer":

    st.header("🔍 Entity Explorer")

    st.write(
        "Select an entity to view its extracted information."
    )

    entity_df = pd.DataFrame(
        entities,
        columns=["Entity", "Type"]
    )

    if entity_df.empty:

        st.warning(
            "No entities found."
        )

    else:

        entity_names = sorted(
            entity_df["Entity"].dropna().unique().tolist()
        )

        selected_entity = st.selectbox(
            "Select Entity",
            entity_names
        )

        selected_lower = selected_entity.lower()


        # ENTITY TYPE

        matching_entities = entity_df[
            entity_df["Entity"].str.lower() == selected_lower
        ]

        st.subheader("🏷 Entity Information")

        st.dataframe(
            matching_entities,
            use_container_width=True
        )


        # RELATIONS

        st.subheader("🔗 Related Information")

        related_rows = []

        for relation in relations:

            if len(relation) >= 3:

                subject = str(relation[0])
                relation_name = str(relation[1])
                obj = str(relation[2])

                if (
                    selected_lower in subject.lower()
                    or
                    selected_lower in obj.lower()
                ):

                    related_rows.append(
                        [
                            subject,
                            relation_name,
                            obj
                        ]
                    )

        if related_rows:

            relation_explorer_df = pd.DataFrame(
                related_rows,
                columns=[
                    "Subject",
                    "Relation",
                    "Object"
                ]
            )

            st.dataframe(
                relation_explorer_df,
                use_container_width=True
            )

        else:

            st.info(
                "No direct relation found for this entity."
            )


        # EVENTS

        st.subheader("📅 Related Events")

        related_events = []

        for event in events:

            event_values = [
                str(value)
                for value in event
                if value is not None
            ]

            if any(
                selected_lower in value.lower()
                for value in event_values
            ):

                related_events.append(
                    event
                )

        if related_events:

            event_explorer_df = pd.DataFrame(
                related_events,
                columns=[
                    "Event",
                    "Person",
                    "Place",
                    "Date"
                ]
            )

            st.dataframe(
                event_explorer_df,
                use_container_width=True
            )

        else:

            st.info(
                "No direct event found for this entity."
            )


        # TEXT OCCURRENCES

        st.subheader("📖 Text Occurrences")

        sentences = [
            sentence.strip()
            for sentence in text.split(".")
            if sentence.strip()
        ]

        matching_sentences = [
            sentence
            for sentence in sentences
            if selected_lower in sentence.lower()
        ]

        if matching_sentences:

            for sentence in matching_sentences:

                st.info(
                    sentence + "."
                )

        else:

            st.info(
                "No sentence occurrence found."
            )


# =========================================================
# NER
# =========================================================

elif page == "🏷 Named Entities":

    st.header("🏷 Named Entity Recognition")

    entity_df = pd.DataFrame(
        entities,
        columns=["Entity", "Type"]
    )

    search = st.text_input(
        "🔍 Search Entity"
    )

    if search:

        entity_df = entity_df[
            entity_df["Entity"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    st.dataframe(
        entity_df,
        use_container_width=True,
        height=450
    )

    if not entity_df.empty:

        counts = entity_df["Type"].value_counts()

        fig = px.pie(
            values=counts.values,
            names=counts.index,
            hole=0.5,
            title="🏷 Entity Type Distribution",
            color_discrete_sequence=px.colors.qualitative.Bold
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.download_button(
        "📥 Download NER CSV",
        entity_df.to_csv(index=False),
        "named_entities.csv",
        "text/csv"
    )


# =========================================================
# POS
# =========================================================

elif page == "🔤 POS Tags":

    st.header("🔤 Part Of Speech Tagging")

    pos_df = pd.DataFrame(
        pos_tags,
        columns=["Word", "POS"]
    )

    st.dataframe(
        pos_df,
        use_container_width=True,
        height=500
    )

    counts = pos_df["POS"].value_counts().head(15)

    fig = px.bar(
        x=counts.index,
        y=counts.values,
        color=counts.index,
        title="🔤 POS Tag Distribution",
        labels={
            "x": "POS Tag",
            "y": "Frequency"
        },
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.download_button(
        "📥 Download POS CSV",
        pos_df.to_csv(index=False),
        "pos_tags.csv",
        "text/csv"
    )


# =========================================================
# RELATIONS
# =========================================================

elif page == "🔗 Relations":

    st.header("🔗 Relation Extraction")

    relation_df = pd.DataFrame(
        relations,
        columns=[
            "Subject",
            "Relation",
            "Object"
        ]
    )

    st.dataframe(
        relation_df,
        use_container_width=True,
        height=450
    )

    counts = relation_df["Relation"].value_counts()

    fig = px.bar(
        x=counts.index,
        y=counts.values,
        color=counts.index,
        title="🔗 Relation Distribution",
        labels={
            "x": "Relation",
            "y": "Frequency"
        },
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.download_button(
        "📥 Download Relations CSV",
        relation_df.to_csv(index=False),
        "relations.csv",
        "text/csv"
    )


# =========================================================
# EVENTS
# =========================================================

elif page == "📅 Events":

    st.header("📅 Event Extraction")

    event_df = pd.DataFrame(
        events,
        columns=[
            "Event",
            "Person",
            "Place",
            "Date"
        ]
    )

    st.dataframe(
        event_df,
        use_container_width=True,
        height=450
    )

    counts = event_df["Event"].value_counts()

    fig = px.pie(
        values=counts.values,
        names=counts.index,
        hole=0.5,
        title="📅 Event Distribution",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.download_button(
        "📥 Download Events CSV",
        event_df.to_csv(index=False),
        "events.csv",
        "text/csv"
    )


# =========================================================
# TEMPORAL
# =========================================================

elif page == "⏰ Temporal Analysis":

    st.header("⏰ Temporal Analysis")

    temporal_df = pd.DataFrame(
        temporal,
        columns=[
            "Date / Time",
            "Sentence"
        ]
    )

    st.dataframe(
        temporal_df,
        use_container_width=True,
        height=450
    )

    st.download_button(
        "📥 Download Temporal CSV",
        temporal_df.to_csv(index=False),
        "temporal_data.csv",
        "text/csv"
    )


# =========================================================
# KNOWLEDGE GRAPH
# =========================================================

elif page == "🌐 Knowledge Graph":

    st.header("🌐 Knowledge Graph")

    graph_file = os.path.join(
        "static",
        "knowledge_graph.html"
    )

    if os.path.exists(graph_file):

        with open(
            graph_file,
            "r",
            encoding="utf-8"
        ) as file:

            graph_html = file.read()

        components.html(
            graph_html,
            height=700,
            scrolling=True
        )

    else:

        st.warning(
            "Knowledge Graph file was not found."
        )


# =========================================================
# STATISTICAL DASHBOARD
# =========================================================

elif page == "📊 Statistical Dashboard":

    st.header("📊 Statistical Dashboard")

    st.markdown(
        """
        <div class="hero">
            <h2>📈 NLP Statistics & Analytics</h2>
            <p>
            Visual analysis of information extracted
            from the Old Monk dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📝 Words", total_words)
    c2.metric("🏷 Entities", len(entities))
    c3.metric("🔗 Relations", len(relations))
    c4.metric("📅 Events", len(events))

    c5, c6, c7, c8 = st.columns(4)

    c5.metric("⏰ Temporal", len(temporal))
    c6.metric("🔤 POS Tags", len(pos_tags))
    c7.metric("📖 Sentences", total_sentences)
    c8.metric("📁 Files", total_files)

    st.divider()

    entity_df = pd.DataFrame(
        entities,
        columns=["Entity", "Type"]
    )

    entity_counts = entity_df["Type"].value_counts()

    st.subheader("🏷 Entity Statistics")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            values=entity_counts.values,
            names=entity_counts.index,
            hole=0.5,
            title="Entity Type Distribution",
            color_discrete_sequence=px.colors.qualitative.Bold
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            x=entity_counts.index,
            y=entity_counts.values,
            color=entity_counts.index,
            title="Entity Frequency",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.subheader("🔤 POS Statistics")

    pos_df = pd.DataFrame(
        pos_tags,
        columns=["Word", "POS"]
    )

    pos_counts = pos_df["POS"].value_counts().head(15)

    fig = px.bar(
        x=pos_counts.index,
        y=pos_counts.values,
        color=pos_counts.index,
        title="POS Distribution",
        labels={
            "x": "POS Tag",
            "y": "Frequency"
        },
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("🔗 Relation Statistics")

    relation_df = pd.DataFrame(
        relations,
        columns=[
            "Subject",
            "Relation",
            "Object"
        ]
    )

    relation_counts = relation_df["Relation"].value_counts()

    fig = px.bar(
        x=relation_counts.index,
        y=relation_counts.values,
        color=relation_counts.index,
        title="Relation Frequency",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("📅 Event Statistics")

    event_df = pd.DataFrame(
        events,
        columns=[
            "Event",
            "Person",
            "Place",
            "Date"
        ]
    )

    event_counts = event_df["Event"].value_counts()

    fig = px.pie(
        values=event_counts.values,
        names=event_counts.index,
        hole=0.5,
        title="Event Distribution",
        color_discrete_sequence=px.colors.qualitative.Pastel1
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("⏰ Temporal Information")

    temporal_df = pd.DataFrame(
        temporal,
        columns=[
            "Date / Time",
            "Sentence"
        ]
    )

    st.dataframe(
        temporal_df,
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

        <h3>🤖 Information Extraction System</h3>

        <p>
            Developed by <b>Aishwarya R H</b>
        </p>

        <p>
            Python • Flask • Streamlit • spaCy • Plotly •
            NetworkX • PyVis
        </p>

        <p>
            NLP Based Information Extraction using Old Monk Dataset
        </p>

    </div>
    """,
    unsafe_allow_html=True
)