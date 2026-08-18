\# Information-Extraction-NLP



\## Information Extraction System Using the Old Monk Text Corpus



An NLP-based Information Extraction System that processes the Old Monk text corpus and extracts structured linguistic and semantic information using POS tagging, Named Entity Recognition, relation extraction, event extraction, temporal analysis, and knowledge graph visualization.



\---



\## 1. Aim / Objective



To build an end-to-end information extraction system for the Old Monk text corpus that extracts POS tags, named entities, relationships, events, and temporal information from unstructured text and presents the results through an interactive dashboard and knowledge graph.



\---



\## 2. Expected Outcomes



\- Identify parts of speech in text.

\- Detect named entities such as people, places, organizations, and dates.

\- Extract relationships between entities.

\- Detect events and time expressions.

\- Arrange and analyze events using temporal information.



\---



\## 3. Tools \& Technologies Used



| Category | Tool / Library | Purpose |

|---|---|---|

| Language / Runtime | Python | Core implementation language |

| POS Tagging \& Parsing | spaCy | POS tagging and dependency parsing |

| Named Entity Recognition | spaCy `en\_core\_web\_sm` | Detects people, locations, organizations, dates, and other entities |

| Relation Extraction | spaCy Dependency Parser | Extracts subject–relation–object relationships |

| Event Extraction | Python + spaCy | Identifies predefined events from text |

| Temporal Analysis | spaCy | Extracts DATE and TIME expressions |

| Knowledge Graph | NetworkX | Constructs entity–relationship graph |

| Graph Visualization | PyVis | Interactive knowledge graph visualization |

| Frontend | Streamlit | Interactive NLP dashboard |

| Backend | Flask | Web application backend |

| Data Processing | Pandas | Organizes and processes extracted information |

| Visualization | Plotly | Statistical charts and analysis |



\---



\## 4. Dataset / Corpus Description



The system was developed and tested using the Old Monk text corpus containing three plain-text files.



```text

dataset/

└── old\_monk/

&#x20;   ├── book1.txt

&#x20;   ├── book2.txt

&#x20;   └── book3.txt

