import spacy
import pandas as pd
import numpy as np
import re
import ast
from pyvis.network import Network
import networkx as nx

# ------------------- SPAcY FUNCTIONS (ENTITY EXTRACTION) -------------------

def initialize_nlp_model():
    """ Load the spaCy language model. """
    return spacy.load("en_core_web_trf")

def remove_duplicates(entity_list):
    """ Remove duplicates from a list of entities, treating different cases as duplicates. """
    return list(dict.fromkeys(entity.lower().strip() for entity in entity_list if entity))

def standardize_named_entities(entity_list):
    """ Standardize named entities by converting to title case. """
    return [entity.title() for entity in entity_list]

def clean_dates(date_str):
    """ Extract and standardize valid date values. """
    if isinstance(date_str, str):
        dates = re.findall(r'\b\d{4}\b', date_str)  # Extract years (e.g., "2022", "2021")
        return sorted(set(dates)) if dates else ["Unknown"]
    return ["Unknown"]

def extract_key_actions(text, nlp_model):
    """ Extract significant actions and related noun phrases from text. """
    doc = nlp_model(text)
    key_actions = set()

    for token in doc:
        if token.pos_ == "VERB":  # Focus on verbs
            action = token.text.lower()
            associated_terms = []

            # Extract dependencies like direct objects, prepositions, etc.
            for child in token.children:
                if child.dep_ in ["dobj", "prep", "acomp", "xcomp", "advcl", "ccomp"] and child.pos_ in ["NOUN", "PROPN", "ADJ"]:
                    associated_terms.append(child.text.lower())

            # Construct action phrase
            if associated_terms:
                action_phrase = f"{action} {' '.join(associated_terms)}"
                key_actions.add(action_phrase)
            else:
                key_actions.add(action)

    return sorted(key_actions)

def extract_entities_and_key_actions(text, nlp_model):
    """ Extract specific entities and key actions from the text. """
    doc = nlp_model(text)
    entities = {
        "PERSON": [], "ORG": [], "GPE": [], "EVENT": [], "PRODUCT": [], "TOPIC": [], "DATE": [], "KEY_ACTIONS": []
    }

    # Extract named entities
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["PERSON"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["ORG"].append(ent.text)
        elif ent.label_ == "GPE":
            entities["GPE"].append(ent.text)
        elif ent.label_ == "EVENT":
            entities["EVENT"].append(ent.text)
        elif ent.label_ == "PRODUCT":
            entities["PRODUCT"].append(ent.text)
        elif ent.label_ in ["NORP", "FAC", "LOC", "LAW", "LANGUAGE"]:
            entities["TOPIC"].append(ent.text)
        elif ent.label_ == "DATE":
            entities["DATE"].append(ent.text)

    # Standardize and clean extracted entities
    for key in ["PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "TOPIC"]:
        entities[key] = standardize_named_entities(remove_duplicates(entities[key]))

    # Standardize and clean dates
    entities["DATE"] = clean_dates(", ".join(entities["DATE"]))

    # Extract and clean key actions
    entities["KEY_ACTIONS"] = extract_key_actions(text, nlp_model)

    return entities

def process_text(df, nlp_model):
    """ Process text data and extract KEY_ACTIONS & Entities. """
    processed_data = df["Text"].apply(lambda x: extract_entities_and_key_actions(str(x), nlp_model) if pd.notnull(x) else {})

    # Convert extracted data into separate DataFrame columns
    entity_df = pd.DataFrame(processed_data.tolist())

    return pd.concat([df, entity_df], axis=1)

# ------------------- GLOVE FUNCTIONS (TOPIC-BASED FILTERING) -------------------

def load_glove_model(glove_file):
    """ Loads the GloVe word embeddings into a dictionary. """
    print("Loading GloVe model...")
    glove_dict = {}
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            glove_dict[word] = vector
    print("✅ GloVe model loaded successfully!")
    return glove_dict

def cosine_similarity(vec1, vec2):
    """ Computes cosine similarity between two vectors. """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_phrase_vector(phrase, glove_dict):
    """ Converts a phrase into a vector by averaging word vectors. """
    words = phrase.lower().split()
    word_vectors = [glove_dict[word] for word in words if word in glove_dict]

    if not word_vectors:
        return None  # No valid words found in GloVe
    
    return np.mean(word_vectors, axis=0)

def is_related_to_topic(action_list, topic_vector, glove_dict, threshold=0.3):
    """ Checks if any entity in KEY_ACTIONS is related to the topic. """
    if not isinstance(action_list, list) or len(action_list) == 0:
        return False

    for action in action_list:
        action_vector = get_phrase_vector(action, glove_dict)
        if action_vector is None:
            continue
        similarity = cosine_similarity(action_vector, topic_vector)

        if similarity > threshold:
            return True  # Found a related entity

    return False

def filter_topic_related_rows(df, topic_vector, glove_dict):
    """ Filters rows where at least one entity in "KEY_ACTIONS" is related to the topic. """
    if "KEY_ACTIONS" not in df.columns:
        raise ValueError("❌ The 'KEY_ACTIONS' column is missing from the dataset!")

    df["KEY_ACTIONS"] = df["KEY_ACTIONS"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
    )

    return df[df["KEY_ACTIONS"].apply(lambda x: is_related_to_topic(x, topic_vector, glove_dict))]

# ------------------- RELATIONSHIP EXTRACTION -------------------

def extract_relationships(text, nlp_model, entities):
    """ Extract relationships between pre-extracted entities in the text. """
    doc = nlp_model(text)
    relationships = []

    for sent in doc.sents:
        entities_in_sent = [ent.text for ent in sent.ents if ent.text in entities]
        if len(entities_in_sent) > 1:
            for i in range(len(entities_in_sent) - 1):
                for j in range(i + 1, len(entities_in_sent)):
                    relationships.append((entities_in_sent[i], entities_in_sent[j]))

    return relationships

def build_relationship_graph(df, nlp_model):
    """ Build a graph of relationships between entities. """
    G = nx.Graph()

    for _, row in df.iterrows():
        text = row["Text"]
        # Combine all entities into a single list
        entities = []
        for key in ["PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "TOPIC"]:
            if isinstance(row[key], list):
                entities.extend(row[key])
        relationships = extract_relationships(text, nlp_model, entities)
        for rel in relationships:
            G.add_edge(rel[0], rel[1])

    return G

def create_interactive_er_diagram(G):
    """ Create an interactive ER diagram using PyVis. """
    net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    net.show_buttons(filter_=["physics"])  # Add interactive buttons
    net.show("er_diagram.html")  # Save and open the diagram in a browser

# ------------------- FILE SAVING -------------------

def save_results(entities_df, filtered_df, topic):
    """ Saves extracted entities and filtered results to 'Result.xlsx'. """
    output_file = "Result.xlsx"

    try:
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            entities_df.to_excel(writer, sheet_name="Entities", index=False)
            filtered_df.to_excel(writer, sheet_name=f"{topic.capitalize()}_Filtered", index=False)
        print(f"✅ Extracted entities saved in 'Entities' sheet & filtered data in '{topic.capitalize()}_Filtered' in '{output_file}'")
    except Exception as e:
        print(f"❌ Error saving file: {e}")

# ------------------- MAIN FUNCTION -------------------

def main():
    file_path = input("Enter the file path to your Excel file: ").strip().strip('"')
    glove_path = input("Enter the file path to your GloVe embeddings: ").strip().strip('"')
    topic = input("Enter the topic/subject to filter by (e.g., 'crime', 'finance', 'sports'): ").strip().lower()

    nlp_model = initialize_nlp_model()
    df = pd.read_excel(file_path, engine="openpyxl")
    df = process_text(df, nlp_model)

    glove_dict = load_glove_model(glove_path)
    topic_vector = glove_dict.get(topic)

    if topic_vector is None:
        raise ValueError(f"❌ The word '{topic}' is not found in the GloVe model.")

    df_filtered = filter_topic_related_rows(df, topic_vector, glove_dict)

    save_results(df, df_filtered, topic)

    # Build and create an interactive ER diagram
    G = build_relationship_graph(df_filtered, nlp_model)
    create_interactive_er_diagram(G)

if __name__ == "__main__":
    main()
