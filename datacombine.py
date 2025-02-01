import spacy
import pandas as pd
from spacy.matcher import Matcher
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="thinc")

def prompt_file_path():
    """
    Prompt the user to enter the file path of the dataset.
    """
    file_path = "/Users/soongshaozhi/Documents/Hackathons/SMUBIA Datathon/wikileaks_parsed.xlsx"
    return file_path


def prompt_phrases():
    """
    Prompt the user to enter phrases for the matcher.
    """
    print("Enter phrases for the matcher, separated by commas.")
    phrases = input("Phrases: ").split(",")
    return [phrase.strip() for phrase in phrases]


def initialize_nlp_model():
    """
    Load the spaCy language model.
    """
    return spacy.load("en_core_web_trf")


def initialize_matcher(nlp_model, phrases):
    """
    Initialize the spaCy Matcher with user-defined phrases.
    """
    matcher = Matcher(nlp_model.vocab)
    
    # Convert each phrase into a pattern (list of dictionaries)
    patterns = [[{"LOWER": token.lower()} for token in phrase.split()] for phrase in phrases]
    
    # Add patterns to the matcher
    matcher.add("DOMAIN_SPECIFIC", patterns)
    return matcher


def remove_duplicates(entity_list):
    """
    Remove duplicates from a list of entities, treating entities with different cases as duplicates.
    """
    # Convert entities to lowercase for normalization, then restore original capitalization
    unique_entities = list(dict.fromkeys(entity.lower() for entity in entity_list))
    return unique_entities


def extract_key_actions(text, nlp_model):
    """
    Extract significant actions and related noun phrases from text.
    """
    doc = nlp_model(text)
    key_actions = []

    for token in doc:
        if token.pos_ == "VERB":  # Focus on verbs
            action = token.text
            associated_terms = []

            # Extract dependencies like direct objects, prepositions, etc.
            for child in token.children:
                if child.dep_ in ["dobj", "prep", "acomp", "xcomp", "advcl", "ccomp"] and child.pos_ in ["NOUN", "PROPN", "ADJ"]:
                    associated_terms.append(child.text)

            # Construct action phrase
            if associated_terms:
                action_phrase = f"{action} {' '.join(associated_terms)}"
                key_actions.append(action_phrase)
            else:
                key_actions.append(action)

    return key_actions


def extract_pattern_based_entities(text, nlp_model, matcher):
    """
    Extract custom entities using spaCy Matcher for predefined patterns.
    """
    doc = nlp_model(text)
    matches = matcher(doc)
    pattern_entities = []

    for match_id, start, end in matches:
        pattern_entities.append(doc[start:end].text)

    return pattern_entities


def extract_entities_and_key_actions(text, nlp_model, matcher):
    """
    Extract specific entities, key actions, and custom matched entities from the text.
    """
    doc = nlp_model(text)
    entities = {
        "PERSON": [],
        "ORG": [],
        "GPE": [],
        "EVENT": [],
        "PRODUCT": [],
        "TOPIC": [],
        "DATE": [],
        "KEY_ACTIONS": [],  # Extracted key actions
    }
    
    # Extract entities
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

    # Extract pattern-based entities
    entities["TOPIC"].extend(extract_pattern_based_entities(text, nlp_model, matcher))

    # Extract key actions
    entities["KEY_ACTIONS"] = extract_key_actions(text, nlp_model)

    # Remove duplicates in all entity lists
    for key in entities:
        entities[key] = remove_duplicates(entities[key])

    return entities


def process_chunk(chunk, nlp_model, matcher):
    """
    Process a chunk of the dataset and extract entities and key actions.
    """
    def process_row(text):
        extracted = extract_entities_and_key_actions(text, nlp_model, matcher)
        return pd.Series({
            "PERSON": ", ".join(extracted["PERSON"]),
            "ORG": ", ".join(extracted["ORG"]),
            "GPE": ", ".join(extracted["GPE"]),
            "EVENT": ", ".join(extracted["EVENT"]),
            "PRODUCT": ", ".join(extracted["PRODUCT"]),
            "TOPIC": ", ".join(extracted["TOPIC"]),
            "DATE": ", ".join(extracted["DATE"]),
            "KEY_ACTIONS": ", ".join(extracted["KEY_ACTIONS"]),
            # Removed "Row" column
        })

    return chunk["Text"].apply(process_row)


def get_match_count(row, group_row):
    """
    Calculate the number of overlapping entities between two rows within the same column.
    """
    # Split entities into sets for comparison
    person_set = set(row["PERSON"].split(","))
    org_set = set(row["ORG"].split(","))
    gpe_set = set(row["GPE"].split(","))
    topic_set = set(row["TOPIC"].split(","))
    
    # Split entities for the group row
    person_set_group = set(group_row["PERSON"].split(","))
    org_set_group = set(group_row["ORG"].split(","))
    gpe_set_group = set(group_row["GPE"].split(","))
    topic_set_group = set(group_row["TOPIC"].split(","))
    
    # Count matching entities between columns in the two rows
    person_match = len(person_set.intersection(person_set_group))
    org_match = len(org_set.intersection(org_set_group))
    gpe_match = len(gpe_set.intersection(gpe_set_group))
    topic_match = len(topic_set.intersection(topic_set_group))

    # Sum the matches
    total_matches = person_match + org_match + gpe_match + topic_match
    
    return total_matches


def combine_texts_with_row_labels(group):
    """
    Combine texts within the same topic, appending the row indices correctly with an offset,
    and create a new column showing only the row numbers beside the combined text.
    """
    # Adjust row numbers by adding an offset of 2 to match your Excel row numbering
    combined_text = "\n\n".join([f"Text: \"{row['Text']}\" (Row {row.name + 2})" for _, row in group.iterrows()])
    
    # Create a column with just the row indices (joined as a string)
    row_indices = ", ".join([str(row.name + 2) for _, row in group.iterrows()])
    
    # Count how many rows share enough matching entities (>=2 matches)
    match_count = 0
    for idx1, row1 in group.iterrows():
        for idx2, row2 in group.iterrows():
            if idx1 != idx2:  # Skip comparing the row with itself
                matches = get_match_count(row1, row2)
                if matches >= 2:  # Set threshold for match count
                    match_count += 1

    # Only include groups with at least 2 matches between rows
    if match_count >= 2:
        return {
            "Topic": group.name,  # Group name (e.g., "TOPIC")
            "Text": combined_text,  # Combined text with adjusted row labels
            "Row Values": row_indices  # Row indices as a string (added as a column)
        }
    else:
        return None  # This will ensure that groups without enough matches are dropped





def main():
    # Prompt for file path and phrases
    file_path = prompt_file_path()
    phrases = prompt_phrases()

    # Load the dataset
    data = pd.read_excel(file_path)

    # Split the dataset into chunks of 100 rows
    chunks = [data[i:i+100] for i in range(0, len(data), 100)]

    # Load models
    nlp_model = initialize_nlp_model()
    matcher = initialize_matcher(nlp_model, phrases)

    # Process the first chunk
    chunk_to_process = chunks[0]  # Process only the first 100 rows
    processed_data = process_chunk(chunk_to_process, nlp_model, matcher)

    # Combine extracted entities and actions with the original data
    output_data = pd.concat([chunk_to_process, processed_data], axis=1)

    # Group by "TOPIC" and combine texts with row labels
    grouped_data = (
        output_data
        .groupby("TOPIC")
        .apply(combine_texts_with_row_labels)
        .dropna()
        .apply(pd.Series)  # Convert dictionaries into a structured DataFrame
        .reset_index(drop=True)
    )
    
    # Save the extracted entities and grouped data to a new Excel file
    output_file = "/Users/soongshaozhi/Documents/Hackathons/SMUBIA Datathon/Output/output_combined.xlsx"
    
    with pd.ExcelWriter(output_file) as writer:
        # Save the extraction data to the first sheet
        output_data.to_excel(writer, sheet_name="Extracted Entities", index=False)
        # Save the grouped data to the second sheet
        grouped_data.to_excel(writer, sheet_name="Grouped Data", index=False)
    
    print(f"Processed data with extracted entities and grouped data has been saved to {output_file}")


if __name__ == "__main__":
    main()
