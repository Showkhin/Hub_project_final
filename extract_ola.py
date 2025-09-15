# extract_ola.py
import os
import pandas as pd
import json
import requests
from difflib import get_close_matches
from pipeline import load_cloud_csv, upload_cloud_csv

# --- OCI Config ---
CONFIG_PATH = r"C:\Users\tousi\.oci\config"
NAMESPACE = 'sdzbwxl65lpx'
BUCKET_NAME = 'incident-data-bucket'

# --- Files ---
INPUT_OBJECT_NAME = 'processed_incidents_with_emotion.csv'
OUTPUT_OBJECT_NAME = 'final_emotion_ensemble.csv'
MAIN_OBJECT_NAME = 'main.csv'

# --- Ollama API ---
OLLAMA_MODEL = 'gemma3'
OLLAMA_URL = 'http://127.0.0.1:11434/api/generate'

# --- Fields in final ensemble ---
FIELDS = [
    "filename",
    "client_name",
    "incident_date",
    "incident_time",
    "location",
    "incident_type",
    "actions_taken",
    "severity",
    "description",
    "reporter",
    "reported_date",
    "organization name",
    "recurrence",
    "emotion"
]

# --- Helper Functions ---
def clean_markdown_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
        return text.replace("```json", "").replace("```", "")
    return text

def extract_fields_from_transcription(transcription: str) -> dict:
    """Send transcription to Ollama API and extract structured fields."""
    if not transcription.strip():
        return {field: "" for field in FIELDS[1:]}  # empty if no transcription
    prompt = (
        "Extract all incident details from the following text as a JSON object.\n"
        f"The object must have these fields exactly:\n{', '.join(FIELDS[1:])}.\n"
        "If a field is missing, fill with empty string or 'N/A'.\n"
        "Return ONLY the JSON object, no explanations or markdown.\n\n"
        f"Text:\n{transcription}"
    )
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
        resp.raise_for_status()
        raw_response = resp.json().get("response", "")
        cleaned_text = clean_markdown_json(raw_response)
        data = json.loads(cleaned_text)
        if isinstance(data, dict):
            for field in FIELDS[1:]:
                if field not in data:
                    data[field] = ""
            return data
    except Exception as e:
        print(f"Ollama extraction failed: {e}")
    return {field: "" for field in FIELDS[1:]}

def best_match(value: str, candidates: list, cutoff: float = 0.6) -> str:
    """Return the closest match if found, else return empty string."""
    if not value or not candidates:
        return ""
    matches = get_close_matches(value, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else ""

def fuzzy_update_client_org(row: pd.Series, df_main: pd.DataFrame) -> pd.Series:
    """
    Update client_name and organization name with best match from df_main.
    If no match found, keep the original value from Ollama API.
    """
    if df_main.empty:
        return row

    main_clients = df_main['client_name'].dropna().unique().tolist()
    main_orgs = df_main['organization name'].dropna().unique().tolist()

    best_client = best_match(row.get("client_name", ""), main_clients)
    best_org = best_match(row.get("organization name", ""), main_orgs)

    # Only overwrite if a good match is found
    if best_client:
        row["client_name"] = best_client
    if best_org:
        row["organization name"] = best_org

    return row

# --- Main Processing Function ---
def process_and_upload() -> list:
    # Load processed incidents
    try:
        df_processed = load_cloud_csv(INPUT_OBJECT_NAME)
    except Exception:
        print(f"{INPUT_OBJECT_NAME} not found.")
        return []

    if df_processed.empty:
        print("No processed data to add.")
        return []

    # Load existing final CSV
    try:
        df_final = load_cloud_csv(OUTPUT_OBJECT_NAME)
        print(f"Loaded existing {OUTPUT_OBJECT_NAME}, {len(df_final)} rows")
    except Exception:
        df_final = pd.DataFrame(columns=FIELDS)
        print(f"{OUTPUT_OBJECT_NAME} not found, creating new DataFrame")

    for field in FIELDS:
        if field not in df_final.columns:
            df_final[field] = ""

    existing_filenames = set(df_final['filename'].values) if 'filename' in df_final.columns else set()

    # Load main.csv for fuzzy matching
    try:
        df_main = load_cloud_csv(MAIN_OBJECT_NAME)
    except Exception as e:
        print(f"Could not load main.csv: {e}")
        df_main = pd.DataFrame(columns=["client_name", "organization name"])

    new_records_list = []

    # Process only missing filenames
    for _, row in df_processed.iterrows():
        fname = row['filename']
        if fname in existing_filenames:
            continue

        transcription_text = row.get('transcription', '') or ''
        extracted_fields = extract_fields_from_transcription(transcription_text)

        # Ensure all fields exist
        record = {field: "" for field in FIELDS}
        record.update(extracted_fields)
        record['filename'] = fname
        record['emotion'] = row.get('emotion', '')

        # Convert to Series for easy fuzzy update
        record_series = pd.Series(record)
        record_series = fuzzy_update_client_org(record_series, df_main)

        # Append to final DataFrame
        df_final = pd.concat([df_final, pd.DataFrame([record_series])], ignore_index=True)
        new_records_list.append(record_series.to_dict())

    if not new_records_list:
        print("No missing filenames found to process.")
        return []

    # Remove duplicates by filename, keep last added (with fuzzy updates)
    df_final = df_final.drop_duplicates(subset='filename', keep='last')

    # Upload final CSV
    upload_cloud_csv(OUTPUT_OBJECT_NAME, df_final)
    print(f"Processed {len(new_records_list)} missing records and updated {OUTPUT_OBJECT_NAME}")

    return new_records_list

# --- Entry Point ---
if __name__ == "__main__":
    process_and_upload()
