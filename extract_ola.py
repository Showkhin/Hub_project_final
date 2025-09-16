# extract_ola.py
import os
import json
import pandas as pd
import requests
from difflib import get_close_matches

from oci_helpers import load_cloud_csv, upload_cloud_csv
from utils import normalize_date, normalize_time

# --- Files in bucket ---
INPUT_OBJECT_NAME = 'processed_incidents_with_emotion.csv'
OUTPUT_OBJECT_NAME = 'final_emotion_ensemble.csv'
MAIN_OBJECT_NAME = 'main.csv'

# --- Ollama API ---
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL_STRUCT", "gemma3")
OLLAMA_URL = os.getenv("OLLAMA_URL_GENERATE", "http://127.0.0.1:11434/api/generate")

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

# --- Helpers ---
def clean_markdown_json(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
        return t.replace("```json", "").replace("```", "")
    return t

def extract_fields_from_transcription(transcription: str) -> dict:
    if not transcription.strip():
        return {field: "" for field in FIELDS[1:]}

    prompt = (
        "Extract all incident details from the following text as a JSON object.\n"
        f"The object must have these fields exactly:\n{', '.join(FIELDS[1:])}.\n"
        "If a field is missing, fill with empty string.\n"
        "Return ONLY the JSON object, no explanations or markdown.\n\n"
        f"Text:\n{transcription}"
    )

    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(OLLAMA_URL, json=payload, headers=headers, timeout=600)
        resp.raise_for_status()
        j = resp.json()

        raw_response = j.get("response", "")
        print("🔎 Ollama raw response:", raw_response[:500])  # <-- DEBUG PRINT

        if not raw_response.strip():
            raise ValueError(f"No response text from Ollama: {j}")

        cleaned_text = clean_markdown_json(raw_response)
        data = json.loads(cleaned_text)

        # ensure all fields exist
        for field in FIELDS[1:]:
            if field not in data:
                data[field] = ""
        return data

    except Exception as e:
        print(f"[❌ Ollama extraction failed] {e}")

    return {field: "" for field in FIELDS[1:]}


def best_match(value: str, candidates: list, cutoff: float = 0.6) -> str:
    if not value or not candidates:
        return value
    matches = get_close_matches(value, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else value

def fuzzy_update_client_org(row: pd.Series, df_main: pd.DataFrame) -> pd.Series:
    if df_main.empty:
        return row
    if "client_name" in df_main.columns:
        row["client_name"] = best_match(row.get("client_name", ""), df_main["client_name"].dropna().unique().tolist())
    if "organization name" in df_main.columns:
        row["organization name"] = best_match(row.get("organization name", ""), df_main["organization name"].dropna().unique().tolist())
    return row

# --- Main Processing ---
def process_and_upload() -> list:
    try:
        df_processed = load_cloud_csv(INPUT_OBJECT_NAME)
    except Exception:
        print(f"{INPUT_OBJECT_NAME} not found.")
        return []

    if df_processed.empty:
        print("No processed data to add.")
        return []

    try:
        df_final = load_cloud_csv(OUTPUT_OBJECT_NAME)
        print(f"Loaded existing {OUTPUT_OBJECT_NAME}, {len(df_final)} rows")
    except Exception:
        df_final = pd.DataFrame(columns=FIELDS)
        print(f"{OUTPUT_OBJECT_NAME} not found, creating new DataFrame")

    # ensure all columns exist
    for field in FIELDS:
        if field not in df_final.columns:
            df_final[field] = ""

    existing_filenames = set(df_final['filename'].astype(str).values)

    try:
        df_main = load_cloud_csv(MAIN_OBJECT_NAME)
    except Exception as e:
        print(f"Could not load main.csv: {e}")
        df_main = pd.DataFrame(columns=["client_name", "organization name"])

    new_records_list = []

    for _, row in df_processed.iterrows():
        fname = str(row.get('filename', '')).strip()
        if not fname or fname in existing_filenames:
            continue

        transcription_text = row.get('transcription', '') or ''
        extracted_fields = extract_fields_from_transcription(transcription_text)

        record = {field: "" for field in FIELDS}
        record.update(extracted_fields)
        record['filename'] = fname
        record['emotion'] = row.get('emotion', '')

        record_series = pd.Series(record)
        record_series = fuzzy_update_client_org(record_series, df_main)

        df_final = pd.concat([df_final, pd.DataFrame([record_series])], ignore_index=True)
        new_records_list.append(record_series.to_dict())

    if not new_records_list:
        print("No missing filenames found to process.")
        return []

    df_final = df_final.drop_duplicates(subset='filename', keep='last')
    upload_cloud_csv(OUTPUT_OBJECT_NAME, df_final)
    print(f"✅ Processed {len(new_records_list)} missing records and updated {OUTPUT_OBJECT_NAME}")

    return new_records_list

if __name__ == "__main__":
    process_and_upload()
