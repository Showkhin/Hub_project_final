# extract_ola.py
import os
import json
import pandas as pd
import requests
from rapidfuzz import fuzz
from oci_helpers import load_cloud_csv, upload_cloud_csv
from utils import normalize_date, normalize_time

# --- Files ---
INPUT_OBJECT_NAME = "processed_incidents_with_emotion.csv"
OUTPUT_OBJECT_NAME = "final_emotion_ensemble.csv"
MAIN_OBJECT_NAME = "main.csv"

# --- Ollama API ---
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL_STRUCT", "gemma3")
GEN_URL = os.getenv("OLLAMA_URL_GENERATE", "http://127.0.0.1:11434/api/generate")
CHAT_URL = os.getenv("OLLAMA_URL_CHAT", "http://127.0.0.1:11434/v1/chat/completions")

# --- Final Fields ---
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
    "emotion",
    "needs_review",
    "final_confidence",
]


# --- Helpers ---
def clean_markdown_json(text: str) -> str:
    """Strip code fences around JSON."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
        return t.replace("```json", "").replace("```", "")
    return t


def _ollama_extract_json(prompt: str) -> dict:
    """Try Ollama generate first, then chat API."""
    try:
        r = requests.post(
            GEN_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=120
        )
        if r.status_code == 200:
            raw = (r.json().get("response") or "").strip()
            if raw:
                return json.loads(clean_markdown_json(raw))
    except Exception as e:
        print(f"[Ollama generate failed] {e}")

    try:
        r = requests.post(
            CHAT_URL,
            json={"model": OLLAMA_MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False},
            timeout=120,
        )
        if r.status_code == 200:
            j = r.json()
            raw = j["choices"][0]["message"]["content"].strip()
            if raw:
                return json.loads(clean_markdown_json(raw))
    except Exception as e:
        print(f"[Ollama chat failed] {e}")

    return {}


def extract_fields_from_transcription(transcription: str) -> dict:
    """Call Ollama to extract fields from transcription text."""
    if not str(transcription).strip():
        return {field: "" for field in FIELDS if field not in ("filename", "emotion", "needs_review", "final_confidence")}

    prompt = (
        "Extract incident details as JSON with keys: "
        "client_name, incident_date, incident_time, location, incident_type, actions_taken, "
        "severity, description, reporter, reported_date, organization name, recurrence.\n"
        "All fields must exist. If unknown, set as empty string.\n"
        "Dates must be ISO YYYY-MM-DD, times HH:MM:SS.\n"
        "Return JSON only.\n\n"
        f"Text:\n{transcription}"
    )

    data = _ollama_extract_json(prompt)
    if not isinstance(data, dict):
        return {k: "" for k in [
            "client_name", "incident_date", "incident_time", "location", "incident_type",
            "actions_taken", "severity", "description", "reporter", "reported_date",
            "organization name", "recurrence"
        ]}

    # Normalize dates and times
    d, _ = normalize_date(data.get("incident_date", ""))
    rd, _ = normalize_date(data.get("reported_date", ""))
    t, _ = normalize_time(data.get("incident_time", ""))
    data["incident_date"] = d
    data["reported_date"] = rd
    data["incident_time"] = t

    return data


def _pairwise_best_match(client_guess: str, org_guess: str, df_main: pd.DataFrame) -> tuple[str, str]:
    """Pick best client/org pair from main.csv."""
    if df_main.empty or "client_name" not in df_main.columns or "organization name" not in df_main.columns:
        return client_guess, org_guess

    best_score = -1.0
    best_client, best_org = client_guess, org_guess
    cg, og = (client_guess or "").lower(), (org_guess or "").lower()

    for _, r in df_main.iterrows():
        c, o = str(r.get("client_name", "")).lower(), str(r.get("organization name", "")).lower()
        s_client = fuzz.ratio(cg, c) / 100.0 if cg else 0
        s_org = fuzz.ratio(og, o) / 100.0 if og else 0
        score = 0.6 * s_client + 0.4 * s_org
        if score > best_score:
            best_score = score
            best_client = r.get("client_name", "")
            best_org = r.get("organization name", "")

    return best_client, best_org


# --- Main Processing ---
def process_and_upload() -> list:
    # Load processed incidents
    df_processed = load_cloud_csv(INPUT_OBJECT_NAME)
    if df_processed.empty:
        print("No processed data to add.")
        return []

    # Load or create final CSV
    try:
        df_final = load_cloud_csv(OUTPUT_OBJECT_NAME)
        if df_final.empty:
            raise ValueError("Empty file")
        print(f"Loaded existing {OUTPUT_OBJECT_NAME}, {len(df_final)} rows")
    except Exception:
        df_final = pd.DataFrame(columns=FIELDS)
        upload_cloud_csv(OUTPUT_OBJECT_NAME, df_final)  # ✅ upload immediately
        print(f"{OUTPUT_OBJECT_NAME} not found, created empty file")

    for c in FIELDS:
        if c not in df_final.columns:
            df_final[c] = ""

    existing = set(df_final["filename"].astype(str).values)

    # Load main.csv for client/org matching
    try:
        df_main = load_cloud_csv(MAIN_OBJECT_NAME)
    except Exception:
        df_main = pd.DataFrame(columns=["client_name", "organization name"])

    new_records = []
    for _, prow in df_processed.iterrows():
        fname = str(prow.get("filename", "")).strip()
        if not fname or fname in existing:
            continue

        # Extract fields from transcription
        extracted = extract_fields_from_transcription(str(prow.get("transcription", "") or ""))
        best_client, best_org = _pairwise_best_match(
            extracted.get("client_name", ""), extracted.get("organization name", ""), df_main
        )
        extracted["client_name"], extracted["organization name"] = best_client, best_org

        # Build final record
        rec = {k: "" for k in FIELDS}
        rec.update(extracted)
        rec["filename"] = fname
        rec["emotion"] = str(prow.get("emotion", "") or "")
        rec["needs_review"] = str(prow.get("needs_review", "") or "")
        rec["final_confidence"] = str(prow.get("final_confidence", "") or "")

        df_final = pd.concat([df_final, pd.DataFrame([rec])], ignore_index=True)
        new_records.append(rec)

    if not new_records:
        print("No missing filenames found to process.")
        return []

    df_final = df_final.drop_duplicates(subset="filename", keep="last")
    upload_cloud_csv(OUTPUT_OBJECT_NAME, df_final)
    print(f"✅ Processed {len(new_records)} missing records and updated {OUTPUT_OBJECT_NAME}")
    return new_records


if __name__ == "__main__":
    process_and_upload()
