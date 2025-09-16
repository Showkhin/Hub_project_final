import pandas as pd
from rapidfuzz import fuzz
import requests
import os

# === CONFIG ===
FINAL_FILE = "final.csv"
PROCESSED_FILE = "stasrt.csv"
OLLAMA_MODEL = "gemma3"

# Ollama host and endpoint
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11435/api/generate")

# --- Load CSVs ---
df_final = pd.read_csv(FINAL_FILE)
df_processed = pd.read_csv(PROCESSED_FILE)

# Use only common columns
common_cols = [c for c in df_final.columns if c in df_processed.columns]
print(f"🔎 Comparing {len(common_cols)} common columns: {common_cols}")

# --- Which columns to use Ollama for ---
SEMANTIC_COLS = {"description", "actions_taken"}

def clean_ollama_json(text: str) -> str:
    """Strip Markdown/json wrappers from Ollama responses."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
        return text.replace("```json", "").replace("```", "")
    return text

def ollama_similarity(val1: str, val2: str) -> float:
    """Ask Ollama if two values mean the same thing (semantic)."""
    if not str(val1).strip() and not str(val2).strip():
        return 1.0

    prompt = f"""
    Compare the following two text values:

    A: "{val1}"
    B: "{val2}"

    Do they mean the same thing, even if phrased differently or with minor spelling/formatting issues?
    Respond with ONLY a number between 0.0 and 1.0 (higher = more similar).
    """

    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(OLLAMA_URL, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        j = resp.json()
        raw_response = j.get("response", "").strip()
        cleaned = clean_ollama_json(raw_response)

        score = float(cleaned.split()[0])
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"[⚠️ Ollama error] {e}")
        return 0.0

def fuzzy_similarity(val1: str, val2: str) -> float:
    """Fuzzy ratio for structured values like dates, times, names, etc."""
    g, p = str(val1).strip(), str(val2).strip()
    if not g and not p:
        return 1.0
    elif g and p:
        return fuzz.ratio(g, p) / 100
    else:
        return 0.0

def compare_cell(col, val1, val2):
    """Decide whether to use Ollama or fuzzy based on column."""
    if col in SEMANTIC_COLS:
        return ollama_similarity(val1, val2)
    else:
        return fuzzy_similarity(val1, val2)

# --- Compare row by row ---
all_scores = []
for i in range(min(len(df_final), len(df_processed))):
    row_scores = {}
    for col in common_cols:
        score = compare_cell(col, df_final.iloc[i][col], df_processed.iloc[i][col])
        row_scores[col] = score
    row_scores["row_index"] = i
    row_scores["row_accuracy"] = sum(row_scores[c] for c in common_cols) / len(common_cols)
    all_scores.append(row_scores)

df_scores = pd.DataFrame(all_scores)

# --- Column-level accuracy ---
col_accuracy = df_scores[common_cols].mean()

# --- Overall accuracy ---
overall_accuracy = df_scores[common_cols].values.mean()

# --- Results ---
print("\n📊 Per-column average accuracy:")
print(col_accuracy)

print("\n📑 First 10 per-row accuracies:")
print(df_scores[["row_index", "row_accuracy"]].head(10))

print(f"\n✅ Overall model accuracy: {overall_accuracy:.3f}")
