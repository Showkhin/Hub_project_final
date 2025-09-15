---
title: Emotion Lab
emoji: 💻
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
python_version: "3.10"
pinned: false
---

# Trashery Emotion Lab

## Overview

Streamlit app for emotion analysis on audio incidents using Whisper, SpeechBrain, Transformers, and Oracle Cloud Object Storage.

## Setup

### 1. Clone repository

git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
### 2. Create and activate Python virtual environment

python -m venv venv

Windows
.\venv\Scripts\activate

macOS/Linux
source venv/bin/activate
### 3. Install dependencies
pip install -r requirements.txt

### 4. Setup environment variables or secrets
OCI_USER_OCID = "YOUR_OCI_USER_OCID"
OCI_TENANCY_OCID = "YOUR_OCI_TENANCY_OCID"
OCI_REGION = "YOUR_OCI_REGION"
OCI_FINGERPRINT = "YOUR_OCI_FINGERPRINT"
OCI_KEY_CONTENT = """-----BEGIN PRIVATE KEY-----
YOUR_PRIVATE_KEY_HERE
-----END PRIVATE KEY-----"""
Create `.streamlit/secrets.toml` (make sure it is in `.gitignore`) and add:


You can also set these as OS environment variables.

### 5. Run the app
streamlit run app.py



