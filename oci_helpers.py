import os
import io
import time
import pandas as pd
import datetime as dt
from typing import List
import streamlit as st
import oci
from oci.object_storage.models import CreatePreauthenticatedRequestDetails

@st.cache_resource(show_spinner=False)
def get_oci_client():
    config = _build_oci_config()
    client = oci.object_storage.ObjectStorageClient(config)
    return client, config

def _build_oci_config() -> dict:
    if "OCI_USER_OCID" in st.secrets:
        return {
            "user": st.secrets["OCI_USER_OCID"],
            "tenancy": st.secrets["OCI_TENANCY_OCID"],
            "region": st.secrets["OCI_REGION"],
            "fingerprint": st.secrets["OCI_FINGERPRINT"],
            "key_content": st.secrets["OCI_KEY_CONTENT"],
        }
    cfg_path = os.getenv("OCI_CONFIG_PATH", os.path.expanduser("~/.oci/config"))
    profile = os.getenv("OCI_PROFILE", "DEFAULT")
    return oci.config.from_file(cfg_path, profile_name=profile)


NAMESPACE = os.getenv("OCI_NAMESPACE", "sdzbwxl65lpx")
BUCKET = os.getenv("OCI_BUCKET", "incident-data-bucket")

@st.cache_data(show_spinner=False)
def list_wavs(prefix: str = "sample") -> List[str]:
    client, _ = get_oci_client()
    names = []
    start = None
    while True:
        resp = client.list_objects(
            NAMESPACE, BUCKET, prefix=prefix, start=start, fields="name"
        )
        for obj in resp.data.objects:
            if obj.name.lower().endswith(".wav"):
                names.append(obj.name)
        if not resp.data.next_start_with:
            break
        start = resp.data.next_start_with
    names.sort()
    return names

def upload_blob(name: str, data: bytes) -> None:
    client, _ = get_oci_client()
    client.put_object(NAMESPACE, BUCKET, name, io.BytesIO(data))

def download_blob(name: str) -> bytes:
    client, _ = get_oci_client()
    resp = client.get_object(NAMESPACE, BUCKET, name)
    return resp.data.content

def delete_objects(names: List[str]) -> None:
    client, _ = get_oci_client()
    for n in names:
        try:
            client.delete_object(NAMESPACE, BUCKET, n)
        except Exception as e:
            st.toast(f"Couldn't delete {n}: {e}")

def load_cloud_csv(object_name: str, columns: List[str] | None = None):
    client, _ = get_oci_client()
    try:
        resp = client.get_object(NAMESPACE, BUCKET, object_name)
        return pd.read_csv(io.BytesIO(resp.data.content))
    except Exception:
        return pd.DataFrame(columns=columns) if columns else pd.DataFrame()

def upload_cloud_csv(object_name: str, df):
    client, _ = get_oci_client()
    bio = io.BytesIO()
    df.to_csv(bio, index=False)
    bio.seek(0)
    client.put_object(NAMESPACE, BUCKET, object_name, bio)

def create_share_link(object_name: str, days: int = 7) -> str | None:
    try:
        client, cfg = get_oci_client()
        details = CreatePreauthenticatedRequestDetails(
            name=f"par-{object_name}-{int(time.time())}",
            access_type="ObjectRead",
            time_expires=(dt.datetime.utcnow() + dt.timedelta(days=days)),
            object_name=object_name,
        )
        par = client.create_preauthenticated_request(NAMESPACE, BUCKET, details).data
        return f"https://objectstorage.{cfg['region']}.oraclecloud.com{par.access_uri}"
    except Exception as e:
        st.warning(f"Couldn't create shareable link for {object_name}: {e}")
        return None
