# preclass_store.py – stockage Supabase
from __future__ import annotations
import streamlit as st
from supabase import create_client, Client
from typing import List, Dict, Any, Tuple
from datetime import datetime

# ───────── Connexion Supabase ─────────────────────────────────────
_SUPABASE_URL: str = st.secrets["supabase"]["url"]
_SUPABASE_KEY: str = st.secrets["supabase"]["key"]
supabase: Client = create_client(_SUPABASE_URL, _SUPABASE_KEY)

TABLE = "email_meta"        # nom de la table Postgres

# ───────── API publique utilisée par app.py ───────────────────────
def get_all_meta(order: str = "updated_at DESC") -> List[Tuple[str, str, str]]:
    """
    Retourne [(email_id, status, updated_at), …] triés selon *order*.
    *order* est une chaîne du type "colonne [ASC|DESC]".
    """
    # Découper l’ordre en colonne + sens
    parts = order.split()
    col = parts[0]
    desc = len(parts) > 1 and parts[1].upper() == "DESC"

    resp = (
        supabase
        .table(TABLE)
        .select("email_id, status, updated_at")
        .order(col, desc=desc)
        .execute()
    )
    rows = resp.data or []
    return [(r["email_id"], r["status"], r["updated_at"]) for r in rows]


def get_meta(uid: str) -> Dict[str, Any]:
    """
    Retourne un dict avec :
      status, preclass_json, user_preclass_json, remarks,
      error_msg, updated_at
    """
    resp = (
        supabase
        .table(TABLE)
        .select("*")
        .eq("email_id", uid)
        .single()
        .execute()
    )
    row = resp.data
    if not row:
        return {}

    return {
        "status":              row.get("status"),
        "preclass_json":       row.get("preclass_json") or {},
        "user_preclass_json":  row.get("user_preclass_json") or {},
        "remarks":             row.get("remarks"),
        "error_msg":           row.get("error_msg"),
        "updated_at":          row.get("updated_at"),
    }


def update_meta(uid: str, meta: Dict[str, Any]) -> None:
    """
    Met à jour ou insère :
      user_preclass_json, remarks, status, error_msg, updated_at
    (ne touche pas à preclass_json).
    """
    payload = {
        "user_preclass_json": meta.get("user_preclass_json", {}),
        "remarks":            meta.get("remarks", ""),
        "status":             meta.get("status", "done"),
        "error_msg":          meta.get("error_msg"),
        "updated_at":         datetime.utcnow().isoformat()
    }

    # UPDATE d’abord ; si aucune ligne modifiée → INSERT
    updated = (
        supabase
        .table(TABLE)
        .update(payload)
        .eq("email_id", uid)
        .execute()
    ).data

    if not updated:
        supabase.table(TABLE).insert({**payload, "email_id": uid}).execute()


def save_user_classification(uid: str, classification: Dict[str, Any]) -> None:
    """
    Écrit la classification validée dans preclass_json et passe status à 'done'.
    """
    payload = {
        "preclass_json": classification,
        "status":        "done",
        "updated_at":    datetime.utcnow().isoformat()
    }
    supabase.table(TABLE).update(payload).eq("email_id", uid).execute()
