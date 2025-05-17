#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
streamlit_app.py – Webmail DeepSeek  (version stable)
• Parcours des e-mails marqués « done »
• Métadonnées éditables + champ « Remarques » persistant
• Aperçu pièces jointes, navigation rapide, purge cache
"""

# ───────── Imports ────────────────────────────────────────────────
import base64, email, io
from typing import Any, List, Tuple

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import json, html
from typing import Any, List, Dict
import extractor                    # helpers pièces jointes
import imap_client                  # fetch_email(uid)
import preclass_store as storage    # get_meta / update_meta / get_all_meta

# ───────── Style global ───────────────────────────────────────────
st.set_page_config(page_title="Webmail DeepSeek", layout="wide")
st.markdown("""
<style>
#MainMenu, footer{visibility:hidden}
section[data-testid="stSidebar"]>div{background:#f0f2f6;padding:1rem;border-radius:12px}
pre{font-family:'Courier New',monospace;font-size:15px;background:#f8f8f8;
    padding:1rem;border-radius:8px;overflow-x:auto}
h1,h2,h3,.st-subheader{color:#2c3e50}
</style>
""", unsafe_allow_html=True)

# ═════════ UTILITAIRES ════════════════════════════════════════════
def _is_empty(v: Any) -> bool:
    if v is None: return True
    if isinstance(v, str): return v.strip() == "" or v.strip().lower() == "none"
    if isinstance(v, (list, tuple, set)): return len(v) == 0 or all(_is_empty(x) for x in v)
    if isinstance(v, dict): return len(v) == 0 or all(_is_empty(x) for x in v.values())
    return False

def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out

def unflatten(flat: dict) -> dict:
    root: dict[str, Any] = {}
    for path, val in flat.items():
        cur = root
        *heads, leaf = path.split(".")
        for h in heads:
            cur = cur.setdefault(h, {})
        cur[leaf] = val
    return root
# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
import ast, json
from typing import Any, List, Dict

def _as_records(v: Any) -> List[Dict[str, Any]] | None:
    """
    Convertit *v* en liste de dicts même si la source est :
      • str JSON valide
      • str repr() Python (guillemets simples)
    """

    # 1) Si chaîne : essaye JSON puis ast.literal_eval
    if isinstance(v, str):
        txt = v.strip()
        if txt.startswith(("{", "[")):
            # d’abord JSON « propre »
            try:
                v = json.loads(txt)
            except Exception:
                # puis représentation Python
                try:
                    v = ast.literal_eval(txt)
                except Exception:
                    return None

    # 2) Liste de listes -> aplatissement
    if isinstance(v, list):
        if len(v) == 1 and isinstance(v[0], list):
            v = v[0]
        if all(isinstance(x, dict) for x in v):
            return v

    # 3) Dict seul
    if isinstance(v, dict):
        return [v]

    return None


# ------------------------------------------------------------------
# Affichage générique
# ------------------------------------------------------------------
def _display_value(v: Any) -> str:
    """
    Retourne une chaîne ou du HTML prêt à être injecté dans le tableau
    Streamlit.  Les listes/dicts de pièces jointes sont rendus en tableau ;
    les autres types conservent l’ancien comportement.
    """
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return ""

    # ----- Pièces jointes → tableau HTML -------
    records = _as_records(v)
    if records:
        head = (
            "<tr><th>📎&nbsp;Fichier</th>"
            "<th>Type</th>"
        )
        body = []
        for rec in records:
            statut = (
                "✅ Traité"        if rec.get("processed") else
                "⏳ En&nbsp;attente" if "processed" in rec else
                ""
            )
            body.append(
                "<tr>"
                f"<td>{html.escape(str(rec.get('filename','')))}</td>"
                f"<td>{html.escape(str(rec.get('type','')))}</td>"
                "</tr>"
            )
        return (
            "<table style='border-collapse:collapse;font-size:12px;'>"
            "<thead>"+head+"</thead><tbody>"+ "".join(body) +"</tbody></table>"
        )

    # ----- Listes simples -------
    if isinstance(v, list):
        return ", ".join(str(x) for x in v if x not in (None, ""))

    # ----- Booléens -------
    if isinstance(v, bool):
        return "Oui" if v else "Non"

    # ----- Autres types -------
    return str(v)


# ───────── Contenu HTML du mail ───────────────────────────────────
def _html_from_msg(msg: email.message.EmailMessage) -> str:
    part = msg.get_body(preferencelist=("html",))
    if part:
        return part.get_content()
    part = msg.get_body(preferencelist=("plain",))
    txt  = part.get_content() if part else ""
    return f"<pre style='white-space:pre-wrap'>{txt}</pre>"

def _html_with_inline_images(msg: email.message.EmailMessage) -> str:
    html=_html_from_msg(msg)
    cid_map={}
    for part in msg.walk():
        if part.get_content_maintype()=="image":
            cid=(part.get("Content-ID") or "").strip("<>")
            if cid:
                data=part.get_payload(decode=True) or b""
                mime=part.get_content_type()
                b64 =base64.b64encode(data).decode()
                cid_map[f"cid:{cid}"]=f"data:{mime};base64,{b64}"
    for ph,uri in cid_map.items():
        html=html.replace(ph,uri)
    return html

# ───────── Pièces jointes ─────────────────────────────────────────
def _attachments_fallback(msg: email.message.EmailMessage)->List[Tuple[str,bytes]]:
    out=[]
    if not msg or not hasattr(msg,"walk"): return out
    for p in msg.walk():
        if (p.get_content_disposition() or "").startswith("attachment") or p.get_filename():
            name=p.get_filename() or f"part-{p.get_content_type()}"
            data=p.get_payload(decode=True)
            if data: out.append((name,data))
    return out

def _show_attachment(name:str,data:bytes):
    if not data:
        st.warning("Pièce jointe vide");return
    low=name.lower()
    if low.endswith((".png",".jpg",".jpeg",".gif",".bmp",".webp")):
        st.image(Image.open(io.BytesIO(data)),use_container_width=True)
    elif low.endswith(".pdf"):
        from pdf2image import convert_from_bytes
        try:
            st.image(convert_from_bytes(data,first_page=1,last_page=1,dpi=150)[0],
                     use_container_width=True)
        except Exception as e:
            st.error(f"Aperçu PDF impossible : {e}")
        href=f'data:application/pdf;base64,{base64.b64encode(data).decode()}'
        st.markdown(f'<a href="{href}" target="_blank" rel="noopener noreferrer">📄 Ouvrir le PDF</a>',
                    unsafe_allow_html=True)
    else:
        st.text(extractor.att_snippet(name,data,limit=20_000))
    st.download_button("Télécharger",data=data,file_name=name,key=f"dl_{name}")

# ───────── Libellés lisibles ──────────────────────────────────────
def human_label(lbl:str)->str:
    mapping={ "email uid":"Identifiant Email","received date":"Date de réception",
        "document types":"Types de document","departments":"Départements","priority":"Priorité",
        "confidentiality":"Confidentialité","from.name":"Nom expéditeur","from.email":"Email expéditeur",
        "from.entity type":"Type entité expéditeur","to.name":"Nom destinataire",
        "to.email":"Email destinataire","to.entity type":"Type entité destinataire",
        "attachments":"Pièces jointes","classification history":"Historique de classification"}
    return mapping.get(lbl.lower(), lbl.replace("_"," ").capitalize())


# ───────── Helpers Streamlit ────────────────────────────
def safe_rerun(scope: str = "app"):
    """
    Relance le script de façon compatible avec toutes les versions de Streamlit.

    • Streamlit ≥ 1.27  → utilise st.rerun()
    • Versions plus anciennes → bascule sur st.experimental_rerun()
    """
    if hasattr(st, "rerun"):
        # Streamlit récent
        st.rerun(scope=scope)
    elif hasattr(st, "experimental_rerun"):
        # Anciennes versions (< 1.27)
        st.experimental_rerun()
    else:
        # Cas très ancien : avertissement (le script continue sans crash)
        st.warning(
            "Votre version de Streamlit est trop ancienne pour supporter le rerun "
            "automatique. Mettez Streamlit à jour :  pip install -U streamlit"
        )

# ───────── Widget d’édition des métadonnées ───────────────────────
# ───────── Widget d’édition des métadonnées ───────────────────────
# ───────── Widget d’édition des métadonnées ───────────────────────
def display_editable_metadata(uid: str, meta: dict):
    """
    Affiche les métadonnées (lecture/édition) + remarques.
    Les modifications sont enregistrées puis l’app se ré-exécute pour rafraîchir.
    """
    # ----- JSON affiché/édité (priorité version utilisateur) ------------------
    data_json = meta.get("user_preclass_json") or meta.get("preclass_json") or {}

    flat    = flatten(data_json)
    cleaned = {k: v for k, v in flat.items() if not _is_empty(v)}

    # Toujours une chaîne
    remarks_txt: str = str(meta.get("remarks") or "")

    # ----- Bascule lecture/édition -------------------------------------------
    key_edit = f"edit_mode_{uid}"
    if key_edit not in st.session_state:
        st.session_state[key_edit] = False

    edit_mode = st.checkbox(
        "Modifier les métadonnées",
        value=st.session_state[key_edit],
        key=f"checkbox_{key_edit}",
    )

    if edit_mode != st.session_state[key_edit]:
        st.session_state[key_edit] = edit_mode
        safe_rerun()

    # ----- MODE ÉDITION -------------------------------------------------------
    if st.session_state[key_edit]:
        with st.form(f"form_meta_{uid}"):
            new_vals: dict[str, Any] = {}
            for k, v in cleaned.items():
                lbl = human_label(k)
                if isinstance(v, bool):
                    new_vals[k] = st.checkbox(lbl, value=v, key=f"{uid}_{k}")
                elif isinstance(v, (int, float)):
                    step = 1 if isinstance(v, int) else 0.01
                    new_vals[k] = st.number_input(lbl, value=v, step=step, key=f"{uid}_{k}")
                elif isinstance(v, str) and len(v) < 90 and "\n" not in v:
                    new_vals[k] = st.text_input(lbl, value=v, key=f"{uid}_{k}")
                else:
                    new_vals[k] = st.text_area(lbl, value=v, height=80, key=f"{uid}_{k}")

            new_remarks = st.text_area(
                "Remarques", value=remarks_txt, height=120, key=f"{uid}_remarks"
            )

            if st.form_submit_button("Enregistrer"):
                meta_update = {
                    "user_preclass_json": unflatten(new_vals),
                    "remarks": str(new_remarks or ""),
                    "status": meta.get("status", "done"),
                }
                storage.update_meta(uid, meta_update)
                st.session_state[key_edit] = False  # repasse en mode lecture
                safe_rerun()

    # ----- MODE LECTURE -------------------------------------------------------
    else:
        rows = []
        for k, v in cleaned.items():
            lbl = human_label(k)
            val_html = _display_value(v)

            # Si _display_value a généré un <table>, on ne l’entoure PAS de <code>
            if isinstance(val_html, str) and val_html.startswith("<table"):
                rows.append(
                    f"<tr><td><strong>{lbl}</strong></td><td>{val_html}</td></tr>"
                )
            else:
                val_lines = "<br>".join(str(val_html).split("\n"))
                rows.append(
                    f"<tr><td><strong>{lbl}</strong></td>"
                    f"<td><code>{val_lines}</code></td></tr>"
                )

        html_tbl = f"""
        <style>
        .elegant-compact {{
            border:1px solid #e1e4e8;
            border-radius:6px;
            box-shadow:0 1px 2px rgba(0,0,0,0.08);
            overflow:hidden;
        }}
        .elegant-compact table {{
            width:100%;
            border-collapse:collapse;
            font-family:'Segoe UI','Helvetica','Arial',sans-serif;
            font-size:12px;
        }}
        .elegant-compact thead th {{
            background-color:#fafbfc;
            color:#2c3e50;
            font-weight:600;
            padding:6px 8px;
            text-align:left;
        }}
        .elegant-compact th, .elegant-compact td {{
            padding:6px 8px;
            border-bottom:1px solid #ececec;
        }}
        .elegant-compact tbody tr:nth-child(even) {{
            background-color:#f4f6f8;
        }}
        .elegant-compact tbody tr:hover {{
            background-color:#e8edf2;
        }}
        .elegant-compact code {{
            background-color:#f0f2f5;
            padding:2px 4px;
            border-radius:3px;
            word-break:break-word;
            color:#2c3e50;
            font-size:11px;
        }}
        </style>
        <div class="elegant-compact">
        <table>
            <thead><tr><th>Métadonnée</th><th>Valeur</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
        </div>
        """
        components.html(
            html_tbl,
            height=min(600, 32 + 108 * len(cleaned)),
            scrolling=True,
        )

        if remarks_txt.strip():
            st.markdown("#### Remarques")
            # 👉 key unique par e-mail pour éviter la persistance
            st.text_area(
                " ",
                value=remarks_txt,
                disabled=True,
                height=120,
                label_visibility="collapsed",
                key=f"{uid}_remarks_readonly",
            )



# ═════════ Récupération des UID “done” ═════════════════════════════
@st.cache_data(ttl=300)
def _uids_done()->List[str]:
    rows=storage.get_all_meta(order="updated_at DESC")
    return [str(r[0]) for r in rows if len(r)>1 and r[1]=="done"]

# ───────── Session & sélection message ────────────────────────────
if "current_uid" not in st.session_state: st.session_state["current_uid"]=None
uids=_uids_done()
if not uids:
    st.sidebar.warning("Aucun e-mail marqué done."); st.stop()
if st.session_state["current_uid"] not in uids:
    st.session_state["current_uid"]=uids[0]

def _set_uid(u:str):
    st.session_state["current_uid"]=u
    st.rerun()

idx=uids.index(st.session_state["current_uid"])
sel=st.sidebar.selectbox("Sélectionnez un e-mail",uids,index=idx,key="sel_uid")
if sel!=st.session_state["current_uid"]:
    _set_uid(sel)

col_prev, col_next = st.sidebar.columns(2)
col_prev.button("⬅", disabled=idx == 0, 
                on_click=lambda: _set_uid(uids[idx - 1]))
col_next.button("➡", disabled=idx == len(uids) - 1, 
                on_click=lambda: _set_uid(uids[idx + 1]))

# ═════════ Chargement du mail ═════════════════════════════════════
@st.cache_data(ttl=300,show_spinner="Chargement…")
def _get_email(uid:str):
    try:
        msg,atts=imap_client.fetch_email(uid)
    except Exception:
        msg,atts=email.message.EmailMessage(),[]
    if not atts: atts=_attachments_fallback(msg)
    return msg,atts

msg,attachments=_get_email(st.session_state["current_uid"])

# ═════════ Layout principal ═══════════════════════════════════════
left,right=st.columns([2.8,1.2])

with left:
    st.subheader(f"✉️ {msg.get('subject') or '(sans objet)'}")
    st.write(f"**De :** {msg.get('from')} | **À :** {msg.get('to')} | {msg.get('date')}")
    st.markdown("### 📎 Pièces jointes")
    if not attachments:
        st.info("Aucune pièce jointe.")
    else:
        for n,d in attachments:
            with st.expander(n,expanded=False):
                _show_attachment(n,d)
    st.markdown("### Contenu")
    components.html(_html_with_inline_images(msg),height=400,scrolling=True)

with right:
    st.subheader("Classification")
    uid  = st.session_state['current_uid']
    meta = storage.get_meta(uid) or {}
    if not meta.get("preclass_json") and not meta.get("user_preclass_json"):
        st.info("Aucune classification disponible."); st.stop()
    display_editable_metadata(uid, meta)


# ═════════ Sidebar utils ══════════════════════════════════════════
st.sidebar.divider()
if st.sidebar.button("🔄 Vider cache"):
    st.cache_data.clear(); st.rerun()
