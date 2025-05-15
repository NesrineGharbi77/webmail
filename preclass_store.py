# storage.py
import sqlite3, json, pathlib, datetime
from typing import List, Dict, Tuple, Any

DB_PATH = pathlib.Path(__file__).with_name("preclass.sqlite3")

###############################################################################
# Helpers
###############################################################################
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn

###############################################################################
# API publique utilisée par streamlit_app.py
###############################################################################
def get_all_meta(order: str = "updated_at DESC") -> List[Tuple[str, str, str]]:
    """Retourne [(uid, status, updated_at), …] triés selon *order*."""
    with _connect() as c:
        cur = c.execute(f"""
            SELECT email_id, status, updated_at
              FROM email_meta
          ORDER BY {order}
        """)
        return [ (str(r["email_id"]), r["status"], r["updated_at"]) for r in cur ]

def get_meta(uid: str) -> Dict[str, Any]:
    """
    Retourne un dict comprenant :
        status, preclass_json, user_preclass_json, remarks, error_msg, updated_at
    Colonnes créées à la volée si manquantes.
    """
    with _connect() as c:
        _ensure_column(c, "user_preclass_json", "TEXT")
        _ensure_column(c, "remarks",            "TEXT")

        cur = c.execute("""
            SELECT status, preclass_json, user_preclass_json,
                   remarks, error_msg, updated_at
              FROM email_meta
             WHERE email_id = ?
        """, (uid,))
        row = cur.fetchone()
        if not row:
            return {}

        return {
            "status":              row["status"],
            "preclass_json":       json.loads(row["preclass_json"] or "{}"),
            "user_preclass_json":  json.loads(row["user_preclass_json"] or "{}"),
            "remarks":             row["remarks"],
            "error_msg":           row["error_msg"],
            "updated_at":          row["updated_at"],
        }


def save_user_classification(uid: str, classification: Dict[str, Any]) -> None:
    """Écrit la classification validée par l’utilisateur dans la même colonne JSON."""
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    with _connect() as c:
        c.execute("""
            UPDATE email_meta
               SET preclass_json = ?, status = 'done', updated_at = ?
             WHERE email_id = ?
        """, (json.dumps(classification, ensure_ascii=False), ts, uid))
        c.commit()

###############################################################################
# Mise à jour complète (métadonnées + remarques) =============================
###############################################################################
def _ensure_column(conn: sqlite3.Connection, col_name: str, col_def: str) -> None:
    """Ajoute la colonne *col_name* si elle n’existe pas encore."""
    cur = conn.execute("PRAGMA table_info(email_meta)")
    if col_name not in [r["name"] for r in cur.fetchall()]:
        conn.execute(f"ALTER TABLE email_meta ADD COLUMN {col_name} {col_def}")
        conn.commit()

def update_meta(uid: str, meta: Dict[str, Any]) -> None:
    """
    Écrit (ou insère) les champs passés dans *meta* sans modifier
    le JSON d'origine (preclass_json).

    Clés attendues :
        - user_preclass_json : dict    ← corrections utilisateur
        - remarks            : str|None
        - status             : str     (optionnel, défaut : 'done')
        - error_msg          : str|None
    """
    ts      = datetime.datetime.now().isoformat(timespec="seconds")
    userjs  = json.dumps(meta.get("user_preclass_json", {}),
                         ensure_ascii=False)
    remarks = meta.get("remarks")
    status  = meta.get("status", "done")
    err     = meta.get("error_msg")

    with _connect() as c:
        # Création auto des colonnes si besoin
        _ensure_column(c, "user_preclass_json", "TEXT")
        _ensure_column(c, "remarks",             "TEXT")

        cur = c.execute("""
            UPDATE email_meta
               SET user_preclass_json = ?,
                   remarks            = ?,
                   status             = ?,
                   error_msg          = ?,
                   updated_at         = ?
             WHERE email_id = ?
        """, (userjs, remarks, status, err, ts, uid))

        if cur.rowcount == 0:                         # pas de ligne → INSERT
            c.execute("""
                INSERT INTO email_meta (email_id, user_preclass_json,
                                         remarks, status, error_msg, updated_at)
                VALUES (?,?,?,?,?,?)
            """, (uid, userjs, remarks, status, err, ts))
        c.commit()
