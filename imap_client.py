"""imap_client.py – robust UID‑based fetching with full attachment coverage"""

import imaplib
# augmente la limite par ligne pour éviter les messages tronqués
imaplib._MAXLINE: int = 10_000_000

import time
import logging
from contextlib import contextmanager
from typing import List, Tuple, Optional

import email
from email.policy import default

import config

logger = logging.getLogger("imap_client")

# ───────────────────── Connexion IMAP robuste ──────────────────────────
@contextmanager
def imap_conn():
    """Context‑manager IMAP avec retry, un seul yield, et cleanup garanti."""
    attempt = 0
    conn = None
    while True:  # la boucle ne précède *que* la tentative de login
        try:
            conn = (
                imaplib.IMAP4_SSL(config.IMAP_HOST, config.IMAP_PORT)
                if config.IMAP_SSL
                else imaplib.IMAP4(config.IMAP_HOST, config.IMAP_PORT)
            )
            conn.login(config.IMAP_USER, config.IMAP_PASS)
            break  # on sort de la boucle → connexion OK
        except imaplib.IMAP4.error as e:
            attempt += 1
            if attempt > config.IMAP_MAX_RETRY:
                logger.error("IMAP login failed after %d attempts: %s", attempt, e)
                raise
            logger.warning(
                "IMAP login failed (%s), retry %d/%d", e, attempt, config.IMAP_MAX_RETRY
            )
            time.sleep(5 * attempt)

    try:
        yield conn  # *seul* yield – aucune autre boucle après
    finally:
        if conn is not None:
            try:
                conn.logout()
            except Exception:
                pass

# ──────────────────────── Listing des UIDs ────────────────────────────
def list_uids_since(last_uid: Optional[str] = None) -> List[str]:
    """Retourne tous les UIDs strictement après *last_uid* (ou tous si None)."""
    with imap_conn() as conn:
        typ, _ = conn.select(config.MAILBOX, readonly=True)
        if typ != "OK":
            logger.error("SELECT %s failed: %s", config.MAILBOX, typ)
            return []

        if last_uid is None:
            typ, data = conn.uid("search", None, "ALL")
        else:
            start = int(last_uid) + 1
            typ, data = conn.uid("search", None, f"{start}:*")

        if typ != "OK":
            logger.error("UID SEARCH failed: %s %s", typ, data)
            return []
        return [uid.decode() for uid in data[0].split() if uid]

# ──────────────────────── Récupération message + PJ ────────────────────
def fetch_email(uid: str) -> Tuple[Optional[email.message.EmailMessage], List[Tuple[str, bytes]]]:
    """Retourne le message complet + liste (filename, bytes) de *toutes* les pièces."""
    with imap_conn() as conn:
        if conn.select(config.MAILBOX, readonly=True)[0] != "OK":
            logger.error("SELECT %s failed for UID %s", config.MAILBOX, uid)
            return None, []

        try:
            typ, data = conn.uid("fetch", uid, "(BODY.PEEK[])")
        except imaplib.IMAP4.error as e:
            logger.warning("IMAP UID fetch failed for %s: %s", uid, e)
            return None, []

        if typ != "OK" or not data or data[0] is None:
            logger.warning("UID %s – fetch gave %s %s", uid, typ, data)
            return None, []

        raw = data[0][1]
        msg = email.message_from_bytes(raw, policy=default)

        attachments: List[Tuple[str, bytes]] = []
        idx = 0
        for part in msg.walk():
            if part.is_multipart():
                continue

            # Cherche un nom
            filename = part.get_filename()
            disp = (part.get("Content-Disposition", "").lower())

            # on veut *tout* : "attachment" OU "inline" avec un payload binaire
            if not filename:
                if "attachment" in disp or "inline" in disp or part.get_content_type().startswith("image/"):
                    ext = part.get_content_type().split("/")[-1] or "bin"
                    filename = f"attachment_{uid}_{idx}.{ext}"
                else:
                    continue  # texte inline sans nom → pas une PJ exploitable

            payload = part.get_payload(decode=True) or b""
            if payload:
                attachments.append((filename, payload))
                idx += 1

        return msg, attachments