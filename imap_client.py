"""imap_client.py – Robust UID‑based IMAP fetching (Streamlit Cloud ready)."""

import imaplib
# Increase line length limit to prevent truncated responses
imaplib._MAXLINE = 10_000_000

import time
import logging
from contextlib import contextmanager
from typing import List, Tuple, Optional

import email
from email.policy import default
import streamlit as st

# ──────────────────────────── Configuration ────────────────────────────
EMAIL_USER = st.secrets["EMAIL_USER"]
EMAIL_PASS = st.secrets["EMAIL_PASS"]
IMAP_HOST  = st.secrets["IMAP_HOST"]
IMAP_PORT  = int(st.secrets.get("IMAP_PORT", 993))
IMAP_SSL   = bool(int(st.secrets.get("IMAP_SSL", "1")))  # "1" → True
MAILBOX    = st.secrets.get("MAILBOX", "INBOX")

IMAP_MAX_RETRY = int(st.secrets.get("IMAP_MAX_RETRY", 3))

logger = logging.getLogger("imap_client")

# ──────────────────────────── IMAP connection ──────────────────────────
@contextmanager
def imap_conn():
    """Yield a logged‑in IMAP connection with automatic retry & cleanup."""
    attempt = 0
    conn: Optional[imaplib.IMAP4] = None
    while True:
        try:
            conn = (
                imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
                if IMAP_SSL else imaplib.IMAP4(IMAP_HOST, IMAP_PORT)
            )
            conn.login(EMAIL_USER, EMAIL_PASS)
            break  # success
        except imaplib.IMAP4.error as e:
            attempt += 1
            if attempt > IMAP_MAX_RETRY:
                logger.error("IMAP login failed after %d attempts: %s", attempt, e)
                raise
            logger.warning("IMAP login failed (%s), retry %d/%d", e, attempt, IMAP_MAX_RETRY)
            time.sleep(5 * attempt)

    try:
        yield conn
    finally:
        if conn is not None:
            try:
                conn.logout()
            except Exception:
                pass

# ──────────────────────────── UID utilities ────────────────────────────
def list_uids_since(last_uid: Optional[str] = None) -> List[str]:
    """Return all UIDs strictly after *last_uid* (or all if None)."""
    with imap_conn() as conn:
        if conn.select(MAILBOX, readonly=True)[0] != "OK":
            logger.error("SELECT %s failed", MAILBOX)
            return []

        criterion = "ALL" if last_uid is None else f"{int(last_uid) + 1}:*"
        typ, data = conn.uid("search", None, criterion)
        if typ != "OK":
            logger.error("UID SEARCH failed: %s %s", typ, data)
            return []
        return [uid.decode() for uid in data[0].split() if uid]

# ───────────────────── Fetch full message + attachments ─────────────────────
def fetch_email(uid: str) -> Tuple[Optional[email.message.EmailMessage], List[Tuple[str, bytes]]]:
    """Fetch the email (RFC‑822) and all attachments as (filename, bytes)."""
    with imap_conn() as conn:
        if conn.select(MAILBOX, readonly=True)[0] != "OK":
            logger.error("SELECT %s failed for UID %s", MAILBOX, uid)
            return None, []

        try:
            typ, data = conn.uid("fetch", uid, "(BODY.PEEK[])")
        except imaplib.IMAP4.error as e:
            logger.warning("IMAP UID fetch failed for %s: %s", uid, e)
            return None, []

        if typ != "OK" or not data or data[0] is None:
            logger.warning("UID %s – fetch returned %s %s", uid, typ, data)
            return None, []

        raw_msg = data[0][1]
        msg = email.message_from_bytes(raw_msg, policy=default)

        attachments: List[Tuple[str, bytes]] = []
        idx = 0
        for part in msg.walk():
            if part.is_multipart():
                continue

            filename = part.get_filename()
            disp = (part.get("Content-Disposition") or "").lower()

            # Accept true attachments or inline binary parts (e.g., images)
            if not filename:
                if ("attachment" in disp or "inline" in disp or
                        part.get_content_type().startswith("image/")):
                    ext = part.get_content_type().split("/")[-1] or "bin"
                    filename = f"attachment_{uid}_{idx}.{ext}"
                else:
                    continue  # skip plain inline text

            payload = part.get_payload(decode=True) or b""
            if payload:
                attachments.append((filename, payload))
                idx += 1

        return msg, attachments
