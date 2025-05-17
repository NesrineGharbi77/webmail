"""extractor – Text extraction from e‑mails & attachments (no external config).

All parameters come from environment variables or Streamlit *secrets* so that
no private `config.py` is required.  Designed for Streamlit Cloud deployment.
"""
from __future__ import annotations

# ─────────────────────────── Imports de base ─────────────────────────────
import os, io, re, tempfile, logging, unicodedata, pathlib, time, concurrent.futures
from pathlib import Path
from io import BytesIO
from typing import List, Tuple, Optional, Callable

import email
import chardet, magic, pytesseract, cv2, numpy as np
from pdf2image import convert_from_bytes
from PIL import Image, ImageFile

import docx, openpyxl
try:
    import xlrd           # .xls legacy
    _HAS_XLRD = True
except ImportError:
    _HAS_XLRD = False

try:
    import textract       # optional, heavy
    _HAS_TEXTRACT = True
except ImportError:
    _HAS_TEXTRACT = False

try:
    from ftfy import fix_text   # accent / unicode fixes
    _HAS_FTFY = True
except ImportError:
    _HAS_FTFY = False

import html2text
import streamlit as st

# ─────────────────────────── Logging & constants ─────────────────────────
logger = logging.getLogger("extractor")
ImageFile.LOAD_TRUNCATED_IMAGES = True

TMP = Path(tempfile.gettempdir()) / "extractor_tmp"
TMP.mkdir(parents=True, exist_ok=True)

# OCR & preprocessing parameters (env‑overridable)
OCR_LANGS         = os.getenv("OCR_LANGS", "fra+eng")
DESKEW_MAX_SKEW   = int(os.getenv("DESKEW_MAX_SKEW", 15))   # °
DESKEW_MIN_IMPROVE= float(os.getenv("DESKEW_MIN_IMPROVE", 0.05))  # +5 %
DESKEW_TIMEOUT    = int(os.getenv("DESKEW_TIMEOUT", 3))    # s

# PDF handling
POPPLER_PATH      = st.secrets.get("POPPLER_PATH", "/usr/bin")
PDF_PAGE_MAX      = int(os.getenv("PDF_PAGE_MAX", 2))       # first pages OCR

# Attachments
ATTACH_LIMIT      = int(os.getenv("ATTACH_LIMIT", 4000))    # chars returned per attachment

# ─────────────────────────── Text helpers ───────────────────────────────
_RX_SPACES = re.compile(r"\s+")
_RX_ALPHA  = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]")

def _norm(t: str) -> str:              return _RX_SPACES.sub(" ", t).strip()
def _nfc(t: str) -> str:               return unicodedata.normalize("NFC", t)
def _strip_ctrl(t: str) -> str:        return re.sub(r"[^\x20-\x7E\n\t]", "", t)

def clean_text(t: str) -> str:
    t = _nfc(_strip_ctrl(t))
    return fix_text(t) if _HAS_FTFY else t

# ───────── Heuristique de qualité OCR ─────────

def ocr_score(txt: str) -> float:
    if not txt:
        return 0.0
    total = len(txt)
    alpha = len(_RX_ALPHA.findall(txt))
    ratio = alpha / total
    words = [w for w in txt.split() if len(w) >= 3]
    dens  = len(words) / max(len(txt.split()), 1)
    return ratio * 0.7 + dens * 0.3

# ───────── Deskew helpers (Tesseract OSD + Hough) ─────────

def _osd_rotate(img_bgr: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    try:
        angle = int(pytesseract.image_to_osd(pil, output_type=pytesseract.Output.DICT).get("rotate", 0))
    except Exception:
        angle = 0
    if angle:
        h, w = img_bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
        img_bgr = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img_bgr

def _order4(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(1); diff = np.diff(pts, axis=1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                     pts[np.argmax(s)], pts[np.argmax(diff)]], dtype="float32")

def _warp_page(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img_bgr
    page = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(page, True)
    quad = cv2.approxPolyDP(page, 0.02 * peri, True)
    area_ratio = cv2.contourArea(page) / (img_bgr.shape[0] * img_bgr.shape[1])
    if len(quad) == 4 and 0.60 <= area_ratio <= 0.95:
        rect = _order4(quad.reshape(4, 2))
        (tl, tr, br, bl) = rect
        w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        img_bgr = cv2.warpPerspective(img_bgr, M, (w, h))
    return img_bgr

def _hough_skew(gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 120, minLineLength=120, maxLineGap=15)
    if lines is None:
        return gray
    angs = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in lines[:, 0]
            if abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) < 45]
    if not angs:
        return gray
    med = np.median(angs)
    if abs(med) < 0.5 or abs(med) > DESKEW_MAX_SKEW:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), med, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def deskew_pipeline(img_bgr: np.ndarray) -> np.ndarray:
    img_bgr = _osd_rotate(img_bgr)
    img_bgr = _warp_page(img_bgr)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray    = _hough_skew(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# ───────── OCR wrappers ─────────

def _ocr_img(img_bgr: np.ndarray) -> str:
    return pytesseract.image_to_string(img_bgr, lang=OCR_LANGS, config="--oem 3 --psm 6")

def preprocess_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img_raw = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    base_txt   = _ocr_img(img_raw)
    base_score = ocr_score(base_txt)

    def _deskew() -> np.ndarray:
        return deskew_pipeline(img_raw.copy())

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            img_deskew = ex.submit(_deskew).result(timeout=DESKEW_TIMEOUT)
    except Exception:
        img_deskew = img_raw

    desk_txt   = _ocr_img(img_deskew)
    desk_score = ocr_score(desk_txt)

    chosen = img_deskew if desk_score - base_score >= DESKEW_MIN_IMPROVE else img_raw
    gray   = cv2.cvtColor(chosen, cv2.COLOR_BGR2GRAY)
    thr    = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 35, 15)
    return thr

def _ocr_bytes(data: bytes) -> str:
    try:
        img  = preprocess_image_bytes(data)
        text = pytesseract.image_to_string(img, lang=OCR_LANGS, config="--oem 3 --psm 6")
        return _norm(text)
    except Exception as exc:
        logger.error("OCR Tesseract échoué : %s", exc)
        return ""

def _ocr_file(path: str | Path) -> str:
    try:
        return _ocr_bytes(Path(path).read_bytes())
    except Exception as exc:
        logger.warning("OCR fichier KO : %s", exc)
        return ""

# ---------------------------------------------------------------------
#                         HANDLERS ATTACHMENTS
# ---------------------------------------------------------------------

def _text_snippet(data: bytes, limit: int) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return _norm(data.decode(enc, errors="ignore"))[:limit]
        except Exception:
            pass
    return ""

import pandas as pd, zipfile

def _ods_snippet(data: bytes, limit: int) -> str:
    try:
        df = pd.read_excel(BytesIO(data), engine="odf", header=None, nrows=100)
        flat = df.astype(str).stack().tolist()[:2000]
        return _norm(" ".join(flat))[:limit]
    except Exception as exc:
        logger.warning("Erreur ODS : %s", exc)
        return ""

def _xlsx_snippet(data: bytes, limit: int) -> str:
    tf = tempfile.NamedTemporaryFile(dir=TMP, suffix=".xlsx", delete=False)
    tf.write(data); tf.close()
    try:
        wb = openpyxl.load_workbook(tf.name, data_only=True)
        cells: list[str] = []
        for ws in wb.worksheets[:2]:
            for row in ws.iter_rows(values_only=True):
                cells.extend([str(c) for c in row if c])
                if len(cells) > 2000:
                    break
        return _norm(" ".join(cells))[:limit]
    except Exception as exc:
        logger.warning("Erreur XLSX : %s", exc)
        return ""
    finally:
        pathlib.Path(tf.name).unlink(missing_ok=True)

def _xls_snippet(data: bytes, limit: int) -> str:
    if not _HAS_XLRD:
        logger.warning("xlrd absent – .xls ignoré")
        return ""
    tf = tempfile.NamedTemporaryFile(dir=TMP, suffix=".xls", delete=False)
    tf.write(data); tf.close()
    try:
        wb = xlrd.open_workbook(tf.name)
        cells: list[str] = []
        for sh in wb.sheets()[:2]:
            for r in range(sh.nrows):
                cells.extend([str(sh.cell_value(r, c)) for c in range(sh.ncols) if sh.cell_value(r, c)])
                if len(cells) > 2000:
                    break
        return _norm(" ".join(cells))[:limit]
    except Exception as exc:
        logger.warning("Erreur .xls : %s", exc)
        return ""
    finally:
        pathlib.Path(tf.name).unlink(missing
