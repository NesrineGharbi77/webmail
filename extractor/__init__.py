# backend_preclass/extractor/__init__.py
# =========================================================
"""
Extraction de texte (e‑mails + pièces jointes) avec OCR robuste.
"""

from __future__ import annotations

# ──────────────────── Imports de base ────────────────────
import os, io, re, tempfile, logging, unicodedata, pathlib, time, concurrent.futures
from pathlib import Path
from io import BytesIO
from typing import List, Tuple, Optional

from typing import Callable
import chardet, email, magic, pytesseract, cv2, numpy as np
from PIL import Image, ImageFile
from pdf2image import convert_from_bytes

import docx, openpyxl
try:
    import xlrd
    _HAS_XLRD = True
except ImportError:
    _HAS_XLRD = False

try:
    import textract
    _HAS_TEXTRACT = True
except ImportError:
    _HAS_TEXTRACT = False

try:
    from ftfy import fix_text
    _HAS_FTFY = True
except ImportError:
    _HAS_FTFY = False

import html2text

# ────────────────────── Logging & OCR ─────────────────────
logger = logging.getLogger("extractor")
ImageFile.LOAD_TRUNCATED_IMAGES = True
OCR_LANGS = os.getenv("OCR_LANGS", "fra+eng")

# ────────────────────── Constants ─────────────────────────
TMP.mkdir(parents=True, exist_ok=True)

DESKEW_MAX_SKEW     = getattr("DESKEW_MAX_SKEW", 15)       # °
DESKEW_MIN_IMPROVE  = getattr("DESKEW_MIN_IMPROVE", 0.05)  # +5 %
DESKEW_TIMEOUT      = getattr("DESKEW_TIMEOUT", 3)         # s

# ────────────────────── Utils texte ───────────────────────
_RX_SPACES = re.compile(r"\s+")
def _norm(t: str) -> str:              return _RX_SPACES.sub(" ", t).strip()
def _nfc(t: str) -> str:               return unicodedata.normalize("NFC", t)
def _strip_ctrl(t: str) -> str:        return re.sub(r"[^\x20-\x7E\n\t]", "", t)
def clean_text(t: str) -> str:
    t = _nfc(_strip_ctrl(t))
    return fix_text(t) if _HAS_FTFY else t

# ─────────── Heuristique de « qualité » OCR ───────────────
_RX_ALPHA = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]")
def ocr_score(txt: str) -> float:
    if not txt:
        return 0.0
    total = len(txt)
    alpha = len(_RX_ALPHA.findall(txt))
    ratio = alpha / total
    words = [w for w in txt.split() if len(w) >= 3]
    dens  = len(words) / max(len(txt.split()), 1)
    return ratio * 0.7 + dens * 0.3        # [0‑1]

# ──────────────── Deskew helpers ──────────────────────────
def _osd_rotate(img_bgr: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    angle = int(pytesseract.image_to_osd(pil, output_type=pytesseract.Output.DICT).get("rotate", 0))
    if angle:
        h, w = img_bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), -angle, 1.0)
        img_bgr = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img_bgr

def _order4(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(1); diff = np.diff(pts, axis=1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                     pts[np.argmax(s)], pts[np.argmax(diff)]], dtype="float32")

def _warp_page(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return img_bgr
    page = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(page, True)
    quad = cv2.approxPolyDP(page, 0.02*peri, True)
    area_ratio = cv2.contourArea(page)/(img_bgr.shape[0]*img_bgr.shape[1])
    if len(quad)==4 and 0.60<=area_ratio<=0.95:
        rect = _order4(quad.reshape(4,2))
        (tl,tr,br,bl)=rect
        w = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
        h = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
        dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        img_bgr = cv2.warpPerspective(img_bgr, M, (w, h))
    return img_bgr

def _hough_skew(gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray, 50,150,apertureSize=3)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,120,minLineLength=120,maxLineGap=15)
    if lines is None: return gray
    angs=[np.degrees(np.arctan2(y2-y1,x2-x1))
          for x1,y1,x2,y2 in lines[:,0] if abs(np.degrees(np.arctan2(y2-y1,x2-x1)))<45]
    if not angs: return gray
    med = np.median(angs)
    if abs(med) < 0.5 or abs(med) > DESKEW_MAX_SKEW: return gray
    h,w=gray.shape[:2]
    M=cv2.getRotationMatrix2D((w/2,h/2), med,1.0)
    return cv2.warpAffine(gray, M,(w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def deskew_pipeline(img_bgr: np.ndarray) -> np.ndarray:
    img_bgr = _osd_rotate(img_bgr)
    img_bgr = _warp_page(img_bgr)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray    = _hough_skew(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# ──────────────── OCR wrappers ───────────────────────────
def _ocr_img(img_bgr: np.ndarray) -> str:
    return pytesseract.image_to_string(img_bgr, lang=OCR_LANGS,
                                       config="--oem 3 --psm 6")

def preprocess_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img_raw = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # OCR « baseline »
    base_txt   = _ocr_img(img_raw)
    base_score = ocr_score(base_txt)

    # Deskew avec timeout
    def _deskew(): return deskew_pipeline(img_raw.copy())
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            img_deskew = ex.submit(_deskew).result(timeout=DESKEW_TIMEOUT)
    except Exception:
        img_deskew = img_raw

    desk_txt   = _ocr_img(img_deskew)
    desk_score = ocr_score(desk_txt)

    chosen = img_deskew if desk_score - base_score >= DESKEW_MIN_IMPROVE else img_raw
    gray   = cv2.cvtColor(chosen, cv2.COLOR_BGR2GRAY)
    thr    = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY,35,15)
    return thr

def _ocr_bytes(data: bytes) -> str:
    try:
        img  = preprocess_image_bytes(data)
        text = pytesseract.image_to_string(img, lang=OCR_LANGS, config="--oem 3 --psm 6")
        return _norm(text)
    except Exception as exc:
        logger.error("OCR Tesseract échoué : %s", exc)
        return ""

def _ocr_file(path: str|Path) -> str:
    try:
        return _ocr_bytes(Path(path).read_bytes())
    except Exception as exc:
        logger.warning("OCR fichier KO : %s", exc)
        return ""

# -----------------------------------------------------------------
#                         HANDLERS DIVERS
# -----------------------------------------------------------------
def _text_snippet(data: bytes, limit: int) -> str:
    for enc in ("utf-8","latin-1"):
        try:
            return _norm(data.decode(enc,errors="ignore"))[:limit]
        except Exception: pass
    return ""

def _ods_snippet(data: bytes, limit: int) -> str:
    """
    Extrait rapidement le texte d’un classeur LibreOffice (.ods).
    Lit au plus ~2 000 cellules pour rester léger.
    Dépendances : pandas >= 1.5  et  odfpy.
    """
    import pandas as pd, io
    try:
        # header=None ➜ aucune ligne d’en‑tête obligatoire
        df = pd.read_excel(io.BytesIO(data), engine="odf", header=None, nrows=100)
        flat = df.astype(str).stack().tolist()[:2000]
        return _norm(" ".join(flat))[:limit]
    except Exception as exc:
        logger.warning("Erreur ODS : %s", exc)
        return ""


def _xlsx_snippet(data: bytes, limit: int) -> str:
    tf = tempfile.NamedTemporaryFile(dir=TMP, suffix=".xlsx", delete=False)
    tf.write(data); tf.close()
    try:
        wb=openpyxl.load_workbook(tf.name,data_only=True)
        cells=[]
        for ws in wb.worksheets[:2]:
            for row in ws.iter_rows(values_only=True):
                cells.extend([str(c) for c in row if c])
                if len(cells)>2000: break
        return _norm(" ".join(cells))[:limit]
    except Exception as exc:
        logger.warning("Erreur XLSX : %s", exc); return ""
    finally:
        pathlib.Path(tf.name).unlink(missing_ok=True)

def _xls_snippet(data: bytes, limit: int) -> str:
    if not _HAS_XLRD:
        logger.warning("xlrd absent – .xls ignoré"); return ""
    tf=tempfile.NamedTemporaryFile(dir=TMP,suffix=".xls",delete=False)
    tf.write(data); tf.close()
    try:
        wb=xlrd.open_workbook(tf.name)
        cells=[]
        for sh in wb.sheets()[:2]:
            for r in range(sh.nrows):
                cells.extend([str(sh.cell_value(r,c)) for c in range(sh.ncols) if sh.cell_value(r,c)])
                if len(cells)>2000: break
        return _norm(" ".join(cells))[:limit]
    except Exception as exc:
        logger.warning("Erreur .xls : %s", exc); return ""
    finally:
        pathlib.Path(tf.name).unlink(missing_ok=True)

def _pdf_snippet(data: bytes, limit: int=config.ATTACH_LIMIT) -> str:
    try:
        pages=convert_from_bytes(data,dpi=200,poppler_path=config.POPPLER_PATH)
    except Exception as exc:
        logger.error("PDF→image KO : %s", exc); return ""
    txt=[]
    for p in pages[:config.PDF_PAGE_MAX]:
        buf=BytesIO(); p.save(buf,format="PNG"); buf.seek(0)
        txt.append(_ocr_bytes(buf.getvalue()))
    return _norm(" ".join(txt))[:limit]

def _image_snippet(data: bytes, limit: int) -> str:
    return _norm(_ocr_bytes(data))[:limit]

def _docx_snippet(data: bytes, limit: int) -> str:
    tf=tempfile.NamedTemporaryFile(dir=TMP,suffix=".docx",delete=False)
    tf.write(data); tf.close()
    try:
        txt=" ".join(par.text for par in docx.Document(tf.name).paragraphs)
        return _norm(txt)[:limit]
    finally:
        pathlib.Path(tf.name).unlink(missing_ok=True)

# -----------------------------------------------------------------
#                    ROUTAGE PIÈCES JOINTES
# -----------------------------------------------------------------
import zipfile

def _handle_zip(name: str, data: bytes, limit: int) -> str:
    texts: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                try:
                    member_data = z.read(info.filename)
                    texts.append(  # appel récursif sur chaque fichier
                        att_snippet(info.filename, member_data, limit)
                    )
                except Exception as e:
                    logger.warning(
                        "Impossible de lire %s dans %s: %s",
                        info.filename, name, e
                    )
    except zipfile.BadZipFile as e:
        logger.error("PJ ZIP corrompue %s: %s", name, e)
    return " ".join(texts)


def att_snippet(
    name: str,
    data: bytes,
    limit: int = config.ATTACH_LIMIT
) -> str:
    """
    Retourne un extrait textuel d’une pièce jointe.
    - Ignore **complètement** les ZIP.
    """
    if not data:
        logger.warning("PJ « %s » vide", name)
        return ""

    ext = Path(name).suffix.lower()

    # ----- 1) SKIP TOTAL DES ZIP -----
    if ext == ".zip":
        logger.info("PJ ZIP ignorée: %s", name)
        return ""

    # ----- 2) Handlers directs par extension -----
    handlers: dict[str, Callable[[bytes, int], str] | None] = {
        ".pdf":  _pdf_snippet,
        ".docx": _docx_snippet,
        ".doc":  _docx_snippet,
        ".xlsx": _xlsx_snippet,
        ".xls":  _xls_snippet,
        ".ods":  _ods_snippet,
        ".txt":  _text_snippet,
        ".md":   _text_snippet,
        ".csv":  _text_snippet,
        ".png":  _image_snippet,
        ".jpg":  _image_snippet,
        ".jpeg": _image_snippet,
        ".bmp":  _image_snippet,
        ".gif":  _image_snippet,
        ".tif":  _image_snippet,
        ".tiff": _image_snippet,
    }

    raw: str | None = None

    if ext in handlers:
        handler = handlers[ext]
        if handler is None:
            logger.info("PJ « %s » : extension reconnue mais ignorée", name)
            return ""
        try:
            raw = handler(data, limit)
        except Exception as e:
            logger.warning("Erreur PJ « %s » (handler %s) : %s", name, ext, e)

    # ----- 3) Fallback MIME -----
    if raw is None:
        mime = magic.from_buffer(data, mime=True)
        try:
            if mime.startswith("text/"):
                raw = data.decode("utf-8", errors="ignore")
            elif mime == "application/pdf":
                raw = _pdf_snippet(data, limit)
            elif mime.startswith("image/"):
                raw = _image_snippet(data, limit)
            elif mime in (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword",
            ):
                raw = _docx_snippet(data, limit)
            elif mime == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                raw = _xlsx_snippet(data, limit)
            elif mime == "application/vnd.ms-excel":
                raw = _xls_snippet(data, limit)
        except Exception as e:
            logger.warning("Fallback MIME KO : %s", e)

    # ----- 4) Optionnel Textract -----
    if raw is None and _HAS_TEXTRACT and os.getenv("ENABLE_TEXTRACT", "0") == "1":
        try:
            raw = textract.process(data).decode("utf-8", errors="ignore")
        except Exception as e:
            logger.warning("Textract KO pour PJ « %s » : %s", name, e)

    # ----- 5) Bilan -----
    if raw is None:
        mime = magic.from_buffer(data, mime=True)
        logger.info(
            "PJ « %s » : type non géré (%s / %s)",
            name,
            ext or "aucune",
            mime,
        )
        return ""

    return clean_text(raw)[:limit]



# -----------------------------------------------------------------
#                      BODY → TEXT
# -----------------------------------------------------------------
def body_to_text(msg: email.message.EmailMessage,
                 prefer: Tuple[str,...]=("plain","html")) -> str:
    def _extract(part, fmt):
        try:
            cont=part.get_content()
            if isinstance(cont,(bytes,bytearray)):
                cont=cont.decode(chardet.detect(cont).get("encoding") or "utf-8", errors="ignore")
            return _norm(html2text.html2text(cont) if fmt=="html" else cont)
        except Exception as e:
            logger.warning("Extract %s KO : %s", fmt, e); return ""
    buckets={"plain":[],"html":[]}
    for part in (msg.walk() if msg.is_multipart() else [msg]):
        ct=part.get_content_type()
        if ct=="text/plain": buckets["plain"].append(_extract(part,"plain"))
        elif ct=="text/html": buckets["html"].append(_extract(part,"html"))
    for fmt in prefer:
        joined=" ".join(filter(None,buckets[fmt])).strip()
        if joined: return joined[:limit]
    try:
        raw=msg.get_payload(decode=True) or b""
        return _norm(raw.decode("utf-8",errors="ignore"))[:limit]
    except Exception as e:
        logger.error("Payload brut KO : %s", e); return ""
