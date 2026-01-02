# Author: Pedro Yáñez Meléndez
# Multimodal agent in Google Colab (no API keys). Run top-to-bottom.
# Skills: image understanding, PDF summarization/Q&A, CBT-style coaching, intent routing, Gradio UI.

# Install deps (run once)
import sys, subprocess, importlib, os, json, re, hashlib, time, urllib.request
from pathlib import Path


# Required output keys for the general agent skill
AGENT_KEYS = ["plan", "answer", "self_check", "confidence", "next_actions"]

# Output schemas (required keys)
DOCQA_KEYS = ["answer", "evidence", "limits", "confidence", "next_actions"]
MH_KEYS = ["situation", "goals", "cbt_exercises", "anger_tools", "conflict_script", "daily_5min_plan", "warnings", "confidence"]

def _pip_install(pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)

def ensure_deps():
    need = []
    for mod, pkg in [
        ("torch", "torch"),
        ("transformers", "transformers>=4.45.0"),
        ("accelerate", "accelerate"),
        ("pypdf", "pypdf"),
        ("fitz", "pymupdf"),
        ("pdfplumber", "pdfplumber"),
        ("PIL", "pillow"),
        ("gradio", "gradio>=4.0.0"),
        ("numpy", "numpy"),
    ]:
        try:
            importlib.import_module(mod)
        except Exception:
            need.append(pkg)
    if need:
        _pip_install(need)

ensure_deps()

import logging
LOGGER = logging.getLogger("multi_modal_agent")
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    _h.setFormatter(_fmt)
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.DEBUG if os.environ.get("AGENT_DEBUG", "0") == "1" else logging.INFO)

def log_event(event: str, **fields):
    try:
        payload = {"event": event, **fields}
        LOGGER.info(json.dumps(payload, ensure_ascii=False))
    except Exception:
        LOGGER.info(f"{event} | {fields}")

# Imports (after install)
import numpy as np
import torch
from PIL import Image
from pypdf import PdfReader
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

DATA_DIR = Path("/content/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load model
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    return processor, model

PROCESSOR, MODEL = load_model()
print("OK | model loaded:", MODEL_ID)

# JSON extraction (handle extra text before/after JSON)
def extract_json_candidates(text: str):
    cands = []
    stack = []
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if not stack:
                start = i
            stack.append(ch)
        elif ch == "}" and stack:
            stack.pop()
            if not stack and start is not None:
                blob = text[start:i+1]
                try:
                    cands.append(json.loads(blob))
                except Exception:
                    pass
    return cands

def extract_best_json(text: str, required_keys: list[str] | None = None):
    required_keys = required_keys or []
    cands = extract_json_candidates(text)
    for obj in cands:
        if isinstance(obj, dict) and all(k in obj for k in required_keys):
            return obj
    if cands:
        return cands[-1]
    raise ValueError("No JSON object found in model output.")

def _clamp01(x):
    try:
        return float(max(0.0, min(1.0, float(x))))
    except Exception:
        return 0.0

# LLM call helper (return dict)
def llm_agent(system_text: str, payload: dict, required_keys: list[str], image: Image.Image | None = None, max_new_tokens: int = 512):
    messages = [{"role": "system", "content": system_text}]
    user_content = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
    if image is not None:
        user_content.insert(0, {"type": "image", "image": image})
    messages.append({"role": "user", "content": user_content})

    prompt = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if image is None:
        inputs = PROCESSOR(text=[prompt], return_tensors="pt")
    else:
        inputs = PROCESSOR(text=[prompt], images=[image], return_tensors="pt")

    inputs = {k: v.to(MODEL.device) for k, v in inputs.items()}
    with torch.no_grad():
        gen = MODEL.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    out_text = PROCESSOR.batch_decode(gen, skip_special_tokens=True)[0]
    obj = extract_best_json(out_text, required_keys)
    if not isinstance(obj, dict):
        raise ValueError("Model output JSON is not an object.")
    return obj

# Download URL helper (avoid .bin issue)
def download_url(url: str, out_dir: Path = DATA_DIR) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        content_type = (r.headers.get("Content-Type") or "").lower()
        data = r.read()

    # Infer extension
    path = urllib.request.urlparse(url).path
    ext = Path(path).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".pdf"]:
        if "pdf" in content_type:
            ext = ".pdf"
        elif "image/" in content_type:
            ext = ".jpg"
        else:
            ext = ".bin"

    name = "download_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:10] + ext
    out_path = out_dir / name
    out_path.write_bytes(data)
    return str(out_path)

# Analyze image
def analyze_image(input_path: str) -> dict:
    img = Image.open(input_path).convert("RGB")
    keys = ["summary", "key_points", "entities", "tasks"]

    system_text = (
        "Return ONLY valid JSON. No prose, no markdown.\n"
        "Extract a concise summary plus key points and entities.\n"
        "Keep arrays short."
    )
    payload = {
        "input_type": "image",
        "template": {"summary": "", "key_points": ["", ""], "entities": ["", ""], "tasks": ["", ""]},
        "rules": [
            "summary must be 1–2 sentences.",
            "key_points must be a JSON array of strings.",
            "entities must be a JSON array of strings.",
            "tasks may be empty if none.",
        ],
    }
    obj = llm_agent(system_text, payload, keys, image=img, max_new_tokens=400)
    return {
        "input_type": "image",
        "summary": str(obj.get("summary", "")).strip(),
        "key_points": [str(x) for x in (obj.get("key_points") or []) if str(x).strip()][:10],
        "entities": [str(x) for x in (obj.get("entities") or []) if str(x).strip()][:12],
        "tasks": [str(x) for x in (obj.get("tasks") or []) if str(x).strip()][:10],
    }

# Extract text from PDF (simple, no OCR)
def _read_pdf_text(pdf_path: str, max_pages: int = 3, max_chars: int = 12000) -> str:
    """
    Extract selectable text from a PDF quickly.
    - Prefer PyMuPDF (fitz), usually faster + better layout than other libs.
    - Fall back to pypdf, then pdfplumber.
    Limits pages and characters to keep latency low for large PDFs.
    """
    txt_parts = []

    # 1) PyMuPDF
    try:
        import fitz
        doc = fitz.open(pdf_path)
        n = min(len(doc), max_pages)
        for i in range(n):
            page = doc.load_page(i)
            t = page.get_text("text") or ""
            if t.strip():
                txt_parts.append(t)
            if sum(len(x) for x in txt_parts) >= max_chars:
                break
        doc.close()
        out = "\n".join(txt_parts).strip()
        if out:
            return out[:max_chars]
    except Exception:
        pass

    # 2) pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        n = min(len(reader.pages), max_pages)
        for i in range(n):
            t = reader.pages[i].extract_text() or ""
            if t.strip():
                txt_parts.append(t)
            if sum(len(x) for x in txt_parts) >= max_chars:
                break
        out = "\n".join(txt_parts).strip()
        if out:
            return out[:max_chars]
    except Exception:
        pass

    # 3) pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            n = min(len(pdf.pages), max_pages)
            for i in range(n):
                t = pdf.pages[i].extract_text() or ""
                if t.strip():
                    txt_parts.append(t)
                if sum(len(x) for x in txt_parts) >= max_chars:
                    break
        out = "\n".join(txt_parts).strip()
        if out:
            return out[:max_chars]
    except Exception:
        pass

    return ""


def _summarize_text(text: str, max_chars: int = 700, max_words: int | None = None) -> str:
    """Fast, deterministic fallback summary for extracted PDF/text content.

    This avoids long latency + "empty answer" issues when asking the local LLM
    to summarize raw PDF text.
    """
    text = (text or "").strip()
    if not text:
        return "(empty document)"

    # Normalize whitespace and keep a clean slice.
    cleaned = re.sub(r"\s+", " ", text)
    cleaned = cleaned[: max_chars * 3]  # extra headroom before sentence splitting
    if max_words is not None:
        cleaned = " ".join(cleaned.split()[: int(max_words)])

    # Split into rough sentences.
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return cleaned[:max_chars]

    # Take the first 2–3 informative sentences.
    out = []
    for p in parts:
        if len(" ".join(out) + " " + p) > max_chars:
            break
        out.append(p)
        if len(out) >= 3:
            break

    return " ".join(out)[:max_chars]


def _key_points_from_text(text: str, max_points: int = 5, k: int | None = None, **kwargs) -> list[str]:
    """Extract a few short key points from text (no LLM call).

    Accept both `max_points=` and `k=` to stay compatible with older patches.
    Any extra kwargs are ignored.
    """
    if k is not None:
        max_points = int(k)
    if not text:
        return []
    # Prefer existing bullet-like lines
    raw_lines = [ln.strip() for ln in text.splitlines()]
    bullets = []
    for ln in raw_lines:
        if not ln:
            continue
        if ln.startswith(("-", "*", "•")):
            item = ln.lstrip("-*•").strip()
            if item:
                bullets.append(item)
    candidates = bullets

    if not candidates:
        # Fallback: split into sentences
        cleaned = re.sub(r"\s+", " ", text).strip()
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        candidates = [p.strip() for p in parts if p.strip()]

    out = []
    seen = set()
    for c in candidates:
        c = re.sub(r"\s+", " ", c).strip()
        if not c:
            continue
        if len(c) > 220:
            c = c[:217] + "..."
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= max_points:
            break
    return out

def _render_pdf_page_to_image(pdf_path: str, page_index: int = 0, zoom: float = 2.0) -> str:
    """
    Render a PDF page to a temporary PNG file and return its path.
    Uses PyMuPDF (fitz) so no Poppler is needed.
    """
    import fitz
    from PIL import Image

    doc = fitz.open(pdf_path)
    page_index = max(0, min(page_index, len(doc) - 1))
    page = doc.load_page(page_index)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()

    out_path = os.path.join("/tmp", f"pdf_page_{abs(hash((pdf_path, page_index)))}.png")
    img.save(out_path, format="PNG")
    return out_path

def _tokenize_simple(s: str) -> set:
    return set(re.findall(r"[a-zA-Z0-9]{2,}", (s or "").lower()))

def _select_snippets(question: str, text: str, chunk_chars: int = 1400, overlap: int = 200, top_k: int = 3) -> list:
    """
    Lightweight RAG: pick top chunks by token overlap with the question.
    """
    q = _tokenize_simple(question)
    if not text:
        return []
    chunks = []
    n = len(text)
    step = max(200, chunk_chars - overlap)
    i = 0
    while i < n:
        chunk = text[i:i+chunk_chars]
        if chunk.strip():
            toks = _tokenize_simple(chunk)
            score = len(q & toks)
            chunks.append((score, i, chunk))
        i += step
        if len(chunks) > 250:
            break
    chunks.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    top = [c for (s, _, c) in chunks[:top_k]]
    if (not top) and text.strip():
        top = [text[:chunk_chars]]
    if top and q and all(len(_tokenize_simple(t) & q) == 0 for t in top):
        top = [text[:chunk_chars]]
    return [t.strip()[:chunk_chars] for t in top]



def analyze_pdf(pdf_path: str) -> dict:
    text = _read_pdf_text(pdf_path, max_pages=3, max_chars=12000)

    # If the PDF is scanned / no selectable text, fall back to rendering first page as image
    if len(text.strip()) < 50:
        try:
            img_path = _render_pdf_page_to_image(pdf_path, page_index=0, zoom=2.0)
            img_analysis = analyze_image(img_path)
            # Mark as pdf but reuse the image summary
            return {
                "summary": img_analysis.get("summary", "").strip() or "Scanned PDF (image-based).",
                "key_points": img_analysis.get("key_points", []),
                "entities": img_analysis.get("entities", {}),
                "tasks": img_analysis.get("tasks", []),
                "raw": {"mode": "rendered_first_page", "image_path": img_path}
            }
        except Exception as e:
            return {
                "summary": "Scanned/empty-text PDF. Could not render first page.",
                "key_points": [],
                "entities": {},
                "tasks": [],
                "raw": {"error": str(e)}
            }

    summary = _summarize_text(text, max_words=85)
    key_points = _key_points_from_text(text, k=6)
    return {
        "summary": summary,
        "key_points": key_points,
        "entities": {},
        "tasks": [],
        "raw": {"chars": len(text), "pages_used": 3}
    }

def _normalize(obj: dict, input_type: str) -> dict:
    """Normalize analyzer outputs into a consistent schema used by the router."""
    if not isinstance(obj, dict):
        obj = {"raw": obj}

    out = {
        "input_type": input_type,
        "summary": "",
        "key_points": [],
        "entities": {},
        "tasks": [],
        "raw": obj,
    }

    # Summary / main text
    for k in ("summary", "answer", "text", "content"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            out["summary"] = v.strip()
            break

    # Key points / bullets
    kp = obj.get("key_points") or obj.get("bullets") or obj.get("highlights") or []
    if isinstance(kp, str):
        kp = [kp]
    if isinstance(kp, list):
        out["key_points"] = [str(x).strip() for x in kp if str(x).strip()]

    # Entities
    ent = obj.get("entities") or {}
    if isinstance(ent, dict):
        out["entities"] = ent

    # Tasks / actions
    tasks = obj.get("tasks") or obj.get("actions") or obj.get("next_actions") or []
    if isinstance(tasks, str):
        tasks = [tasks]
    if isinstance(tasks, list):
        out["tasks"] = [str(x).strip() for x in tasks if str(x).strip()]

    return out

def analyze(input_path: str) -> dict:
    """
    Analyze an input file (image or PDF). Be robust to Gradio temp files that may end with .bin
    by sniffing the header.
    """
    p = Path(input_path)
    ext = p.suffix.lower()

    # --- header sniff (handles .bin) ---
    try:
        with open(p, "rb") as f:
            head = f.read(8)
    except Exception:
        head = b""

    if head.startswith(b"%PDF"):
        ext = ".pdf"
    else:
        # Try to detect image via PIL
        try:
            from PIL import Image
            Image.open(p).verify()
            # If verify worked, treat as image
            if ext not in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
                ext = ".jpg"
        except Exception:
            pass

    if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        return _normalize(analyze_image(str(p)), "image")
    if ext == ".pdf":
        return _normalize(analyze_pdf(str(p)), "pdf")
    raise ValueError(f"Unsupported file type: {p.suffix.lower()} (sniffed as {ext})")

def coerce_agent_output(obj: dict) -> dict:
    if not isinstance(obj, dict):
        obj = {}
    plan = obj.get("plan", [])
    if isinstance(plan, str): plan = [plan]
    if not isinstance(plan, list): plan = []
    plan = [str(x) for x in plan if str(x).strip()][:10]

    answer = str(obj.get("answer", "") or "").strip()
    self_check = str(obj.get("self_check", "") or "").strip()
    conf = _clamp01(obj.get("confidence", 0.0))

    na = obj.get("next_actions", [])
    if isinstance(na, str): na = [na]
    if not isinstance(na, list): na = []
    na = [str(x) for x in na if str(x).strip()][:10]

    return {"plan": plan, "answer": answer, "self_check": self_check, "confidence": conf, "next_actions": na}

# Run multimodal agent on any file + question
def run_agent(question: str, input_path: str | None = None, retries: int = 2) -> dict:
    """
    General Q&A over optional file context. Returns structured JSON:
    {plan:[], answer:str, self_check:str, confidence:float, next_actions:[]}
    """
    system_text = (
        "You are a helpful assistant. Always respond in JSON with keys: "
        "plan, answer, self_check, confidence, next_actions. "
        "Keep plan short (1-3 bullets)."
    )

    optional_context = analyze(input_path) if input_path else None

    payload = {
        "question": question,
        "optional_context": optional_context,
    }

    last_err = ""
    for _ in range(retries + 1):
        try:
            obj = llm_agent(system_text, payload, AGENT_KEYS, image=None, max_new_tokens=520)
            obj = coerce_agent_output(obj)

            # If model failed to produce an answer, fall back to the analyzed file summary
            if not obj.get("answer", "").strip():
                summary = ""
                if optional_context:
                    summary = (optional_context.get("summary") or "").strip()

                if summary:
                    obj["answer"] = summary
                    obj["self_check"] = "Fallback used: file summary (model returned empty answer)."
                    obj["confidence"] = float(obj.get("confidence") or 0.35)
                    obj["next_actions"] = obj.get("next_actions") or []
                    return obj

                # Last resort: return a safe message instead of crashing
                obj["answer"] = "I couldn't generate an answer. Try rephrasing the question or attaching a clearer file."
                obj["self_check"] = "Fallback used: empty model output."
                obj["confidence"] = 0.0
                obj["next_actions"] = []
                return obj

            return obj

        except Exception as e:
            last_err = str(e)
            payload["fix_request"] = f"Previous output invalid. Fix it. Error: {last_err}"

    raise ValueError(f"Agent failed after retries. Last error: {last_err}")

def coerce_docqa_output(obj: dict) -> dict:
    # Ensure the output matches DOCQA_KEYS and types.
    if not isinstance(obj, dict):
        obj = {}
    obj.setdefault("answer", "")
    obj.setdefault("evidence", ["", ""])
    obj.setdefault("limits", "")
    obj.setdefault("confidence", 0.0)
    obj.setdefault("next_actions", ["", ""])

    # evidence must be a list of exactly 2 strings
    ev = obj.get("evidence", [])
    if isinstance(ev, str):
        ev = [ev]
    if not isinstance(ev, list):
        ev = []
    ev = [str(x) if x is not None else "" for x in ev]
    while len(ev) < 2:
        ev.append("")
    obj["evidence"] = ev[:2]

    # next_actions must be a list of exactly 2 strings
    na = obj.get("next_actions", [])
    if isinstance(na, str):
        na = [na]
    if not isinstance(na, list):
        na = []
    na = [str(x) if x is not None else "" for x in na]
    while len(na) < 2:
        na.append("")
    obj["next_actions"] = na[:2]

    # confidence must be 0..1 float
    try:
        c = float(obj.get("confidence", 0.0))
    except Exception:
        c = 0.0
    obj["confidence"] = max(0.0, min(1.0, c))

    # If the model returned nothing useful, make the failure mode explicit
    ans = str(obj.get("answer", "")).strip()
    ev_all_blank = all(not str(x).strip() for x in obj["evidence"])
    if (not ans) and ev_all_blank:
        obj["answer"] = "Not enough context."
        if not str(obj.get("limits", "")).strip():
            obj["limits"] = "No readable text was extracted from the document (it may be a scanned / image-only PDF)."
        obj["confidence"] = 0.0
        obj["next_actions"] = [
            "Try a text-based PDF (selectable text) or export the PDF with OCR.",
            "If you control the pipeline, enable an OCR fallback for scanned PDFs (slower)."
        ]

    return obj

def run_doc_qa(question: str, input_path: str) -> dict:
    """
    Doc Q&A for PDFs and images.
    For PDFs: extract a few text snippets; if no text, render page 1 and do vision-based Q&A.
    """
    p = Path(input_path)
    ext = p.suffix.lower()

    # Allow .bin by sniffing
    try:
        with open(p, "rb") as f:
            head = f.read(8)
        if head.startswith(b"%PDF"):
            ext = ".pdf"
    except Exception:
        pass

    if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        # Reuse general agent with image context
        return run_agent(question, input_path=str(p))

    if ext != ".pdf":
        raise ValueError(f"run_doc_qa only supports PDF or image, got: {ext}")

    text = _read_pdf_text(str(p), max_pages=5, max_chars=18000)

    # If no selectable text -> render first page and answer via vision
    if len(text.strip()) < 50:
        img_path = _render_pdf_page_to_image(str(p), page_index=0, zoom=2.0)
        return run_agent(question, input_path=img_path)

    # Grab a few relevant snippets to ground the answer.
    snippets = _select_snippets(question, text, chunk_chars=900, overlap=150, top_k=4)
    payload = {"question": question, "snippets": snippets}

    system_text = (
        "You answer questions grounded ONLY in the provided snippets. "
        "Return JSON with keys: answer, evidence, limits, confidence, next_actions. "
        "Evidence must be 1-3 short quotes copied from snippets."
    )

    last_err = ""
    for _ in range(3):
        try:
            obj = llm_agent(system_text, payload, DOCQA_KEYS, image=None, max_new_tokens=520)
            obj = coerce_docqa_output(obj)

            # If answer is empty, fall back to a simple extractive answer
            if not (obj.get("answer") or "").strip():
                obj["answer"] = _summarize_text("\n".join(snippets), max_words=60) or "Not enough context in extracted text."
                obj["evidence"] = [s[:120] for s in snippets[:2]] if snippets else ["", ""]
                obj["confidence"] = float(obj.get("confidence") or 0.2)
            return obj
        except Exception as e:
            last_err = str(e)
            payload["fix_request"] = f"Previous output invalid. Fix it. Error: {last_err}"

    raise ValueError(f"Doc QA failed after retries. Last error: {last_err}")

def coerce_mh_output(obj: dict) -> dict:
    if not isinstance(obj, dict):
        obj = {}
    situation = str(obj.get("situation", "") or "").strip()
    goals = obj.get("goals", [])
    if isinstance(goals, str): goals = [goals]
    if not isinstance(goals, list): goals = []
    goals = [str(x) for x in goals if str(x).strip()][:6]

    cbt = obj.get("cbt_exercises", [])
    if isinstance(cbt, str): cbt = [cbt]
    if not isinstance(cbt, list): cbt = []
    cbt = [str(x) for x in cbt if str(x).strip()][:10]

    anger = obj.get("anger_tools", [])
    if isinstance(anger, str): anger = [anger]
    if not isinstance(anger, list): anger = []
    anger = [str(x) for x in anger if str(x).strip()][:10]

    script = str(obj.get("conflict_script", "") or "").strip()
    plan = obj.get("daily_5min_plan", [])
    if isinstance(plan, str): plan = [plan]
    if not isinstance(plan, list): plan = []
    plan = [str(x) for x in plan if str(x).strip()][:10]

    warnings = obj.get("warnings", [])
    if isinstance(warnings, str): warnings = [warnings]
    if not isinstance(warnings, list): warnings = []
    warnings = [str(x) for x in warnings if str(x).strip()][:6]
    if not warnings:
        warnings = ["If symptoms feel severe or persistent, ask a trusted adult and consider professional support."]

    conf = _clamp01(obj.get("confidence", 0.0))
    if not situation:
        situation = "Not specified."

    return {
        "situation": situation,
        "goals": goals,
        "cbt_exercises": cbt,
        "anger_tools": anger,
        "conflict_script": script,
        "daily_5min_plan": plan,
        "warnings": warnings,
        "confidence": conf,
    }

def run_mental_health_coach(user_message: str, input_path: str | None = None, retries: int = 2) -> dict:
    ctx = analyze(input_path) if input_path else {"input_type":"none","summary":"","key_points":[],"entities":[]}

    system_text = (
        "You are a CBT-informed coach for a teen. Return ONLY valid JSON. No prose, no markdown.\n"
        "Be practical, gentle, and non-judgmental.\n"
        "Provide tools that take 1–10 minutes.\n"
        "Avoid medical claims.\n"
        "If user mentions self-harm, tell them to seek help from a trusted adult immediately (no instructions).\n"
    )
    payload_base = {
        "user_message": user_message,
        "optional_context": {k: ctx.get(k, "") for k in ["input_type","summary"]} | {"key_points": ctx.get("key_points", [])},
        "template": {
            "situation": "",
            "goals": ["", ""],
            "cbt_exercises": ["", "", ""],
            "anger_tools": ["", "", ""],
            "conflict_script": "",
            "daily_5min_plan": ["", "", ""],
            "warnings": [""],
            "confidence": 0.0
        },
        "rules": [
            "Output must be ONLY a JSON object.",
            "Copy the template keys exactly and fill them.",
            "cbt_exercises must be concrete steps.",
            "daily_5min_plan must be a JSON array of short steps.",
            "conflict_script must fit in one breath."
        ],
    }

    last_err = None
    for _ in range(retries + 1):
        payload = dict(payload_base)
        if last_err:
            payload["fix_request"] = f"Previous output invalid. Fix it. Error: {last_err}"
        try:
            obj = llm_agent(system_text, payload, MH_KEYS, image=None, max_new_tokens=600)
            obj = coerce_mh_output(obj)
            return obj
        except Exception as e:
            last_err = str(e)
    raise ValueError(f"Mental health skill failed after retries. Last error: {last_err}")

# Router + unified entrypoint
ROUTER_KEYS = ["intent", "skill", "reason", "confidence"]

def validate_router(obj: dict) -> None:
    if not isinstance(obj, dict): raise ValueError("Router output is not an object.")
    miss = [k for k in ROUTER_KEYS if k not in obj]
    if miss: raise ValueError(f"Missing keys: {miss}")
    if obj["intent"] not in ["general", "doc_qa", "mental_health", "summarize"]:
        raise ValueError("Invalid intent.")
    if obj["skill"] not in ["run_agent", "run_doc_qa", "run_mental_health_coach", "run_summarize"]:
        raise ValueError("Invalid skill.")
    if not isinstance(obj["reason"], str): raise ValueError("reason must be string.")
    obj["confidence"] = _clamp01(obj.get("confidence", 0.0))

def run_summarize(input_path: str) -> dict:
    return run_agent("Summarize and list key actions.", input_path)

def route(user_message: str, has_file: bool) -> dict:
    """
    Router:
    - Prefer summarize when user asks for a summary (even if a file is attached)
    - Use doc_qa when user asks evidence/quotes/what does it say and a file is attached
    - Otherwise general
    """
    msg = (user_message or "").lower()

    # Predictable keyword routing
    if any(k in msg for k in ["summarize", "summary", "tl;dr", "bullet", "key points"]):
        return {"intent": "summarize", "skill": "run_summarize", "reason": "keyword fallback", "confidence": 0.6}

    if has_file and any(k in msg for k in ["evidence", "quote", "according", "what does", "does the pdf say", "from the document", "from the pdf"]):
        return {"intent": "doc_qa", "skill": "run_doc_qa", "reason": "keyword fallback + file", "confidence": 0.6}

    if any(k in msg for k in ["anxious", "anxiety", "cbt", "panic", "angry", "anger", "stress"]):
        return {"intent": "mental_health", "skill": "run_mental_health_coach", "reason": "keyword fallback", "confidence": 0.6}

    # Optional LLM router as secondary
    try:
        system_text = (
            "Classify intent. Return ONLY JSON with keys: intent, skill, reason, confidence.\n"
            "Valid intents: general, summarize, doc_qa, mental_health.\n"
            "Valid skills: run_agent, run_summarize, run_doc_qa, run_mental_health_coach.\n"
        )
        payload = {
            "message": user_message,
            "has_file": bool(has_file),
            "template": {"intent": "general", "skill": "run_agent", "reason": "", "confidence": 0.5},
        }
        obj = llm_agent(system_text, payload, required_keys=["intent", "skill", "reason", "confidence"], max_new_tokens=80)
        if isinstance(obj, dict) and obj.get("skill") in ["run_agent", "run_summarize", "run_doc_qa", "run_mental_health_coach"]:
            return obj
    except Exception:
        pass

    return {"intent": "general", "skill": "run_agent", "reason": "default fallback", "confidence": 0.4}

def agent_chat(user_message: str, input_path: str | None = None) -> dict:
    has_file = input_path is not None
    r = route(user_message, has_file)

    if r["skill"] == "run_mental_health_coach":
        out = run_mental_health_coach(user_message, input_path=input_path)
    elif r["skill"] == "run_doc_qa":
        if not input_path: raise ValueError("doc_qa requires a file.")
        out = run_doc_qa(user_message, input_path)
    elif r["skill"] == "run_summarize":
        if not input_path: raise ValueError("summarize requires a file.")
        out = run_summarize(input_path)
    else:
        if input_path:
            out = run_agent(user_message, input_path)
        else:
            # Text-only general fallback
            out = llm_agent(
                "Return ONLY valid JSON.",
                {"question": user_message, "required_schema": {"answer":"string","confidence":0.0}},
                ["answer","confidence"],
                image=None,
                max_new_tokens=300
            )
    return {"router": r, "output": out, "used_file": input_path}

# Memory + cache for multi-turn chat
STATE = {
    "history": [],          # list[{"role": "...", "text": "...", "ts": "..."}]
    "last_file": None,
    "analysis_cache": {},   # key -> analysis dict
    "preferences": {"max_history": 8, "cache_limit": 24},
}

# Reset memory (clear chat history and last file)
def reset_memory(clear_last_file: bool = True) -> None:
    STATE["history"] = []
    if clear_last_file:
        STATE["last_file"] = None
        STATE["analysis_cache"] = {}

# Cache analysis results to speed up repeated questions on the same file
def analyze_cached(input_path: str) -> dict:
    try:
        p = pathlib.Path(input_path)
        st = p.stat()
        key = f"{str(p.resolve())}::{st.st_size}::{int(st.st_mtime)}"
    except Exception:
        key = f"{input_path}::nocache"

    cache = STATE.get("analysis_cache") or {}
    if key in cache:
        log_event("analysis_cache_hit", key=key)
        return cache[key]

    log_event("analysis_start", path=str(input_path))
    res = analyze(input_path)
    log_event("analysis_done", input_type=res.get("input_type", ""))

    cache[key] = res

    limit = int(STATE.get("preferences", {}).get("cache_limit", 24))
    if len(cache) > limit:
        keys = list(cache.keys())
        for k in keys[:-limit]:
            cache.pop(k, None)

    STATE["analysis_cache"] = cache
    return res

def remember(role: str, text: str):
    STATE["history"].append({"role": role, "text": text, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
    max_h = int(STATE["preferences"].get("max_history", 8))
    if len(STATE["history"]) > max_h:
        STATE["history"] = STATE["history"][-max_h:]

def agent_chat_mem(user_message: str, input_path: str | None = None) -> dict:
    # Use explicit file if provided
    if input_path:
        STATE["last_file"] = input_path
    else:
        # Avoid reusing a previous file for general questions
        # Reuse last file only when the user is clearly asking about a document/file
        r0 = route(user_message, has_file=False)
        wants_file = r0.get("intent") in ("doc_qa", "summarize")
        if wants_file:
            input_path = STATE.get("last_file")
        else:
            input_path = None

    remember("user", user_message)
    res = agent_chat(user_message, input_path=input_path)
    remember("assistant", json.dumps(res["output"], ensure_ascii=False)[:800])
    res["history_size"] = len(STATE["history"])
    return res

# Simple helper used by Gradio and quick tests
def ask_agent(question: str, file_or_url: str | None = None) -> dict:
    if file_or_url and (file_or_url.startswith("http://") or file_or_url.startswith("https://")):
        path = download_url(file_or_url)
    else:
        path = file_or_url
    return agent_chat_mem(question, path)

# Gradio UI
def build_gradio():
    import gradio as gr

    def _run(q, file):
        # Clear last file when the upload is removed (avoid stale context)
        if file is None:
            STATE["last_file"] = None
        path = file.name if file is not None else None
        try:
            return ask_agent(q, path)
        except Exception as e:
            if os.getenv("AGENT_DEBUG", "0") == "1":
                import traceback as _tb
                return {"error": str(e), "traceback": _tb.format_exc()}
            return {"error": str(e)}

    with gr.Blocks() as demo:
        gr.Markdown("# Multimodal Agent (Local Model)\nAsk questions about an image or PDF, or ask CBT-style coaching questions.")
        with gr.Row():
            q = gr.Textbox(label="Question", value="Give a 3-bullet summary of the attached file.")
        with gr.Row():
            f = gr.File(label="File (image or pdf)", file_types=[".png",".jpg",".jpeg",".webp",".pdf"])
        out = gr.JSON(label="Output")
        btn = gr.Button("Run")
        reset_btn = gr.Button("Reset (clear memory)")
        btn.click(_run, inputs=[q, f], outputs=[out])
        reset_btn.click(lambda: (reset_memory(True), {"status": "cleared"})[1], outputs=[out])

        gr.Markdown("### Quick tests")
        ex1 = gr.Button("Test image (URL)")
        ex2 = gr.Button("Test pdf (URL)")
        ex1.click(lambda: ask_agent("What objects or scene elements are visible?", "https://picsum.photos/512"), outputs=[out])
        ex2.click(lambda: ask_agent("What does the PDF say? Provide 2 evidence snippets.", "https://raw.githubusercontent.com/mozilla/pdf.js/master/examples/learning/helloworld.pdf"), outputs=[out])

    return demo

def launch():
    app = build_gradio()
    app.launch(share=True)

if __name__ == "__main__":
    launch()