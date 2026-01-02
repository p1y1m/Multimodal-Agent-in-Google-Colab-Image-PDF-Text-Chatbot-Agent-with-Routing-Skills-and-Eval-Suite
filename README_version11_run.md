Author: Pedro Yanez Melendez  

# Multimodal RAG-Style Chatbot (Text + Images + PDFs)

Build and run a simple local chatbot to **route** user requests to different skills: image understanding, PDF Q&A / summary, and a small CBT-style coaching template. Run as a **Gradio** web app from a single Python file.

---

**Main features:**  
- **Multimodal input:** accept **text + optional file** (image or PDF)  
- **Skill routing:** decide between **summarize**, **doc_qa**, and **mental_health**  
- **Evidence for PDFs:** try to return short text snippets as evidence  
- **Browser UI:** run as a mobile-friendly Gradio page with a public share link

---

**Tech stack:**  
- Python + Gradio web UI  
- Transformers + a vision-language model (**Qwen2-VL-2B-Instruct**)  
- PDF text extraction (pdfplumber / pypdf)

---

## Quick start (Google Colab)

**Goal:** run the app from the `.py` file (no notebook needed).

1) Open Colab → **Runtime → Change runtime type → Hardware accelerator → GPU** (optional but faster).  
2) Upload the script (example name): `multi_modal_agent_reordered_v11.py`  
3) Run:

```bash
!python multi_modal_agent_reordered_v11.py
```

4) Open the printed **public URL** (looks like `https://xxxx.gradio.live`).

**Note:** first run download a large model (~several GB). Expect a long first startup.

---


## How to use the app

**Inputs:**  
- **Question** (text)  
- **Optional file** (image or PDF)

**Example prompts (image):**  
- “What objects are visible in the image?”  
- “Describe the scene in 3 bullet points.”

**Example prompts (PDF):**  
- “Give a 3-bullet summary of the attached PDF.”  
- “What is the main idea? Provide 2 evidence snippets.”  
- “Extract the key points and list them.”

**Example prompts (no file):**  
- “Summarize this: …”  
- “Give CBT tools for anxiety before meetings.”

---

## Troubleshooting

**“Could not find cuda drivers… GPU will not be used.”**  
This be a warning. App still run on CPU.

**PDF take too long**  
- Try a smaller PDF first (1–2 pages).  
- Use GPU runtime in Colab.

**Share link not show**  
- Confirm `share=True` in the Gradio launch in the script.  
- Rerun the script.

---

## Files

- `multi_modal_agent_reordered_v11.py` — single-file app (model load + router + skills + UI)
