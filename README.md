Pedro Yáñez Meléndez

# Multimodal RAG Chatbot Agent (Colab)

## Goal
- Build a Colab-first chatbot agent
- Route requests across text, image, and PDF
- Return structured JSON outputs for easy integration

## What this project do
- Analyze images (objects, scene, short summary)
- Read PDFs (extract text, answer questions with evidence)
- Answer text-only prompts (include CBT-style coaching + conflict script)
- Keep lightweight conversation memory
- Run automated evaluation tests (pass/fail + latency)

## Core skills
- Route intent (mental_health, summarize, doc_qa, general)
- Run multimodal extraction (image/PDF → text context)
- Run retrieval-style answers for PDFs (answer + evidence snippets)
- Enforce output schemas (validate keys, coerce types, avoid empty fields)

## Tech stack
- Google Colab runtime
- Hugging Face Transformers + a vision-language instruct model
- Gradio web UI (mobile responsive)
- JSON-first I/O for reliable downstream use

## How to run
- Open the notebook in Colab
- Run cells in order after a fresh runtime restart
- Launch Gradio UI and test with:
  - Upload PDF/image (optional)
  - Write question (English recommended unless Spanish routing patch is enabled)
  - Read JSON output in UI

## Example prompts
- “What objects are visible in the image?”
- “Summarize the attached PDF in 3 bullets and add 2 evidence snippets.”
- “I feel anxious before meetings. Give CBT tools and a calm conflict script.”

## Outputs
- Produce router metadata (intent, skill, reason, confidence)
- Produce skill result JSON (answer, evidence, plan, next actions)
- Save evaluation report JSON for reproducible results

## Known limits
- Depend on PDF text extraction quality
- Perform best on clear images and readable PDFs
- Require stable cell order (avoid redefining functions out of order)

## Next improvements
- Add stronger Spanish intent routing
- Add better PDF chunking and retrieval for long documents
- Add exportable HTML report for sharing results
