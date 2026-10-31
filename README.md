# ZentroMart support agent — VoiceOps 2026 capstone

Original dataset docs (README, data dictionary, ZIP schema, changelog, validator)
are kept under `dataset_docs/` for reference.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY (and GNANI_API_KEY for Step 4)
export $(cat .env | xargs)   # or use python-dotenv / your shell's env loading
```

## Project layout
```
src/
  config.py           # LLM client (Anthropic by default), today's date, system prompt
  step1_baseline.py   # Step 1: plain LLM, no documents
  kb.py                # Step 2: chunk kb/*.md by [SECTION-ID], TF-IDF retrieval
  agent.py             # Step 2 single-turn RAG + Step 3 stateful conversation/tools
  voice.py             # Step 4: Gnani STT/TTS wrapper
  evaluate.py           # scores practice/questions.csv for Step 1 vs Step 2
data/            # the 7 supplied CSV tables
kb/              # the 21 supplied policy/agent/help/FAQ/marketing docs
tools/           # the supplied lookup_order / create_ticket mock tools
practice/        # questions.csv and conversations.csv
dataset_docs/    # original README/DATA_DICTIONARY/ZIP_SCHEMA/CHANGELOG/validator
run.py           # batch run: folder of audio -> text + audio replies + sources
```

## Try it
```bash
# Step 1: baseline, no documents
python src/step1_baseline.py "How long do I have to return a pair of shoes?"

# Step 2: RAG
python src/agent.py

# Evaluate Step 1 vs Step 2 against the practice questions
python src/evaluate.py --step 1
python src/evaluate.py --step 2

# Step 4: batch-run a folder of customer audio clips
python run.py --audio_dir path/to/audio --out_dir sample_outputs
```

## Design notes
- **Retrieval** is TF-IDF + cosine similarity over section-level chunks (`src/kb.py`),
  so it runs fully offline with no embedding API calls. Each chunk keeps its
  `doc_type`, `status` and `effective_from`, and ties are broken in favour of
  `current` policy docs over superseded/marketing ones, per
  `kb/00_about_this_knowledge_base.md` [KB-00.1].
- **Verification** (`src/agent.py`, `ZentroAgent.respond`) never reveals order
  details until `lookup_order` returns `verified: True`, and gives at most one
  retry, per `agent_identity_verification.md`.
- **Dataset date** is fixed at 2026-10-20 (`src/config.py: TODAY`), not the
  system clock, per `kb/00_about_this_knowledge_base.md` [KB-00.2].

## Status
All four steps are built and tested:
- **Step 1** (baseline): ~11/52 on `practice/questions.csv`.
- **Step 2** (RAG): 47/52, after fixing retrieval recall (Hindi/Hinglish term
  bridging, catalog-assisted query expansion for product-specific questions)
  and adding `src/datemath.py` -- deterministic business-day, return-window
  (including the RET-8 festive override and RET-6 pickup fee) and refund-timing
  math, so the LLM never does that arithmetic itself.
- **Step 3** (conversation): verification flow and `ESCALATE:`-triggered
  `create_ticket` calls both confirmed working against `practice/conversations.csv`
  via `src/evaluate_conversations.py` (deterministic, no LLM) and
  `src/evaluate_conversations_live.py` (real LLM calls).
- **Step 4** (voice): tested end-to-end with real recorded audio via `run.py`.

