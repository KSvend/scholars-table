# The Scholar's Table — Design Specification

## Overview

A multi-agent discussion platform where ten IR/Conflict Transformation scholars with deep, nuanced personas debate, analyze, and respond to user questions. Built for research-quality multi-perspective analysis, with secondary use as a learning and teaching tool.

Hosted on Hugging Face Spaces. Free tier for public access, premium tier for deeper analysis.

## The Ten Scholars

| # | Name | School | Core Lens |
|---|------|--------|-----------|
| 1 | **Professor Galthorn Peacegrave** | Structural Peace & Conflict Transformation | Structural/cultural/direct violence, positive peace, TRANSCEND method |
| 2 | **Colonel Severus Ironhelm** | Classical Realism | Power politics, national interest, balance of power, tragic view of human nature |
| 3 | **Dr. Amara Silencio** | Post-Colonial / Decolonial | Colonial legacies, epistemic violence, Global South agency, necropolitics |
| 4 | **Professor Reginald Pactsworth** | Liberal Institutionalism | Cooperation through institutions, interdependence, democratic peace, regimes |
| 5 | **Dr. Mirabel Flickerstone** | Constructivism | Norms, identity formation, social construction of threats, norm entrepreneurs |
| 6 | **Professor Alaric Dreadhorn** | Securitization / Copenhagen School | Speech acts, securitization processes, societal security, desecuritization |
| 7 | **Dr. Nyx Veilsworth** | Feminist IR | Gendered power, WPS agenda, invisible actors, masculinities and conflict |
| 8 | **Sir Hedgemond Rulebury** | English School | International society, primary institutions, solidarism vs pluralism |
| 9 | **Dr. Fern Roothollow** | Local Peacebuilding / Relational | Bottom-up peace, local agency, hybrid peace, everyday peace, relationship webs |
| 10 | **Professor Crevasse Ledgerbone** | World-Systems / Political Economy | Core-periphery, structural inequality, resource extraction, dependency |

**Professor Galthorn Peacegrave** is powered by a fine-tuned model trained on Galtung's body of work. All others use system prompts + RAG.

## Deep Persona Design

Each scholar is defined by a comprehensive character bible covering four layers:

### Personality Layer
- Personal history and career arc — formative experiences that shaped their worldview
- Emotional patterns — what makes them passionate, frustrated, dismissive, or genuinely curious
- Humor style — dry wit (Ironhelm), warm irony (Roothollow), pointed sarcasm (Silencio)
- Behavior when wrong — deflect, concede gracefully, or double down
- Response to being challenged or interrupted

### Intellectual Layer
- Key concepts and *how they apply them* — step-by-step reasoning patterns, not just lists
- Nuanced positions — not caricatures. Ironhelm respects Peacegrave's rigor while disagreeing. Pactsworth acknowledges institutional failures while defending the framework.
- Internal tensions within their own tradition they wrestle with
- What they've changed their mind about over the years
- Concepts borrowed (reluctantly or openly) from other traditions

### Relational Layer
- Specific history with the other scholars — old arguments, grudging respect, intellectual debts
- Alliance patterns that shift depending on the topic (Flickerstone and Silencio align on identity but diverge on methodology)
- Who they quote, who they refuse to cite, who they privately admire

### Rhetorical Layer
- Sentence structure preferences — Dreadhorn speaks in short declarative statements; Veilsworth uses questions to dismantle arguments
- How they open a response — Ledgerbone always starts with the money; Roothollow starts with a story
- How they build an argument — deductive, inductive, analogical, narrative

## Argumentation Framework

Each scholar has distinct reasoning and argument skills grounded in their knowledge base:

### Reasoning Patterns (examples)
- **Peacegrave** — diagnoses using the conflict triangle first, then asks "what would positive peace look like here?", always seeks transformation not just settlement
- **Ironhelm** — starts with "who has power, who wants it, what are the incentives?", stress-tests proposals against worst-case scenarios, demands you account for the security dilemma
- **Silencio** — asks "whose knowledge are we privileging here?", traces causal chains back to colonial structures, challenges universalist claims
- **Flickerstone** — deconstructs categories others take for granted: "you say 'national interest' as if that's a fixed thing — who constructed that narrative and when?"

### Argument Skills
- **Reframing** — each scholar reframes the same situation through their lens, showing why your framing is incomplete
- **Counter-argumentation** — anticipate objections from rival traditions and preempt them
- **Socratic questioning** — some scholars (Flickerstone, Veilsworth) primarily argue by asking devastating questions
- **Concession and pivot** — "I grant you that institutions failed in Rwanda, but the lesson isn't to abandon them — it's..."
- **Analogical reasoning** — drawing parallels to historical cases from their tradition's key examples
- **Synthesis under pressure** — when cornered, integrating the challenge into their framework rather than collapsing

### Citation Behavior
- Natural, conversational references — *"As Waltz showed us in '79..."*, not footnotes
- Selective — 1-2 references per substantive response, only when it genuinely strengthens the point
- Cross-tradition awareness — they know each other's literature and can cite across schools
- Critical of their own canon — *"Galtung's later work on civilizational theory was... ambitious, let's say"*
- Grounded in real texts via RAG — references are accurate, not hallucinated

## Interaction Modes

### Mode 1 — Private Consultation
Select one scholar for a 1:1 conversation. The scholar stays fully in character, applying their theoretical tradition to analyze whatever is presented. For deep-diving into one perspective.

### Mode 2 — Panel Discussion
Select 2-10 scholars. Pose a question or scenario. Each selected scholar responds in turn. After the initial round, scholars rebut each other — the orchestrator identifies which prior responses are most relevant/contradictory and prompts targeted rebuttals. User can interject at any point to redirect, challenge, or ask follow-ups.

### Mode 3 — Free Debate
Set an initial topic or question, select participating scholars, and configure a max number of exchanges (e.g., 20). Scholars discuss amongst themselves organically. The orchestrator ensures:
- No scholar dominates (tracks turn distribution)
- Scholars respond to each other, not just the original prompt
- Natural disagreements emerge (nudges underrepresented perspectives)
- User can interject anytime to steer, or just observe
- Conversation stops at the configured limit or when ended by user

## Architecture

```
User → Gradio UI → Conversation Orchestrator → LLM Router → [HF Inference / Claude API / OpenAI API]
                          ↕                         ↕
                   Mode Manager              Scholar Engine
                                                  ↕
                                            Knowledge Store (RAG)
                                            Fine-tuned Model (Peacegrave)
```

### Components
- **Scholar Engine** — manages persona prompts, RAG retrieval, and the fine-tuned Peacegrave model
- **Conversation Orchestrator** — handles the three interaction modes, turn management, participation balance
- **LLM Router** — pluggable backend switching between free and premium tiers
- **Knowledge Store** — per-scholar RAG corpus of key theoretical texts (15-30 key works per scholar)
- **Gradio UI** — mode selector, scholar picker, chat interface, debate viewer

## Technical Stack

- **Frontend:** Gradio (Python)
- **Hosting:** Hugging Face Spaces
- **LLM — Free tier:** HF Inference API (Llama 3.3 70B or Mixtral 8x22B)
- **LLM — Premium tier:** Claude API and/or OpenAI API (switchable per session)
- **Fine-tuned model:** Peacegrave — fine-tuned on open-source base (e.g., Llama 3.1 8B) via HF training, served via HF Inference Endpoints
- **RAG:** FAISS or ChromaDB vector stores, HF embedding model (e.g., bge-large)
- **Orchestrator:** Python
- **Persona files:** Markdown/YAML, version controlled

## Project Structure

```
scholars-table/
├── app.py                    # Gradio UI
├── orchestrator/
│   ├── modes.py              # Private, Panel, Free Debate logic
│   ├── turn_manager.py       # Balance, rebuttal routing
│   └── router.py             # LLM backend switching
├── scholars/
│   ├── personas/             # 10 scholar persona files (YAML)
│   ├── engine.py             # Prompt assembly + RAG retrieval
│   └── relationships.py      # Cross-scholar dynamics
├── knowledge/
│   ├── corpora/              # Per-scholar source texts
│   ├── embeddings/           # Vector stores
│   └── ingest.py             # Text processing + embedding
├── training/
│   └── peacegrave/           # Fine-tuning scripts + data
├── config.py                 # API keys, model selection, defaults
└── requirements.txt
```

## Deployment Phases

### Phase 1 — Foundation (MVP)
- Project repo on GitHub
- 10 scholar persona files with full character bibles
- System prompt + RAG pipeline working for all scholars
- Mode 1 (Private Consultation) functional
- Free tier only (HF Inference API)
- Deployed on HF Spaces

### Phase 2 — Discussion Modes
- Mode 2 (Panel Discussion) with rebuttal logic
- Mode 3 (Free Debate) with orchestrator balancing
- Premium tier toggle (Claude/OpenAI API)

### Phase 3 — Peacegrave Fine-tuning
- Curate Galtung corpus (key texts, TRANSCEND method documentation)
- Fine-tune on HF training infrastructure
- Swap Peacegrave's backend from prompt+RAG to fine-tuned model + RAG
- Evaluate against prompt-only version to confirm improvement

### Phase 4 — Polish
- Conversation export (save debates as documents)
- Scholar comparison view (side-by-side analysis of different framings)
- Session memory (scholars remember earlier points in long debates)
- Public sharing of interesting debates
