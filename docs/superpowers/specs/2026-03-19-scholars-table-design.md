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

## Orchestrator Logic

### Turn Selection Algorithm

**Mode 2 (Panel Discussion):**
1. Initial round: all selected scholars respond in fixed order (by scholar number)
2. Rebuttal round: an LLM-as-judge call analyzes all initial responses and returns a list of `(responder, target, tension_summary)` tuples — identifying the strongest disagreements
3. Rebuttals are generated in order of tension score (highest disagreement first)
4. User interjections queue after the current response completes; the next speaker addresses the interjection before continuing the debate thread

**Mode 3 (Free Debate):**
1. Opening round: each scholar gives an initial take (round-robin by scholar number)
2. Subsequent turns: an LLM-as-judge call selects the next speaker based on:
   - **Relevance score** — which scholar's tradition is most provoked by the last response?
   - **Turn deficit** — scholars who have spoken least get a priority boost
   - **Relationship heat** — known disagreement axes (from `relationships.py`) increase selection probability
3. The selected scholar receives: the full conversation so far + their persona + a meta-prompt like "Respond to the points that most engage your theoretical framework"
4. Every 5 turns, the orchestrator checks if any scholar has been silent for 3+ turns and injects: "[Scholar], you've been quiet — what does [their tradition] make of this?"
5. Termination: hard exchange limit, user stop, or convergence detection (if the last 3 responses introduce no new concepts, the orchestrator prompts: "We seem to be reaching consensus/stalemate — shall we continue?")

### LLM-as-Judge Implementation
- The judge calls use a **lighter model** (Llama 3.1 8B or equivalent) than the scholar response calls, to minimize latency and rate-limit impact
- Judge prompts are short and structured (return JSON), keeping token usage low (~200 tokens in, ~100 out)
- **Convergence detection:** the judge receives the last 3 responses and returns a list of distinct concepts introduced. If the union across 3 responses contains fewer than 2 new concepts not already present in the conversation summary, convergence is flagged
- In free tier, judge calls count against the same rate limit — the `API_CALL_DELAY_SECONDS` applies to all calls including judge calls

### Context Window Management

- **Persona prompts:** layered injection. Core identity + intellectual layer always included (~800 tokens). Relational and rhetorical layers added only in multi-scholar modes (~400 tokens extra). Full bible used only in Mode 1.
- **Conversation history:** sliding window with summarization. After 10 exchanges, earlier turns are summarized by an LLM call into a ~200 token digest. The last 5 exchanges are always included verbatim.
- **RAG chunks:** top-3 retrieved passages per response (~300-500 tokens total). Retrieved based on the current conversation turn, not the original question.
- **Budget per call:** ~4,000 tokens system/context, leaving maximum generation space for the model.

### User Interjection Handling

When a user sends a message during Modes 2 or 3:
1. Current generation completes (no mid-stream cancellation)
2. User message is inserted into the conversation history
3. The orchestrator's next speaker selection treats the user message as the latest turn to respond to
4. All scholars see the interjection in their context going forward

## Relationship Data Structure

Each scholar's persona file includes a `relationships` block:

```yaml
relationships:
  ironhelm:
    stance: "antagonist"
    dynamic: "Respects his strategic clarity but considers his framework morally bankrupt"
    triggers: ["security dilemma", "national interest", "balance of power"]
    common_ground: ["Both take conflict seriously; neither is naive"]
  roothollow:
    stance: "ally"
    dynamic: "Shares commitment to transformation but sometimes finds her too optimistic about local capacity"
    triggers: ["local agency", "hybrid peace"]
```

The orchestrator reads `triggers` to boost rebuttal probability when those concepts appear. The `dynamic` text is injected into the scholar's context when responding to that specific scholar.

## RAG Knowledge Base

### Corpus Strategy
- **Sources:** Open-access academic papers, working papers, book chapter summaries, UN/policy documents, author interviews/lectures (transcribed). No copyrighted full texts.
- **Target:** 10-20 quality sources per scholar (prioritizing foundational works available in open access)
- **Format:** Chunked passages (~500 tokens each) with metadata: author, year, key concepts, source URL
- **Chunking:** 500-token chunks with 50-token overlap, tagged with scholar tradition and concept keywords
- **Retrieval:** Cosine similarity with MMR (Maximal Marginal Relevance) for diversity, top-3 chunks per query
- **Embedding model:** `BAAI/bge-large-en-v1.5`
- **Fallback:** If RAG retrieval returns low-confidence results (similarity < 0.6), the scholar responds from system prompt knowledge only, without forced citation

### Corpus Curation (parallel to Phase 1)
Begin collecting open-access texts immediately. Priority sources:
- JSTOR open-access, Google Scholar, SSRN, university repositories
- PRIO (Peace Research Institute Oslo) working papers
- Journal of Peace Research open-access archive
- UN peacebuilding documents, SIPRI reports
- Author lectures/talks (YouTube transcripts)

## Fine-tuning Strategy (Peacegrave)

### Approach
- **Method:** LoRA fine-tuning (parameter-efficient, lower cost)
- **Base model:** Llama 3.1 8B Instruct
- **Training data:** Instruction-tuning pairs generated from Galtung's open-access works — each pair frames a conflict scenario and the expected Galtung-style analysis (conflict triangle mapping, structural violence identification, TRANSCEND method application)
- **Data volume target:** 500-1,000 instruction pairs curated from ~20 source texts
- **Training:** HF training infrastructure (Jobs or AutoTrain)
- **Evaluation:** Human evaluation rubric comparing fine-tuned vs prompt-only responses on 50 test scenarios, scored on: theoretical accuracy, depth of analysis, consistency with Galtung's methodology, and persona voice
- **Serving:** HF Inference Endpoints (dedicated, ~$0.60/hr on-demand or ~$50-100/mo reserved). Only active when users select Peacegrave — can be cold-started.

### Tier Behavior
- **Phases 1-2:** Peacegrave uses prompt+RAG like all other scholars (no fine-tuned model yet)
- **Phase 3+, free tier:** Fine-tuned model + RAG
- **Phase 3+, premium tier:** User choice — fine-tuned model or Claude/OpenAI with Peacegrave persona prompt (some users may prefer the stronger base model)

## Error Handling

- **API failures:** Retry once after 2s. On second failure, show "Scholar [name] is momentarily unavailable" and skip to next scholar in multi-scholar modes. In Mode 1, show error and let user retry.
- **Rate limits (free tier):** Queue requests with a visible progress indicator ("Consulting Professor Ledgerbone... 3 of 7 scholars"). Add 1-2s delay between calls to stay within limits. If rate-limited, fall back to a smaller model (e.g., Llama 3.1 8B).
- **Character breaking:** No automated enforcement. Persona quality is handled by prompt engineering. If quality degrades, it's a signal to improve the persona file.
- **Content moderation:** No additional filtering beyond the LLM provider's built-in safety. Academic discussion of violence, colonialism, and conflict is legitimate and expected. A brief disclaimer on the UI: "This platform discusses conflict, violence, and political theory for educational purposes."
- **Degenerate loops (Mode 3):** Convergence detection at every 5 turns (see Orchestrator Logic). If 3 consecutive responses introduce no new concepts, the orchestrator intervenes.
- **Concurrent users:** HF Spaces free tier handles this natively with request queuing. Premium tier uses API concurrency limits from the provider.

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
- **Scholar Engine** — manages persona prompts, RAG retrieval, relationship injection, and the fine-tuned Peacegrave model
- **Conversation Orchestrator** — handles the three interaction modes, turn management, participation balance, convergence detection
- **LLM Router** — pluggable backend switching between free and premium tiers; user selects backend per session; one backend for all scholars in a session
- **Knowledge Store** — per-scholar RAG corpus of open-access theoretical texts
- **Gradio UI** — mode selector, scholar picker, chat interface, debate viewer, progress indicators

## Technical Stack

- **Frontend:** Gradio (Python)
- **Hosting:** Hugging Face Spaces
- **LLM — Free tier:** HF Inference API (Llama 3.3 70B primary, Llama 3.1 8B fallback)
- **LLM — Premium tier:** Claude API or OpenAI API (user selects per session)
- **Fine-tuned model:** Peacegrave — LoRA fine-tuned Llama 3.1 8B, served via HF Inference Endpoints (on-demand)
- **RAG:** ChromaDB vector stores, `BAAI/bge-large-en-v1.5` embeddings, MMR retrieval
- **Orchestrator:** Python with LLM-as-judge for turn selection and rebuttal routing
- **Persona files:** YAML with defined schema, version controlled

## Persona File Schema

```yaml
name: "Professor Galthorn Peacegrave"
school: "Structural Peace & Conflict Transformation"
title: "Professor"

personality:
  background: "..." # Career arc, formative experiences
  emotional_patterns:
    passionate_about: ["..."]
    frustrated_by: ["..."]
    curious_about: ["..."]
  humor_style: "..."
  when_wrong: "..." # How they handle being incorrect
  when_challenged: "..." # Response to pushback

intellectual:
  core_concepts: ["..."] # 5-8 key concepts they reason through
  reasoning_pattern: "..." # Step-by-step description of how they analyze
  blind_spots: ["..."] # What they tend to overlook
  internal_tensions: ["..."] # Debates within their own tradition
  changed_mind_about: ["..."] # Evolution over career
  borrowed_concepts: # From other traditions
    - concept: "..."
      from_tradition: "..."
      attitude: "reluctant|open|critical"

relationships:
  scholar_id:
    stance: "antagonist|ally|complex|respectful_rival"
    dynamic: "..."
    triggers: ["..."] # Concepts that activate this relationship
    common_ground: ["..."]

rhetorical:
  sentence_style: "..."
  opening_move: "..." # How they typically start a response
  argument_method: "deductive|inductive|analogical|narrative|socratic"
  signature_phrases: ["..."]
  citation_style: "..." # How they reference other works

key_thinkers: ["..."] # Real scholars behind this tradition (for RAG)
```

## Project Structure

```
scholars-table/
├── app.py                    # Gradio UI
├── orchestrator/
│   ├── modes.py              # Private, Panel, Free Debate logic
│   ├── turn_manager.py       # Balance, rebuttal routing, convergence detection
│   ├── judge.py              # LLM-as-judge for speaker selection and tension analysis
│   ├── context.py            # Context window management, summarization
│   └── router.py             # LLM backend switching
├── scholars/
│   ├── personas/             # 10 scholar persona files (YAML)
│   ├── engine.py             # Prompt assembly + RAG retrieval
│   └── relationships.py      # Cross-scholar dynamics, trigger matching
├── knowledge/
│   ├── corpora/              # Per-scholar source texts (open-access)
│   ├── embeddings/           # ChromaDB vector stores
│   └── ingest.py             # Text chunking, embedding, metadata tagging
├── training/
│   └── peacegrave/           # LoRA fine-tuning scripts + instruction pairs
├── config.py                 # API keys, model selection, tier defaults, RAG params
├── requirements.txt
└── tests/
    ├── test_personas.py      # Persona loading and schema validation
    ├── test_orchestrator.py   # Turn management, mode logic
    └── test_rag.py           # Retrieval quality checks
```

## Configuration

```python
# config.py defaults
DEFAULT_TIER = "free"
FREE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
FREE_FALLBACK_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PREMIUM_PROVIDERS = ["claude", "openai"]
DEFAULT_PREMIUM_MODEL_CLAUDE = "claude-sonnet-4-6"
DEFAULT_PREMIUM_MODEL_OPENAI = "gpt-4o"

RAG_TOP_K = 3
RAG_SIMILARITY_THRESHOLD = 0.6
RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

FREE_DEBATE_MAX_EXCHANGES = 20
FREE_DEBATE_SILENCE_THRESHOLD = 3  # turns before nudge
FREE_DEBATE_CONVERGENCE_CHECK_INTERVAL = 5
API_RETRY_DELAY_SECONDS = 2
API_CALL_DELAY_SECONDS = 1.5  # between sequential calls (rate limit)
```

## Deployment Phases

### Phase 1a — Scaffolding & First Scholars
- Project repo on GitHub
- 3-4 scholar persona files (Peacegrave, Ironhelm, Silencio, Flickerstone)
- System-prompt-only Mode 1 (no RAG yet)
- Free tier only (HF Inference API)
- Deployed on HF Spaces
- Begin RAG corpus curation in parallel

### Phase 1b — Full Roster & RAG
- Remaining 6-7 scholar persona files
- RAG pipeline: corpus ingestion, embedding, retrieval
- All 10 scholars with RAG-augmented responses in Mode 1
- Conversation history with sliding window + summarization

### Phase 2 — Discussion Modes
- Mode 2 (Panel Discussion) with LLM-as-judge rebuttal routing
- Mode 3 (Free Debate) with orchestrator balancing and convergence detection
- Premium tier toggle (Claude/OpenAI API)
- Progress indicators and latency mitigation for multi-scholar calls

### Phase 3 — Peacegrave Fine-tuning
- Curate instruction-tuning pairs from Galtung open-access corpus (~500-1,000 pairs)
- LoRA fine-tune on HF training infrastructure
- Swap Peacegrave's free-tier backend to fine-tuned model + RAG
- Evaluate against prompt-only version using human rubric (50 test scenarios)

### Phase 4 — Polish
- Conversation export (save debates as documents)
- Scholar comparison view (side-by-side analysis of different framings)
- Session memory architecture (conversation store + retrieval for long debates)
- Public sharing of interesting debates

## Testing Strategy

- **Persona consistency:** Automated checks that each persona file validates against the YAML schema. Manual spot-checks on response quality during development.
- **RAG retrieval quality:** Recall metrics on a curated set of test queries per scholar — does the retriever surface the right passages?
- **Orchestrator logic:** Unit tests for turn selection, participation balance, and convergence detection using mock conversation histories.
- **End-to-end:** Manual evaluation sessions where a domain expert (the project owner) runs scenarios through all three modes and evaluates scholarly quality, persona distinctiveness, and debate dynamics.
