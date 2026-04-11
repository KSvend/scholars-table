---
title: The Scholar's Table
emoji: 🎓
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
license: mit
---

# The Scholar's Table

A multi-agent discussion platform where AI scholars representing distinct traditions
in International Relations and Conflict Transformation debate, analyze, and respond
to your questions.

## Scholars

- Professor Galthorn Peacegrave (Structural Peace & Conflict Transformation)
- Colonel Severus Ironhelm (Classical Realism)
- Dr. Amara Silencio (Post-Colonial / Decolonial)
- Dr. Mirabel Flickerstone (Constructivism)

## Discussion Modes

- **Private Consultation** — One-on-one dialogue with a single scholar.
- **Panel Discussion** — Multiple scholars respond to your question, then rebut
  each other. Rebuttal pairs are chosen by an LLM-as-judge that scores tensions
  between the initial responses.
- **Free Debate** — Organic multi-scholar debate guided by an orchestrator that
  selects the next speaker, nudges silent voices, and detects convergence so the
  user can decide when to wrap up.
