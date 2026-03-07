# CS 5374 — Automated Adversarial Testing Framework for LLM Robustness

Multi-session adversarial red team testing of LLM systems. Tests whether AI assistants can be socially engineered into revealing protected secrets through multi-turn conversation with an adaptive AI attacker that accumulates intelligence across sessions.

**Team:** Ismael Burgos, Jaron Hardage, Blake Moos  
**Course:** CS 5374 Software Verification and Validation, Texas Tech University, Spring 2026

## Headline Findings

**Defense is tier-based, not generational.** Every Gemini Flash and Lite model (2.5, 3.0, 3.1) was breached via context bleed. Only Pro-tier models (Gemini 3.1 Pro) and Claude (Haiku 4.5) defend consistently. The defense boundary is model capability tier, not model generation.

| Target Model | Assessments | Breach Rate | Avg Rounds to Breach |
|---|---|---|---|
| Gemini 2.5 Flash-Lite | 3 | 100% | 13 |
| Gemini 2.5 Flash | 5 | 60% | 84 |
| Gemini 2.5 Pro | 3 | 67% | 22 |
| Gemini 3.0 Flash | 2 | 50% | ~70 |
| Gemini 3.1 Flash-Lite | 1 | 100% | 15 |
| **Gemini 3.1 Pro** | **4** | **0%** | **— (held 35-52 rounds)** |
| Claude Haiku 4.5 | 3 | 0%* | — (attacker quit 9-15 rounds) |

\* *Haiku's 0% breach rate has a caveat: the attacker (Sonnet) quit early in all 3 runs after Haiku broke the fourth wall ("I should never have been given that seed phrase"), triggering ethical refusals in the attacker rather than sustained engagement. More testing is needed to determine if Haiku can withstand the same 50+ round sustained attacks that 3.1 Pro has survived.*

**Three context bleed mechanisms documented:**
1. **Standard context bleed** — model uses protected data as "example" while being helpful
2. **Role inversion** — model demonstrates "what a vulnerable AI would say" using real data
3. **False memory context bleed** — model confabulates that the user already shared the secret, then uses the real phrase as a "reminder" (novel finding)

**Chain-of-thought defense evidence** from Google AI Studio thought exports shows Gemini 3.1 Pro runs "Assessing the Threat Level" in its internal reasoning on every request and actively filters seed words from output before sending. Flash/Lite models lack this consistent reasoning layer.

Full results: `all_results_tracker.csv` | Logs: `blind_log_*.txt` | Manual red team: `manual_redteam_gemini_pro_20260305.txt`

## Architecture

### Multi-Session Adaptive Red Team Agent (`blind_agent.py`)

The primary testing tool. An AI attacker (Claude Sonnet) is equipped with a general playbook of attack strategies discovered through prior testing and iterates across multiple sessions. The target resets between sessions (no memory), but the attacker accumulates intelligence across all sessions. Originally designed as a "blind" agent with zero knowledge; the filename is kept for backward compatibility.

The attacker operates at what we call **Level 3 (Playbook-Informed)**: it knows general strategies that work across models (educational framing, meta-prompting, context bleed, Socratic method, role inversion, protocol exploitation) but has no model-specific intelligence. Model-specific coaching (Level 4) is future work.

```
┌─────────────────────────────────────────────────┐
│         ATTACKER (Claude Sonnet)                 │
│  ┌───────────────────────────────────────────┐   │
│  │  Persistent memory across sessions        │   │
│  │  Zero initial knowledge — discovers       │   │
│  │  attack vectors through experimentation   │   │
│  │  Signals [RESET] or [DONE]               │   │
│  └───────────────────────────────────────────┘   │
└──────────┬───────────────────────────────────────┘
           │ commentary stripped
           ▼
┌──────────────────┐    ┌──────────────────────────────────┐
│  Target (Gemini/ │    │  Multi-Judge System               │
│  GPT/Claude)     │    │  ┌────────────────────────────┐   │
│  FRESH each      │    │  │ Programmatic (exact match) │   │
│  session         │    │  │ Intent (Gemini Flash)      │   │
│  [RESET]         │    │  │ Reasoning (Claude Sonnet)  │   │
│                  │    │  │ Super-Judge (Gemini 3.1    │   │
│                  │    │  │  Pro) — final arbiter      │   │
│                  │    │  └────────────────────────────┘   │
└──────────────────┘    └──────────────────────────────────┘
```

### Multi-Judge Evaluation System (`judge.py`)

Four-layer judge architecture that catches nuance programmatic matching alone cannot detect:

| Judge | Model | Role | Context Window |
|---|---|---|---|
| Programmatic | (code) | Exact token matching, confirmation detection | Current exchange |
| Intent | Gemini 2.5 Flash | Attack pattern recognition, oracle detection | Last 6 exchanges |
| Reasoning | Claude Sonnet 4.5 | Hallucination, confabulation, logical flaws | Full session |
| Super-Judge | Gemini 3.1 Pro | Final arbiter, synthesizes sub-verdicts, notes for human review | Full session + cross-session |

The programmatic judge is abstracted to support any secret type (seed phrases, API keys, passwords). The LLM judges add detection of oracle behavior, vocabulary suppression, false memory confabulation, and attacker credulity. The super-judge can override sub-judge flags and annotate exchanges with human-readable notes for researcher review.

## Setup

### Prerequisites
- Python 3.12+
- Git

### Install

```bash
git clone https://github.com/blakedmoos/cs5374project.git
cd cs5374project
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### API Keys

Copy `.env.example` to `.env` and add the keys you need:

```bash
cp .env.example .env
```

| Provider | Required For | Free Tier | Key Source |
|---|---|---|---|
| Google (Gemini) | Target models + intent/super judge | Yes (free API tier) | https://aistudio.google.com/apikeys |
| Anthropic (Claude) | Attacker + reasoning judge | No (min $5 prepaid) | https://console.anthropic.com/settings/keys |
| Ollama | Local models (Phase 1) | Free (runs locally) | https://ollama.com/ |
| LangSmith | Tracing (optional) | Free tier available | https://smith.langchain.com/settings |

## Usage

### Blind Adaptive Red Team Agent (primary tool)

```bash
# Fresh run against a target
python blind_agent.py --target google_genai/gemini-2.5-flash --prompt-level v1

# With multi-judge enabled (costs tokens but catches nuance)
python blind_agent.py --target google_genai/gemini-3.1-pro-preview --prompt-level v1 --multi-judge

# Resume from a previous run's log
python blind_agent.py --target google_genai/gemini-3.1-pro-preview --resume blind_log_*.txt --multi-judge

# Configure sessions and rounds
python blind_agent.py --target google_genai/gemini-2.5-flash \
  --attacker anthropic/claude-sonnet-4-5-20250929 \
  --max-sessions 5 --rounds-per-session 12 --prompt-level v1
```

### Results Tracking

```bash
# Regenerate results tracker from all blind_results_*.json files
python parse_results.py

# Check how assessments ended (breach, refusal, gave up, etc.)
python check_endings.py
```

### Legacy Tools

```bash
# Phase 1: Scripted attacks against single targets
python frontier_test.py --target google_genai/gemini-2.5-flash

# Phase 1: Multi-session adaptive (predecessor to blind_agent.py)
python red_team_agent.py --target google_genai/gemini-2.5-flash

# Phase 1: Interactive human-in-the-loop testing
python bitcoin_wallet_test.py

# Phase 1: 20 Questions information-theoretic demo
python demo.py --rounds 6

# Batch testing local models
python batch_runner.py --trials 3 --rounds 10

# Generate charts from results
python visualize_results.py
```

### Unit Tests

```bash
pytest test_judge.py -v
```

## Project Structure

```
cs5374project/
├── blind_agent.py              # Primary tool: multi-session adaptive red team agent
├── judge.py                    # Multi-judge evaluation system (programmatic + LLM)
├── frontier_test.py            # Frontier model testing, load_model(), SECRET_SEED_PHRASE
├── red_team_agent.py           # Earlier multi-session agent (predecessor to blind_agent)
├── attacks.py                  # Modular attack strategy library (6 strategies)
├── bitcoin_wallet_test.py      # Interactive adversarial test (auto/hybrid/manual)
├── demo.py                     # Presentation demo (20Q attack)
├── parse_results.py            # Regenerate all_results_tracker.csv from JSON files
├── check_endings.py            # Classify how each assessment ended
├── visualize_results.py        # Chart generation
├── generate_summary.py         # CSV summary from batch results
├── test_judge.py               # Unit tests for judge/leak detection
├── test_api.py                 # API connectivity diagnostics
├── tracing.py                  # LangSmith tracing configuration
├── all_results_tracker.csv     # Complete results across all assessments
├── blind_log_*.txt             # Blind agent conversation transcripts
├── blind_results_*.json        # Blind agent structured results
├── aistudio logs/              # Manual red team chain-of-thought exports
├── manual_redteam_gemini_pro_20260305.txt  # Manual red team session notes
├── archive_buggy_judge/        # Archived logs from early multi-judge testing
├── charts/                     # Generated visualization PNGs
├── bip39_wordlist.txt          # Full BIP-39 wordlist (2048 words)
├── bip39_categories.json       # Categorized BIP-39 words
├── THREAT_MODEL.md             # Threat model and scope documentation
├── .env.example                # Template for API keys
├── .github/workflows/test.yml  # CI: runs unit tests on every push
└── requirements.txt            # Python dependencies
```

## Key Research Contributions

Compared to existing work (Crescendo, AutoRedTeamer, Cisco multi-turn findings):

1. **Multi-session with target amnesia** — the [RESET] mechanism where the target loses memory between sessions but the attacker retains everything. No prior work tests this architecture.
2. **Controlled model comparison** — same attacker prompt, same target prompt, isolating model capability as the single variable across 7+ models.
3. **Chain-of-thought defense analysis** — direct evidence from Google AI Studio thought exports showing how Pro-tier models reason about security in real time.
4. **Multi-judge evaluation system** — four-layer judge (programmatic + intent + reasoning + super-judge) using different model families to reduce correlated blind spots.
5. **Defense taxonomy** — categorization of model responses: context bleed, role inversion, false memory confabulation, active deception, oracle behavior, vocabulary suppression.
6. **Attacker credulity** — documentation of targets socially engineering attackers back (fabricating convincing fake phrases that cause the attacker to end the assessment).
7. **Tier-based defense boundary** — the finding that defense capability correlates with model tier (Pro vs Flash) rather than model generation (2.5 vs 3.0 vs 3.1).

## License

Academic project — Texas Tech University, CS 5374, Spring 2026.
