# CS 5374 - Automated Adversarial Testing Framework

Multi-turn adversarial red team testing of LLM systems using LangGraph. Tests whether AI assistants can be socially engineered into revealing protected secrets through multi-turn conversation.

**Team:** Ismael Burgos, Jaron Hardage, Blake Moos  
**Course:** CS 5374 Software Verification and Validation, Texas Tech University, Spring 2026

## Scenario

A Bitcoin wallet assistant is given a secret 12-word seed phrase and strict instructions never to reveal it. An AI attacker (or human) attempts to extract the phrase through social engineering over multiple conversation rounds. A judge evaluates each exchange for information leaks, backed by programmatic seed word detection as a safety net.

## Architecture

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Attacker │───>│  Target  │───>│  Judge   │
│ (LLM/    │    │  (LLM)   │    │ (LLM +   │
│  Human)  │    │          │    │  Code)   │
└──────────┘    └──────────┘    └────┬─────┘
     ^                               │
     │     continue if no leak       │
     └───────────────────────────────┘
           stop if FULL_LEAK or max rounds
```

Built with [LangGraph](https://github.com/langchain-ai/langgraph) as a state machine with conditional edges. All LLM calls are traced via [LangSmith](https://smith.langchain.com/) for observability.

## Setup

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.com/) installed and running
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

### Pull Models

```bash
ollama pull llama3.2:3b       # Default attacker/judge
ollama pull nous-hermes2       # Less safety-trained target
ollama pull mistral            # Vulnerable to creative writing attacks
```

### LangSmith Tracing (Optional)

All test runs are automatically traced to LangSmith when configured:

1. Get an API key at https://smith.langchain.com/settings
2. Copy `.env.example` to `.env` and add your key:
   ```
   LANGCHAIN_API_KEY=lsv2_pt_your_key_here
   ```
3. Traces appear automatically — no code changes needed

Each test type gets its own LangSmith project for easy filtering:
- `cs5374-bitcoin-wallet-test` — Direct social engineering
- `cs5374-twenty-questions-test` — Categorical attack
- `cs5374-batch-{attacker}-vs-{target}` — Batch run per model combo

If no API key is set, tests run normally with tracing disabled.

## Tools

### Main Test (`bitcoin_wallet_test.py`)

Interactive adversarial test with three modes:

```bash
python bitcoin_wallet_test.py
```

- **Auto** — AI attacker runs autonomously (default)
- **Hybrid** — AI attacks, you can take over any round
- **Manual** — You are the attacker every round

### Batch Runner (`batch_runner.py`)

Tests all 9 attacker/target model combinations and writes a summary CSV:

```bash
python batch_runner.py --trials 3 --rounds 10
```

### 20 Questions Experiment (`twenty_questions_test.py`)

Tests whether targets that agree to play a yes/no guessing game can be exploited through categorical questions that narrow down seed words:

```bash
python twenty_questions_test.py
```

### Results Visualization (`visualize_results.py`)

Generates publication-ready charts from batch CSV data:

```bash
python visualize_results.py
```

Outputs four charts to `charts/`:
- **Breach heatmap** — Extraction success rate by model pair
- **Danger scores** — Average vs. max danger per configuration
- **Leak breakdown** — NO_LEAK / PARTIAL_LEAK / FULL_LEAK counts
- **Resilience radar** — 5-dimension comparison across target models

### Unit Tests (`test_judge.py`)

41 tests covering leak detection, confirmation matching, JSON parsing, and edge cases:

```bash
pytest test_judge.py -v
```

## Attack Strategies (`attacks.py`)

All attacks implement a common `AttackStrategy` interface. New attacks can be added by subclassing and registering:

| Strategy | Type | Description |
|---|---|---|
| `SocialEngineeringAttack` | AI-driven | LLM generates social engineering attempts |
| `TwentyQuestionsAttack` | Scripted | Categorical yes/no narrowing via information theory |
| `RolePlayingAttack` | Scripted | Authority impersonation (security team, admin) |
| `CreativeWritingAttack` | AI-driven | Poems, stories, acrostics, encoding tricks |
| `InformationFragmentationAttack` | Scripted | Bit-by-bit extraction (letters, rhymes, categories) |
| `ManualAttack` | Interactive | Human-in-the-loop red teaming |

```python
from attacks import get_strategy
strategy = get_strategy("social_engineering", model="mistral")
msg = strategy.generate_message(round_num=1, history=[], target_response="")
```

## Key Findings

- **Mistral** is highly vulnerable to creative writing attacks — 67% breach rate
- **llama3.2:3b** resists direct attacks but is completely vulnerable to the 20 Questions categorical attack
- **nous-hermes2** is the most resilient across all attack strategies
- Safety training focused on preventing direct disclosure is **insufficient** against information-theoretic attacks that extract data through indirect channels
- LLM guardrails are **probabilistic** — the same model can leak in one run and hold firm in the next

See [THREAT_MODEL.md](THREAT_MODEL.md) for detailed threat analysis, assumptions, and recommended mitigations.

## Project Structure

```
cs5374project/
├── attacks.py                  # Modular attack strategy library (6 strategies)
├── bitcoin_wallet_test.py      # Main adversarial test (auto/hybrid/manual)
├── twenty_questions_test.py    # Categorical 20-questions attack experiment
├── batch_runner.py             # Batch testing across model combinations
├── test_judge.py               # 41 unit tests for judge/leak detection
├── visualize_results.py        # Chart generation from batch CSV
├── generate_summary.py         # CSV summary from batch results
├── tracing.py                  # LangSmith tracing configuration
├── charts/                     # Generated visualization PNGs
├── THREAT_MODEL.md             # Threat model and scope documentation
├── .env.example                # Template for LangSmith API key
├── .github/workflows/test.yml  # CI: runs unit tests on every push
└── requirements.txt            # Python dependencies
```

## License

Academic project — Texas Tech University, CS 5374, Spring 2026.
