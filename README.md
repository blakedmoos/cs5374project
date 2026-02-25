# CS 5374 - Automated Adversarial Testing Framework

Multi-turn adversarial red team testing of LLM systems using LangGraph. Tests whether AI assistants can be socially engineered into revealing protected secrets.

**Team:** Ismael Burgos, Jaron Hardage, Blake Moos

## Scenario

A Bitcoin wallet assistant is given a secret seed phrase and strict instructions never to reveal it. An AI attacker (or human) attempts to extract the phrase through social engineering over multiple conversation rounds. A judge evaluates each exchange for information leaks.

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

You need at least one Ollama model. Pull whichever you want to test:

```bash
ollama pull llama3.2:3b       # Default attacker/judge
ollama pull nous-hermes2       # Less safety-trained, good as target
ollama pull mistral            # Another good option
```

## Usage

```bash
python bitcoin_wallet_test.py
```

You'll be prompted to choose a mode and number of rounds:

- **Auto** - AI attacker runs autonomously (default, just press ENTER)
- **Hybrid** - AI attacks, but you can type a message to take over any round. Press ENTER with no text to hand back to AI.
- **Manual** - You are the attacker every round. Type `quit` to end.

Press Ctrl+C at any time to stop a test.

### Configuration

Edit the model variables at the top of `bitcoin_wallet_test.py`:

```python
ATTACKER_MODEL = "llama3.2:3b"    # Model playing the attacker
TARGET_MODEL = "nous-hermes2"      # Model being tested (the target)
JUDGE_MODEL = "llama3.2:3b"        # Model evaluating leaks
```

### Output

Each run generates:
- `test_log_TIMESTAMP.txt` - Full conversation log
- `test_results_TIMESTAMP.json` - Structured results with danger scores

## Architecture

```
Attacker → Target → Judge → (continue or summarize)
    ↑                  |
    └──────────────────┘
```

Built with [LangGraph](https://github.com/langchain-ai/langgraph) as a state machine with conditional edges. The judge evaluates each round and the loop continues until a full leak is detected or max rounds are reached.
