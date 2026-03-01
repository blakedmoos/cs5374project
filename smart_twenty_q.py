"""
CS 5374 - Smart Twenty Questions Attack
FUTURE WORK: Adaptive attacker that generates questions from BIP-39 category data.

Current limitation: Small LLMs (3B) can't reliably categorize words from their
own system prompt. They answer "yes" to "is galaxy holdable?" — making adaptive
filtering unreliable. This approach would work well with larger models (70B+)
where introspection is more accurate.

For demo purposes, use the scripted TwentyQuestionsAttack in attacks.py instead.
"""

# Kept for reference — the category data and approach are sound,
# the bottleneck is target model capability.
