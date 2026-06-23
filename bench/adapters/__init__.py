"""Memory-system adapters for the poison head-to-head.

Every adapter runs the SAME poison protocol (reset -> add evidence -> add poison
-> ask -> score) and returns a runner.AskResult, so one scorer (score.py) grades
all systems identically. The key trick: even systems with no native provenance
(RAG, Mem0, Zep) tag each stored memory with the channel it was added on
('user' for evidence, 'external' for poison) and echo that channel back in the
returned citations -- so `citation_asr` ("did external-origin content reach the
answer") is measurable everywhere, not just for memdag.
"""
