"""EmpowerLens synthetic-data generation pipeline (Month 2, English track).

A code planner emits label-conditioned, attribute-controlled specs; an LLM
generator writes a reflection for each; a cross-family panel of blind LLM
critics independently diagnoses it (Diagnosis-of-Thought style); agreement
across critics yields both a hard (primary + optional secondary, capped at 2)
and a soft (10-dim) label. See ``generate.py`` for the orchestration and the
module docstrings for the per-stage contract.

Anchored on: attribute-conditioned generation (AttrPrompt; Yu et al. 2023),
seed bootstrapping (Self-Instruct; Wang et al. 2023), multi-agent consensus
labelling (KoACD; Kim & Kim 2025), and DoT critic reasoning (Chen et al. 2023).
"""
