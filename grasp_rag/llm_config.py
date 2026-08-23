"""
Shared model configuration for the GRASP-RAG redo (see MODELS.md for how these
were chosen/verified). Centralizing here means one edit updates generation for
every system under test (NaiveRAG, LightRAG, PathRAG, GRASP-RAG) plus the
classifier and cascade verifier, without touching the pinned vendored clones
in references/LightRAG or references/PathRAG.
"""

GENERATION_MODEL = "gpt-5.4-mini"   # generation + classifier + cascade verifier, all systems
ORACLE_LABEL_MODEL = "gpt-5.4"      # stronger tier, used only for Experiment 2 ground-truth labels


async def generation_complete(prompt, system_prompt=None, history_messages=None, **kwargs):
    from lightrag.llm.openai import openai_complete_if_cache
    return await openai_complete_if_cache(
        GENERATION_MODEL, prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        **kwargs,
    )


async def pathrag_generation_complete(prompt, system_prompt=None, history_messages=None, **kwargs):
    from PathRAG.llm import openai_complete_if_cache as pr_complete_if_cache
    return await pr_complete_if_cache(
        GENERATION_MODEL, prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        **kwargs,
    )
