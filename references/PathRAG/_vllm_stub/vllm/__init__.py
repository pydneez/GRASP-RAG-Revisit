"""Placeholder for the real `vllm` package.

PathRAG/llm.py does `from vllm import LLM` unconditionally at module load,
as part of a local-HF-model fallback path this project never uses (we only
ever construct PathRAG with the OpenAI backend, via llm_model_func=
pathrag_generation_complete in grasp_rag/llm_config.py). The real `vllm`
package needs compiled CUDA extensions that don't build on this Windows/CPU
setup (`ModuleNotFoundError: No module named 'vllm._C_stable_libtorch'`
even after a full `pip install vllm`), so this stub exists purely to satisfy
the import line. `LLM` is never actually instantiated by any code path this
project exercises; if that ever changes, this stub will need to be replaced
with a real, working vllm install.

Put this directory's parent on sys.path *before* the real PathRAG checkout
whenever importing PathRAG, so this shadows any real (broken) `vllm`
install rather than relying on `vllm` being absent from site-packages.
"""


class LLM:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "vllm.LLM is a stub (see references/PathRAG/_vllm_stub/vllm/__init__.py) "
            "— this project never uses vllm's local-model backend."
        )
