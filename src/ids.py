import hashlib
import uuid


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def stable_id(prefix: str, *parts: str) -> str:
    """A content-derived id: same inputs -> same id, across processes and runs.

    Used wherever an id ends up rendered into an LLM prompt (see
    src/context_utils.py). With uuid4 ids the same CDC produces a different
    prompt string on every run, which defeats the prompt-hash cache in
    src/llm_cache.py. Callers must pass enough parts to be unique within a run
    — an enumeration index when content alone could repeat.
    """
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"
