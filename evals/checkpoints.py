"""Disk-backed LangGraph checkpointer for the benchmark runner — no new dependency.

`run_benchmark.py` needs a case to survive the process that started it: a run is ten
cases of several minutes each, and losing the in-flight one to a Ctrl-C is expensive.
The Streamlit app gets that from `PostgresSaver` (`src/db.py`), but a benchmark should
not need a database.

`langgraph-checkpoint` already ships everything required:

- `InMemorySaver` keeps its whole state in exactly three dict attributes — `storage`,
  `writes`, `blobs` (declared on the class, built in its `__init__`).
- `PersistentDict` is a `defaultdict` subclass that pickles itself to a file on
  `sync()`, via a temp file + `shutil.move`, i.e. an atomic commit.

So a durable saver is an `InMemorySaver` whose three dicts are `PersistentDict`s, with
`sync()` called after each graph turn. Note this is done by *assigning* the attributes
rather than passing `factory=` to the constructor: the constructor calls the factory
three times with no way to vary the filename, so all three would collide on one file.

That assignment is the only place the eval harness reaches into LangGraph internals;
`tests/test_benchmark_harness.py` asserts the three attribute names still exist, so an
upgrade that renames them fails loudly instead of silently losing durability.

One thing to know when resuming: LangGraph warns once per type about
"deserializing unregistered type src.state.X from checkpoint", because reading a
checkpoint written by *another process* is the only time it has to rebuild our pydantic
models from scratch. It is permissive today and blocks in a future version. Registering
an allowlist (`JsonPlusSerializer(allowed_msgpack_modules=[...])`, passed as
`InMemorySaver(serde=...)`) flips the default the other way — everything *not* listed is
then blocked — so it is deliberately not done here. Do it when the upgrade forces it,
enumerating every BaseModel in `src/state.py`, and re-run a real resume to check nothing
got blocked.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Callable

from langgraph.checkpoint.memory import InMemorySaver, PersistentDict

# Written by PersistentDict.sync(); `storage` holds the checkpoints themselves.
FILENAMES = ("storage.pkl", "writes.pkl", "blobs.pkl")


def has_state(dirpath: Path | str) -> bool:
    """True if `dirpath` holds a checkpoint a previous process left behind."""
    path = Path(dirpath) / "storage.pkl"
    return path.exists() and path.stat().st_size > 0


def open_saver(dirpath: Path | str) -> tuple[InMemorySaver, Callable[[], None]]:
    """A checkpointer persisting to `dirpath`, plus the function that flushes it.

    Reloads whatever is already there, so calling this twice on the same directory
    resumes rather than restarts. The returned `sync` is what the caller invokes after
    every graph turn.
    """
    directory = Path(dirpath)
    directory.mkdir(parents=True, exist_ok=True)

    # Mirrors InMemorySaver.__init__'s own defaults: storage nests two levels
    # (thread -> namespace -> checkpoint), writes maps to a plain dict, blobs is flat.
    storage = PersistentDict(
        lambda: defaultdict(dict), filename=str(directory / "storage.pkl")
    )
    writes = PersistentDict(dict, filename=str(directory / "writes.pkl"))
    blobs = PersistentDict(filename=str(directory / "blobs.pkl"))

    dicts = (storage, writes, blobs)
    for d in dicts:
        # __init__ does not read the file — loading is explicit, and an empty file
        # (a crash between open and dump) must not abort the run.
        if Path(d.filename).exists():
            try:
                d.load()
            except (EOFError, OSError):
                pass

    saver = InMemorySaver()
    saver.storage = storage
    saver.writes = writes
    saver.blobs = blobs

    def sync() -> None:
        for d in dicts:
            d.sync()

    return saver, sync


def discard(dirpath: Path | str) -> None:
    """Drop a case's checkpoint once it no longer needs to be resumable."""
    shutil.rmtree(Path(dirpath), ignore_errors=True)
