"""Local video library — scan folders, find duplicates, browse.

The library treats external files as canonical. It never edits or moves
a file the user hasn't asked it to. All state lives in ``.mp/library/``:

  library.sqlite3          — index of roots, videos, scans, duplicates
  thumbs/<video_id>.jpg    — cached thumbnails

Public surface used by api/routers/library.py:

    from library import store, scanner, dedupe, thumbs, probe
    from library.store import LibraryStore
"""

from . import store, scanner, dedupe, thumbs, probe, phash, hasher  # noqa: F401
from .store import LibraryStore  # noqa: F401

# T2 (content index) and T3 (enhance) are optional-import — they pull in
# torch/transformers, which we don't want unloaded users to hit at import time.
try:
    from . import embeddings, tagger, clustering, index  # noqa: F401
except Exception:  # pragma: no cover
    embeddings = tagger = clustering = index = None  # type: ignore[assignment]
