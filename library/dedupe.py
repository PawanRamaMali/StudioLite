"""Cluster duplicates.

Exact duplicates are grouped by SHA-256 in ``LibraryStore.exact_duplicate_clusters``.
Near duplicates use pHash pairs from the store, filtered by Hamming distance,
and clustered by union-find so a chain of similar-enough frames rolls up
into one group rather than several two-file pairs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .phash import hamming
from .store import LibraryStore, Video


DEFAULT_NEAR_THRESHOLD = 8   # 64-bit hash: up to 8 bit differences = "close enough"


class _DSU:
    """Tiny union-find over video ids."""
    def __init__(self):
        self.parent: Dict[int, int] = {}
    def find(self, x: int) -> int:
        while self.parent.get(x, x) != x:
            self.parent[x] = self.parent.get(self.parent[x], self.parent[x])
            x = self.parent[x]
        return x
    def union(self, a: int, b: int) -> None:
        self.parent.setdefault(a, a); self.parent.setdefault(b, b)
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.parent[ra] = rb


def near_duplicate_clusters(store: LibraryStore, *,
                            threshold: int = DEFAULT_NEAR_THRESHOLD,
                            min_group: int = 2) -> List[Dict[str, Any]]:
    """Return near-duplicate clusters. Each cluster is a group of videos
    where every pair is within `threshold` Hamming distance of at least
    one other member - union-find, not clique."""
    pairs = store.phash_candidate_pairs()
    if not pairs:
        return []
    dsu = _DSU()
    by_id: Dict[int, Video] = {}
    for a, b in pairs:
        by_id.setdefault(a.id, a); by_id.setdefault(b.id, b)
        if a.phash_hex and b.phash_hex and hamming(a.phash_hex, b.phash_hex) <= threshold:
            dsu.union(a.id, b.id)

    # Also make sure singletons that came out of the candidate pairs stay
    # accessible - but we only emit clusters with ≥ min_group members.
    groups: Dict[int, List[Video]] = {}
    for vid, v in by_id.items():
        if vid in dsu.parent:
            root = dsu.find(vid)
            groups.setdefault(root, []).append(v)

    out: List[Dict[str, Any]] = []
    for root, members in groups.items():
        if len(members) < min_group:
            continue
        # Sort members by (bigger size first, then oldest mtime) so the UI
        # can highlight the "keeper candidate" at the top.
        members.sort(key=lambda v: (-v.size_bytes, v.mtime))
        out.append({
            "kind": "near",
            "key": f"cluster-{root}",
            "members": [v.__dict__ for v in members],
        })
    # Largest clusters first - those are usually the most interesting.
    out.sort(key=lambda c: len(c["members"]), reverse=True)
    return out


def combined_clusters(store: LibraryStore,
                      near_threshold: int = DEFAULT_NEAR_THRESHOLD) -> Dict[str, Any]:
    """Return both exact and near-duplicate clusters + summary stats."""
    exact = store.exact_duplicate_clusters()
    near = near_duplicate_clusters(store, threshold=near_threshold)
    exact_saveable = sum(
        sum(m["size_bytes"] for m in c["members"][1:]) for c in exact
    )
    near_saveable = sum(
        sum(m["size_bytes"] for m in c["members"][1:]) for c in near
    )
    return {
        "exact_clusters": exact,
        "near_clusters": near,
        "near_threshold": near_threshold,
        "exact_saveable_bytes": exact_saveable,
        "near_saveable_bytes": near_saveable,
    }


# ---------------------------------------------------------------------------
# Deletion planning - "delete all but one per cluster"
# ---------------------------------------------------------------------------

_KEEPER_STRATEGIES = {"largest", "smallest", "oldest", "newest", "shortest_path"}


def _keeper_key(strategy: str):
    """Return a sort key so that the KEEPER lands at index 0 of the sorted list."""
    if strategy == "largest":
        return lambda m: (-int(m["size_bytes"] or 0), m["mtime"])
    if strategy == "smallest":
        return lambda m: (int(m["size_bytes"] or 0), m["mtime"])
    if strategy == "newest":
        return lambda m: (-float(m["mtime"] or 0),)
    if strategy == "shortest_path":
        return lambda m: (len(m["abs_path"] or ""), -int(m["size_bytes"] or 0))
    # default: oldest
    return lambda m: (float(m["mtime"] or 0),)


def build_deletion_plan(store: LibraryStore, *,
                        include_exact: bool = True,
                        include_near: bool = True,
                        near_threshold: int = DEFAULT_NEAR_THRESHOLD,
                        keeper_strategy: str = "largest",
                        cluster_keeper_overrides: Optional[Dict[str, int]] = None,
                        ) -> Dict[str, Any]:
    """Return a preview of what a bulk 'delete all but one' would remove.

- `keeper_strategy` picks a default keeper per cluster.
- `cluster_keeper_overrides` maps `cluster.key -> video_id` and wins over
      the strategy for that one cluster (the UI uses this so the user can
      hand-pick a keeper in specific clusters before confirming).

    The returned plan is stable JSON - same input, same output - so a UI
    can show it, let the user tweak, and re-fetch."""
    strategy = keeper_strategy if keeper_strategy in _KEEPER_STRATEGIES else "largest"
    overrides = cluster_keeper_overrides or {}

    clusters: List[Dict[str, Any]] = []
    if include_exact:
        clusters.extend(store.exact_duplicate_clusters())
    if include_near:
        clusters.extend(near_duplicate_clusters(store, threshold=near_threshold))

    key = _keeper_key(strategy)
    plan_clusters: List[Dict[str, Any]] = []
    total_delete_bytes = 0
    total_delete_files = 0
    all_delete_ids: List[int] = []

    for c in clusters:
        members = list(c["members"])
        if len(members) < 2:
            continue
        override_id = overrides.get(c["key"])
        if override_id is not None:
            keeper = next((m for m in members if int(m["id"]) == int(override_id)), None)
            if keeper is None:
                # override id no longer valid - fall back to strategy default
                members.sort(key=key)
                keeper = members[0]
            else:
                # Move keeper to the front
                members = [keeper] + [m for m in members if m is not keeper]
        else:
            members.sort(key=key)
            keeper = members[0]

        to_delete = members[1:]
        cluster_delete_bytes = sum(int(m["size_bytes"] or 0) for m in to_delete)
        total_delete_bytes += cluster_delete_bytes
        total_delete_files += len(to_delete)
        all_delete_ids.extend(int(m["id"]) for m in to_delete)

        plan_clusters.append({
            "kind": c["kind"],
            "key": c["key"],
            "keeper": keeper,
            "delete": to_delete,
            "delete_bytes": cluster_delete_bytes,
        })

    return {
        "strategy": strategy,
        "near_threshold": near_threshold,
        "clusters": plan_clusters,
        "total_delete_files": total_delete_files,
        "total_delete_bytes": total_delete_bytes,
        "delete_ids": all_delete_ids,
    }
