"""Hook/beat structure templates for Story Mode.

Pure-function module: given a concept, a total duration target, and a template
id, return a list of scene stubs with titles, directorial hints, and durations
that sum to the target. The LLM script generator fills in the actual visuals
and narration from these stubs; without this, prompts are flat and pacing
feels uniform - the hook template is what makes short-form watchable.

Templates are tuned for 15-60s social shorts. Each one encodes a pacing shape:
very short hook up front (first ~2.5s is where most retention is lost),
1-2 build beats, peak, and a resolution or CTA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BeatStub:
    role: str          # "hook", "build", "peak", "cta", "setup", etc.
    title: str
    directive: str     # a directorial note fed to the script LLM
    weight: float      # fraction of total duration this beat should take


@dataclass(frozen=True)
class BeatTemplate:
    id: str
    label: str
    description: str
    beats: tuple[BeatStub, ...]
    min_duration_s: float
    max_duration_s: float


# Weights are relative; they're renormalised to the target duration.
# The hook is always first and always short - 1.5-3.5 seconds is what the
# retention data wants, no exceptions.
TEMPLATES: dict[str, BeatTemplate] = {
    "hook_build_payoff": BeatTemplate(
        id="hook_build_payoff",
        label="Hook → Build → Payoff",
        description="Classic three-act. Hook grabs, build raises stakes, payoff resolves. Default for most shorts.",
        beats=(
            BeatStub("hook", "The hook",
                     "Open on something arresting, unresolved, or surprising. No preamble, no title cards. The viewer must want to know what happens next within 1.5 seconds.",
                     weight=0.12),
            BeatStub("build", "The build",
                     "Raise stakes or deepen the question from the hook. New information, new angle, or escalating tension.",
                     weight=0.30),
            BeatStub("build", "The turn",
                     "Pivot: a complication, twist, or new piece of information that recontextualises the build.",
                     weight=0.28),
            BeatStub("peak", "The payoff",
                     "Resolve or reveal. The viewer should feel the answer to whatever the hook promised.",
                     weight=0.22),
            BeatStub("cta", "The tag",
                     "Short tag line or closing image. No hard ask unless the brief requires one.",
                     weight=0.08),
        ),
        min_duration_s=12.0,
        max_duration_s=90.0,
    ),
    "question_answer": BeatTemplate(
        id="question_answer",
        label="Question → Answer",
        description="Lead with a specific question in the hook, deliver the answer in acts 2-3. Great for educational / explainer shorts.",
        beats=(
            BeatStub("hook", "The question",
                     "Open with a single specific question the viewer wants answered. On-camera, spoken. Avoid rhetorical fluff.",
                     weight=0.15),
            BeatStub("build", "Why it matters",
                     "One beat on why the answer is worth caring about. Grounded, concrete.",
                     weight=0.25),
            BeatStub("peak", "The answer",
                     "Deliver the answer. Short declarative statement, then one supporting detail.",
                     weight=0.40),
            BeatStub("cta", "The implication",
                     "One-line implication or what to do with the answer.",
                     weight=0.20),
        ),
        min_duration_s=10.0,
        max_duration_s=60.0,
    ),
    "problem_solution": BeatTemplate(
        id="problem_solution",
        label="Problem → Solution",
        description="Name a pain point, then deliver the fix. Default for product and before/after shorts.",
        beats=(
            BeatStub("hook", "The pain",
                     "Open on the specific frustration. Show it, don't say it.",
                     weight=0.18),
            BeatStub("build", "Why it persists",
                     "Why the obvious fix doesn't work, or what people usually try first.",
                     weight=0.22),
            BeatStub("peak", "The method",
                     "The actual solution. Specific, concrete, demonstrated.",
                     weight=0.40),
            BeatStub("cta", "The result",
                     "After-state. Short.",
                     weight=0.20),
        ),
        min_duration_s=12.0,
        max_duration_s=75.0,
    ),
    "list_countdown": BeatTemplate(
        id="list_countdown",
        label="List countdown (5 → 1)",
        description="Numbered countdown. Each item is its own beat. Great for retention - 'keep watching for #1'.",
        beats=(
            BeatStub("hook", "The promise",
                     "'Five Xs, ranked worst to best.' State the category clearly in the hook.",
                     weight=0.12),
            BeatStub("build", "#5",
                     "First (lowest-ranked) item. Brief - leave the viewer wanting the higher-ranked ones.",
                     weight=0.16),
            BeatStub("build", "#4",
                     "Second item. Slight escalation in interest or quality.",
                     weight=0.16),
            BeatStub("build", "#3",
                     "Middle item. Bridge.",
                     weight=0.18),
            BeatStub("build", "#2",
                     "Second-highest. Build anticipation for #1.",
                     weight=0.18),
            BeatStub("peak", "#1",
                     "The top item. Longest, most visually interesting beat.",
                     weight=0.20),
        ),
        min_duration_s=25.0,
        max_duration_s=90.0,
    ),
    "before_after": BeatTemplate(
        id="before_after",
        label="Before → After",
        description="Two-act transformation. Minimum friction, maximum satisfaction. Great for makeovers, tutorials, demos.",
        beats=(
            BeatStub("hook", "Before",
                     "Show the starting state clearly. Do not explain - let the viewer judge.",
                     weight=0.30),
            BeatStub("build", "The process",
                     "Compressed montage of the change happening. No dead frames.",
                     weight=0.40),
            BeatStub("peak", "After",
                     "The result. Hold on it. Let the viewer feel the delta.",
                     weight=0.30),
        ),
        min_duration_s=8.0,
        max_duration_s=45.0,
    ),
    "narrative_arc": BeatTemplate(
        id="narrative_arc",
        label="Narrative arc (5 beats)",
        description="Setup → inciting incident → rising action → climax → resolution. For story-driven shorts.",
        beats=(
            BeatStub("hook", "Setup",
                     "Establish character, place, and normalcy. Keep it very short.",
                     weight=0.12),
            BeatStub("build", "Inciting incident",
                     "The thing that breaks normalcy. Should feel inevitable in retrospect.",
                     weight=0.18),
            BeatStub("build", "Rising action",
                     "Consequences multiply. Stakes rise. One new complication minimum.",
                     weight=0.28),
            BeatStub("peak", "Climax",
                     "Confrontation or decisive moment. The highest-energy beat.",
                     weight=0.26),
            BeatStub("cta", "Resolution",
                     "The new normal. What changed.",
                     weight=0.16),
        ),
        min_duration_s=18.0,
        max_duration_s=90.0,
    ),
}


def list_templates() -> list[dict]:
    """Serializable catalog for the frontend selector."""
    out = []
    for t in TEMPLATES.values():
        out.append({
            "id": t.id,
            "label": t.label,
            "description": t.description,
            "beat_count": len(t.beats),
            "min_duration_s": t.min_duration_s,
            "max_duration_s": t.max_duration_s,
        })
    return out


def resolve_template(template_id: Optional[str]) -> Optional[BeatTemplate]:
    """Return the template, or None if unknown/blank (caller then falls back to uniform scenes)."""
    if not template_id:
        return None
    return TEMPLATES.get(str(template_id).lower())


def build_scene_stubs(template_id: str, total_duration_s: float,
                      min_scene_s: float = 1.2) -> list[dict]:
    """Render a template into concrete scene stubs that sum to ``total_duration_s``.

    Each stub is a dict with ``title``, ``directive``, ``duration``, and
    ``role``. Durations are weight-proportional, then nudged so (a) no beat is
    shorter than ``min_scene_s`` (the hook floor - under this it reads as a
    glitch, not a beat) and (b) the total matches ``total_duration_s`` within
    one decimal. Returns [] when the template is unknown.
    """
    tmpl = resolve_template(template_id)
    if tmpl is None:
        return []

    total = max(1.0, float(total_duration_s))
    weights = [b.weight for b in tmpl.beats]
    w_sum = sum(weights) or 1.0
    raw = [total * (w / w_sum) for w in weights]

    # Enforce the floor on every beat. If any were below, pull from the
    # largest remaining beats proportionally so the total still matches.
    durations = [max(min_scene_s, d) for d in raw]
    diff = total - sum(durations)
    if abs(diff) > 0.05:
        # Find beats that can absorb the correction (above the floor)
        adjustable = [i for i, d in enumerate(durations) if d > min_scene_s + 0.05]
        if adjustable:
            per = diff / len(adjustable)
            for i in adjustable:
                durations[i] = max(min_scene_s, durations[i] + per)

    # Last tiny rounding pass
    durations = [round(d, 2) for d in durations]
    drift = round(total - sum(durations), 2)
    if abs(drift) > 0.01 and durations:
        durations[-1] = round(durations[-1] + drift, 2)

    stubs = []
    for beat, dur in zip(tmpl.beats, durations):
        stubs.append({
            "role": beat.role,
            "title": beat.title,
            "directive": beat.directive,
            "duration": max(min_scene_s, dur),
        })
    return stubs


def pacing_hint(template_id: str) -> str:
    """One-line directorial summary for the script LLM to respect the template."""
    tmpl = resolve_template(template_id)
    if tmpl is None:
        return ""
    return (
        f"Pacing: {tmpl.label}. {tmpl.description} "
        f"Respect the beat roles when writing visuals and narration - the hook must be arresting, "
        f"the payoff must resolve the hook's promise, no filler scenes."
    )
