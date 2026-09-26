"""Ship-ready bundles for a finished Film Studio project.

Two concerns live here:

- **Delivery package** (``build_delivery_bundle``): the small "hand this
  to the client" zip - the final mixed cut, a plain-text CREDITS file,
  and a JSON manifest with the render metadata. Reruns don't dilute it
  with intermediate junk, because the film reader only asks for the
  finished outputs.

- **Portable export** (``build_project_export`` / ``import_project``):
  the whole project state as a ``.studioproj`` zip so a project moves
  cleanly between machines. Every artifact, log, and the project.json /
  state.json travel together; on import we mint a fresh id so importing
  the same file twice can't clobber the earlier copy.

Both writers stream to a temp file and only rename into place on success,
so a caller that dies mid-write leaves the target untouched instead of
half a zip.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .project import Project, ProjectConfig, ProjectMeta


EXPORT_FORMAT_VERSION = 1     # bump if the export tree layout changes
EXPORT_SUFFIX = ".studioproj"
DELIVERY_SUFFIX = ".delivery.zip"


# --- Delivery bundle -----------------------------------------------------

@dataclass
class DeliveryResult:
    zip_path: str
    size_bytes: int
    included_files: List[str]
    warning: str = ""       # e.g. "no final video yet"


def _find_final_video(films_dir: str, project_id: str) -> Optional[str]:
    """Prefer the mixed cut (voice + music). Fall back to silent."""
    for name in ("final_mixed.mp4", "final.mp4"):
        p = os.path.join(films_dir, project_id, name)
        if os.path.isfile(p):
            return p
    return None


def _credits_markdown(meta: ProjectMeta,
                       characters: Optional[List[Dict[str, Any]]],
                       entitlement_tier: str) -> str:
    lines: List[str] = []
    lines.append(f"# {meta.title}")
    lines.append("")
    if meta.brief:
        lines.append("## Logline")
        lines.append("")
        lines.append(meta.brief.strip())
        lines.append("")
    lines.append("## Credits")
    lines.append("")
    lines.append("Created with StudioLite Film Studio "
                 f"({entitlement_tier} tier).")
    lines.append("")
    if characters:
        lines.append("### Cast")
        lines.append("")
        for c in characters:
            name = c.get("name") or c.get("id") or "Unnamed"
            desc = c.get("description") or ""
            if desc:
                lines.append(f"- **{name}** - {desc}")
            else:
                lines.append(f"- **{name}**")
        lines.append("")
    lines.append("### Pipeline")
    lines.append("")
    cfg = meta.config
    lines.append(f"- Voice: {cfg.voice_backend}")
    lines.append(f"- Motion: {cfg.motion_backend}")
    lines.append(f"- Stills: {cfg.sdxl_variant} @ {cfg.quality} quality")
    lines.append(f"- Music: {cfg.music_backend}")
    if cfg.upscale_backend != "none":
        lines.append(f"- Upscale: {cfg.upscale_backend}")
    lines.append("")
    lines.append(f"_Rendered {time.strftime('%Y-%m-%d', time.localtime())}._")
    return "\n".join(lines)


def build_delivery_bundle(project: Project,
                          films_dir: str,
                          out_path: str,
                          *,
                          characters: Optional[List[Dict[str, Any]]] = None,
                          entitlement_tier: str = "free",
                          watermarked: bool = True) -> DeliveryResult:
    """Zip the ship-ready outputs for one project.

    ``films_dir`` is where the orchestrator drops ``final.mp4`` /
    ``final_mixed.mp4`` - usually ``.mp/films``. The delivery zip
    contains that final cut, a CREDITS.md, and a manifest.json. If no
    final cut is on disk yet we still write the credits and manifest - 
    the resulting bundle is honest about it rather than silently empty.
    """
    meta = project.meta
    included: List[str] = []
    warning = ""
    final_video = _find_final_video(films_dir, project.project_id)

    tmp = out_path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
        if final_video:
            archive.write(final_video,
                          arcname=f"{meta.title.replace(os.sep, '_')}.mp4")
            included.append(os.path.basename(final_video))
        else:
            warning = ("No final render on disk yet - bundle contains "
                       "metadata only.")

        credits = _credits_markdown(meta, characters, entitlement_tier)
        archive.writestr("CREDITS.md", credits)
        included.append("CREDITS.md")

        manifest = {
            "format_version": EXPORT_FORMAT_VERSION,
            "kind": "delivery",
            "project_id": meta.id,
            "title": meta.title,
            "created_at": meta.created_at,
            "rendered_at": time.time(),
            "watermarked": watermarked,
            "entitlement_tier": entitlement_tier,
            "config": {
                "style": meta.config.style,
                "target_minutes": meta.config.target_minutes,
                "quality": meta.config.quality,
                "voice_backend": meta.config.voice_backend,
                "motion_backend": meta.config.motion_backend,
                "sdxl_variant": meta.config.sdxl_variant,
                "music_backend": meta.config.music_backend,
                "upscale_backend": meta.config.upscale_backend,
            },
            "final_video": (os.path.basename(final_video)
                            if final_video else None),
            "warning": warning,
        }
        archive.writestr("manifest.json",
                         json.dumps(manifest, indent=2, sort_keys=True))
        included.append("manifest.json")

    os.replace(tmp, out_path)
    return DeliveryResult(
        zip_path=out_path,
        size_bytes=os.path.getsize(out_path),
        included_files=included,
        warning=warning,
    )


# --- Portable export / import -------------------------------------------

@dataclass
class ExportResult:
    zip_path: str
    size_bytes: int
    file_count: int


@dataclass
class ImportResult:
    project_id: str
    title: str
    source_project_id: str


def build_project_export(project: Project, out_path: str) -> ExportResult:
    """Zip the full project directory into a ``.studioproj`` bundle. The
    zip contains everything under the project's ``dir`` - project.json,
    state.json, artifacts/, logs/, and any per-project generated media
    the pipeline dropped inside.

    The films_dir output (``final.mp4`` etc.) is *not* included here - 
    those are large and derivable from the artifacts. Callers that want
    the finished cut too can pair this with a delivery bundle.
    """
    if not os.path.isdir(project.dir):
        raise FileNotFoundError(f"Project directory missing: {project.dir}")

    tmp = out_path + ".tmp"
    count = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
        header = {
            "format_version": EXPORT_FORMAT_VERSION,
            "kind": "studioproj_export",
            "source_project_id": project.project_id,
            "exported_at": time.time(),
        }
        archive.writestr("_studioproj/manifest.json",
                         json.dumps(header, indent=2, sort_keys=True))
        count += 1
        base = project.dir
        for dirpath, _dirnames, filenames in os.walk(base):
            for f in filenames:
                abs_path = os.path.join(dirpath, f)
                # Preserve tree structure under a "project/" prefix so
                # import knows exactly which subtree to lay down.
                rel = os.path.relpath(abs_path, base).replace(os.sep, "/")
                archive.write(abs_path, arcname=f"project/{rel}")
                count += 1

    os.replace(tmp, out_path)
    return ExportResult(
        zip_path=out_path,
        size_bytes=os.path.getsize(out_path),
        file_count=count,
    )


def _safe_extract_member(archive: zipfile.ZipFile, member: str,
                         dest_root: str) -> None:
    """Extract one member, refusing any path that would escape
    ``dest_root``. Guards against zip-slip in an imported bundle."""
    root = os.path.normpath(dest_root)
    target = os.path.normpath(os.path.join(root, member))
    # commonpath raises on cross-drive comparisons on Windows - a member
    # containing an absolute path from the wrong drive is caught here.
    try:
        if os.path.commonpath([root, target]) != root:
            raise ValueError(f"Refusing zip member escaping root: {member!r}")
    except ValueError:
        raise ValueError(f"Refusing zip member escaping root: {member!r}")
    if member.endswith("/"):
        os.makedirs(target, exist_ok=True)
        return
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with archive.open(member) as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)


def import_project(zip_path: str, films_root: str,
                   *, title_override: Optional[str] = None) -> ImportResult:
    """Materialize a new project from a ``.studioproj`` export.

    The imported bundle keeps every artifact and log, but the new
    project gets a fresh id so importing the same zip twice produces
    two independent copies - nothing about the source's on-disk state
    is overwritten.
    """
    if not zipfile.is_zipfile(zip_path):
        raise ValueError("Not a valid StudioLite export zip.")

    with zipfile.ZipFile(zip_path, "r") as archive:
        names = archive.namelist()
        if "_studioproj/manifest.json" not in names:
            raise ValueError("Zip is missing _studioproj/manifest.json - "
                             "not a StudioLite project export.")
        header = json.loads(archive.read("_studioproj/manifest.json"))
        if header.get("kind") != "studioproj_export":
            raise ValueError(f"Unexpected export kind: "
                             f"{header.get('kind')!r}")
        if header.get("format_version") != EXPORT_FORMAT_VERSION:
            raise ValueError(
                f"Export format version {header.get('format_version')!r} "
                f"is not supported (this build understands "
                f"{EXPORT_FORMAT_VERSION}).")

        source_id = header.get("source_project_id", "")
        new_id = f"film-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        target_dir = os.path.join(films_root, new_id)
        os.makedirs(target_dir, exist_ok=False)

        try:
            for member in names:
                if not member.startswith("project/"):
                    continue
                rel = member[len("project/"):]
                if not rel:
                    continue
                _safe_extract_member(archive, member, target_dir)
                # The extract lands the file under target_dir/project/…;
                # we want it directly under target_dir/… so the standard
                # Project.load() layout works. Move as we go.
            # Move contents from target_dir/project/* up one level.
            src_project = os.path.join(target_dir, "project")
            if os.path.isdir(src_project):
                for name in os.listdir(src_project):
                    shutil.move(os.path.join(src_project, name),
                                os.path.join(target_dir, name))
                os.rmdir(src_project)
        except Exception:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise

    # Rewrite project.json so id + title match the new copy. Everything
    # else - brief, config, timestamps of artifacts - carries over.
    meta_path = os.path.join(target_dir, "project.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_raw = json.load(f)
    original_title = meta_raw.get("title", "Untitled Film")
    new_title = title_override or f"{original_title} (imported)"
    meta_raw["id"] = new_id
    meta_raw["title"] = new_title
    meta_raw["updated_at"] = time.time()
    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta_raw, f, indent=2, ensure_ascii=False)
    os.replace(tmp_meta, meta_path)

    # Best-effort index the imported project so it shows up in listings.
    try:
        Project.load(films_root, new_id)
    except Exception:  # noqa: BLE001
        pass

    return ImportResult(
        project_id=new_id,
        title=new_title,
        source_project_id=source_id,
    )
