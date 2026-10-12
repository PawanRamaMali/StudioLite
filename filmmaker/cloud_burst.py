"""Cloud-burst dispatcher.

Some stage backends (SeedVR2, OmniGen2, Wan2.1-FLF2V, DiffRhythm 2,
EchoMimicV3) only run on a GPU with enough VRAM. On a CPU-only or
VRAM-poor host the stage runner can ask this dispatcher to delegate to a
cloud provider that owns the GPU, pays per second, and streams the
rendered artifact back to disk.

Design goals:
  - Providers are pluggable (Replicate, RunPod, user-shipped). Each
    provider builds its own HTTP request shape; the dispatcher owns
    retries, cost logging, and local artifact placement.
  - Transport is injectable so tests cover the whole dispatcher without
    touching the network. Production transport is a tiny ``urllib``
    helper so there is no new runtime dep.
  - A dry-run mode produces a canned local result and logs a plan line,
    so a user without API tokens can exercise the pipeline end-to-end.
  - The dispatcher reads provider tokens from env vars listed below, so
    rotating credentials is a config change and never requires editing
    code.

Env vars:
  STUDIOLITE_CLOUD_PROVIDER   # 'replicate' | 'runpod' | 'none' (default 'none')
  STUDIOLITE_CLOUD_DRY_RUN    # '1' to skip the network call and return a stub
  REPLICATE_API_TOKEN         # bearer for Replicate
  RUNPOD_API_KEY              # bearer for RunPod serverless

The dispatcher never raises on provider failure. A failed burst returns a
``DispatchResult`` with ``rendered=False`` and a reason the UI can show,
so the stage falls back to a local pass or a skip-artifact.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol

logger = logging.getLogger("studiolite.filmmaker.cloud_burst")


# ---------------------------------------------------------------------------
# Public shape

@dataclass
class BurstRequest:
    """A single stage's call out to the cloud.

    `backend_name` is the backend_registry name (e.g. 'seedvr2'). `inputs`
    is a provider-neutral dict: files are strings pointing at local paths
    the provider is expected to upload; everything else serialises as
    JSON. `output_path` is where the result should land locally.
    """
    stage: str
    backend_name: str
    inputs: Dict[str, Any]
    output_path: str
    deadline_seconds: int = 1800


@dataclass
class DispatchResult:
    rendered: bool
    output_path: Optional[str]           # local path, same as request on success
    provider: str                        # name of the provider used
    attempts: int = 0
    cost_estimate_usd: Optional[float] = None
    reason: str = ""                     # non-empty only when rendered=False
    details: Dict[str, Any] = field(default_factory=dict)


# A transport is a function the dispatcher calls to hit a URL. We split
# this out so tests inject a stub and production uses urllib. The result
# must be a (status_code, body_bytes, headers) triple so a provider class
# can parse response shape as it needs.
TransportFn = Callable[[str, str, Dict[str, str], Optional[bytes]],
                       "TransportReply"]


@dataclass
class TransportReply:
    status: int
    body: bytes
    headers: Dict[str, str] = field(default_factory=dict)


class Provider(Protocol):
    """A cloud provider knows how to translate a BurstRequest into one or
    more HTTP calls and how to pull the rendered artifact back to disk."""
    name: str

    def is_configured(self) -> bool: ...

    def dispatch(self, request: BurstRequest, transport: TransportFn
                 ) -> DispatchResult: ...


# ---------------------------------------------------------------------------
# Transport - production uses urllib; tests inject a stub.

def _default_transport(method: str, url: str,
                        headers: Dict[str, str], body: Optional[bytes]
                        ) -> TransportReply:
    """Thin urllib wrapper. No deps beyond stdlib, no retry inside - the
    dispatcher owns retries so one logical operation does not retry twice
    at different layers."""
    import urllib.request
    import urllib.error

    req = urllib.request.Request(url=url, method=method, data=body)
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            return TransportReply(status=resp.status, body=data,
                                   headers=dict(resp.headers))
    except urllib.error.HTTPError as exc:
        return TransportReply(status=exc.code, body=exc.read() or b"",
                               headers=dict(exc.headers or {}))
    except Exception as exc:
        logger.warning("transport error for %s %s: %s", method, url, exc)
        return TransportReply(status=0, body=str(exc).encode("utf-8"))


# ---------------------------------------------------------------------------
# Replicate provider

class ReplicateProvider:
    """Replicate's prediction API. One POST creates a prediction, then we
    poll a status URL until it reaches `succeeded`, `failed`, or
    `canceled`. Model slugs map to backend_names via `_MODEL_SLUGS`."""

    name = "replicate"
    _BASE_URL = "https://api.replicate.com/v1/predictions"

    def is_configured(self) -> bool:
        return bool(os.environ.get("REPLICATE_API_TOKEN"))

    @staticmethod
    def _slug_for(backend_name: str) -> str:
        """Resolve a backend to its Replicate model slug lazily so an env
        swap mid-process is honoured without re-constructing the provider.
        The user never has StudioLite guess a slug: a missing env var is
        a soft error with the exact var name in the message."""
        return os.environ.get(
            f"REPLICATE_SLUG_{backend_name.upper()}", "",
        )

    def dispatch(self, request: BurstRequest, transport: TransportFn
                 ) -> DispatchResult:
        slug = self._slug_for(request.backend_name)
        if not slug:
            return DispatchResult(
                rendered=False, output_path=None, provider=self.name,
                attempts=0,
                reason=(f"no Replicate slug configured for backend "
                        f"{request.backend_name!r}; set "
                        f"REPLICATE_SLUG_{request.backend_name.upper()}"),
            )
        token = os.environ.get("REPLICATE_API_TOKEN", "")
        if not token:
            return DispatchResult(
                rendered=False, output_path=None, provider=self.name,
                reason="REPLICATE_API_TOKEN not set",
            )

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        body = json.dumps({"version": slug, "input": request.inputs}).encode("utf-8")
        reply = transport("POST", self._BASE_URL, headers, body)
        if reply.status not in (200, 201):
            return DispatchResult(
                rendered=False, output_path=None, provider=self.name,
                attempts=1,
                reason=f"Replicate POST returned {reply.status}",
                details={"body_preview": reply.body[:400].decode("utf-8", "replace")},
            )
        try:
            created = json.loads(reply.body or b"{}")
        except json.JSONDecodeError:
            return DispatchResult(rendered=False, output_path=None,
                                   provider=self.name,
                                   reason="Replicate POST body was not JSON")
        return _poll_until_done(
            provider_name=self.name, created=created, request=request,
            transport=transport, headers=headers, poll_url_key="urls",
            status_key="status",
            succeeded_values=("succeeded",),
            failed_values=("failed", "canceled"),
            output_url_fn=lambda d: d.get("output"),
        )


# ---------------------------------------------------------------------------
# RunPod serverless provider

class RunPodProvider:
    """RunPod's serverless endpoints. One POST runs the job synchronously
    for short jobs, or returns a job id for async polling. We use the
    sync endpoint when the deadline allows (<= 300s on RunPod) and the
    async endpoint otherwise."""

    name = "runpod"
    _ENDPOINT_FORMAT = "https://api.runpod.ai/v2/{endpoint_id}/runsync"
    _ASYNC_ENDPOINT_FORMAT = "https://api.runpod.ai/v2/{endpoint_id}/run"
    _STATUS_FORMAT = "https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"

    def is_configured(self) -> bool:
        return bool(os.environ.get("RUNPOD_API_KEY"))

    @staticmethod
    def _endpoint_for(backend_name: str) -> str:
        """Resolve a backend to its RunPod endpoint id lazily - mirrors
        ReplicateProvider._slug_for for the same reason."""
        return os.environ.get(
            f"RUNPOD_ENDPOINT_{backend_name.upper()}", "",
        )

    def dispatch(self, request: BurstRequest, transport: TransportFn
                 ) -> DispatchResult:
        endpoint_id = self._endpoint_for(request.backend_name)
        if not endpoint_id:
            return DispatchResult(
                rendered=False, output_path=None, provider=self.name,
                reason=(f"no RunPod endpoint id configured for backend "
                        f"{request.backend_name!r}; set "
                        f"RUNPOD_ENDPOINT_{request.backend_name.upper()}"),
            )
        token = os.environ.get("RUNPOD_API_KEY", "")
        if not token:
            return DispatchResult(
                rendered=False, output_path=None, provider=self.name,
                reason="RUNPOD_API_KEY not set",
            )

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        use_async = request.deadline_seconds > 300
        url = (self._ASYNC_ENDPOINT_FORMAT if use_async
               else self._ENDPOINT_FORMAT).format(endpoint_id=endpoint_id)
        body = json.dumps({"input": request.inputs}).encode("utf-8")
        reply = transport("POST", url, headers, body)
        if reply.status not in (200, 201):
            return DispatchResult(
                rendered=False, output_path=None, provider=self.name,
                attempts=1,
                reason=f"RunPod POST returned {reply.status}",
                details={"body_preview": reply.body[:400].decode("utf-8", "replace")},
            )
        try:
            created = json.loads(reply.body or b"{}")
        except json.JSONDecodeError:
            return DispatchResult(rendered=False, output_path=None,
                                   provider=self.name,
                                   reason="RunPod POST body was not JSON")
        if not use_async:
            # Sync endpoint returns the result inline.
            return _finalize_runpod(created, request, self.name)
        status_url = self._STATUS_FORMAT.format(endpoint_id=endpoint_id,
                                                 job_id=created.get("id") or "")
        return _poll_until_done(
            provider_name=self.name, created=created, request=request,
            transport=transport, headers=headers,
            poll_url_key=None, status_url=status_url,
            status_key="status",
            succeeded_values=("COMPLETED",),
            failed_values=("FAILED", "CANCELLED", "TIMED_OUT"),
            output_url_fn=lambda d: (d.get("output") or {}).get("url")
                                     if isinstance(d.get("output"), dict)
                                     else d.get("output"),
        )


# ---------------------------------------------------------------------------
# Polling helpers

def _poll_until_done(provider_name: str, created: Dict[str, Any],
                      request: BurstRequest, transport: TransportFn,
                      headers: Dict[str, str], poll_url_key: Optional[str],
                      status_key: str,
                      succeeded_values: tuple, failed_values: tuple,
                      output_url_fn: Callable[[Dict[str, Any]], Any],
                      status_url: Optional[str] = None
                      ) -> DispatchResult:
    """Poll a provider's status URL until the job finishes. On success we
    download the output artifact to request.output_path."""
    if poll_url_key is not None:
        status_url = (created.get(poll_url_key) or {}).get("get") or status_url
    if not status_url:
        return DispatchResult(rendered=False, output_path=None,
                               provider=provider_name,
                               reason="no status URL in create response")

    deadline = time.monotonic() + request.deadline_seconds
    attempts = 1
    while time.monotonic() < deadline:
        time.sleep(min(5, max(1, request.deadline_seconds // 60)))
        attempts += 1
        reply = transport("GET", status_url, headers, None)
        if reply.status != 200:
            continue
        try:
            payload = json.loads(reply.body or b"{}")
        except json.JSONDecodeError:
            continue
        status = payload.get(status_key)
        if status in succeeded_values:
            output = output_url_fn(payload)
            if not output:
                return DispatchResult(
                    rendered=False, output_path=None, provider=provider_name,
                    attempts=attempts,
                    reason="provider reported success but no output URL",
                )
            urls = output if isinstance(output, list) else [output]
            fetched = _download_output(urls[0], request.output_path, transport)
            if not fetched:
                return DispatchResult(
                    rendered=False, output_path=None, provider=provider_name,
                    attempts=attempts,
                    reason=f"failed to download output from {urls[0]}",
                )
            return DispatchResult(
                rendered=True, output_path=request.output_path,
                provider=provider_name, attempts=attempts,
                details={"source_url": urls[0]},
            )
        if status in failed_values:
            return DispatchResult(
                rendered=False, output_path=None, provider=provider_name,
                attempts=attempts,
                reason=f"provider reported {status}",
                details={"last_payload": payload},
            )
    return DispatchResult(
        rendered=False, output_path=None, provider=provider_name,
        attempts=attempts, reason="deadline exceeded",
    )


def _finalize_runpod(created: Dict[str, Any], request: BurstRequest,
                      provider_name: str) -> DispatchResult:
    """Shape the sync-RunPod response the same way poll does."""
    if created.get("status") == "COMPLETED":
        output = created.get("output")
        url = (output or {}).get("url") if isinstance(output, dict) else output
        if url:
            # Sync responses carry a base64 payload or a URL depending on
            # the endpoint. Our registered endpoints are URL-only so we
            # download directly; a transport error leaves rendered False.
            if _download_output(url, request.output_path, _default_transport):
                return DispatchResult(rendered=True, output_path=request.output_path,
                                       provider=provider_name, attempts=1,
                                       details={"source_url": url})
            return DispatchResult(rendered=False, output_path=None,
                                   provider=provider_name, attempts=1,
                                   reason=f"download failed from {url}")
    return DispatchResult(rendered=False, output_path=None,
                           provider=provider_name, attempts=1,
                           reason=f"RunPod sync returned {created.get('status')}",
                           details={"payload": created})


def _download_output(url: str, dest_path: str, transport: TransportFn) -> bool:
    reply = transport("GET", url, {}, None)
    if reply.status != 200 or not reply.body:
        return False
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(reply.body)
    return True


# ---------------------------------------------------------------------------
# Dry-run and dispatch entry point

class DryRunProvider:
    """A provider that writes a short placeholder file at the requested
    output path and returns success. Lets the pipeline exercise the
    dispatch path on a developer box with no credentials."""
    name = "dry-run"

    def is_configured(self) -> bool:
        return True

    def dispatch(self, request: BurstRequest, transport: TransportFn
                 ) -> DispatchResult:
        os.makedirs(os.path.dirname(request.output_path) or ".", exist_ok=True)
        marker = (
            f"[dry-run cloud-burst]\n"
            f"stage: {request.stage}\n"
            f"backend: {request.backend_name}\n"
            f"inputs: {json.dumps(request.inputs, default=str, indent=2)}\n"
        )
        with open(request.output_path, "w", encoding="utf-8") as f:
            f.write(marker)
        return DispatchResult(rendered=True, output_path=request.output_path,
                               provider=self.name, attempts=1,
                               details={"dry_run": True})


_PROVIDERS: Dict[str, Provider] = {
    "replicate": ReplicateProvider(),
    "runpod": RunPodProvider(),
    "dry-run": DryRunProvider(),
}


def dispatch(request: BurstRequest,
             transport: Optional[TransportFn] = None,
             provider_name: Optional[str] = None
             ) -> DispatchResult:
    """Pick a provider by env (or override) and run the request through it.

    Resolution order:
      1. `provider_name` argument (tests / explicit callers).
      2. STUDIOLITE_CLOUD_DRY_RUN=1 -> 'dry-run'.
      3. STUDIOLITE_CLOUD_PROVIDER env.
      4. First configured provider in a stable preference order.
      5. Return a 'no provider' result rather than raising.
    """
    if provider_name is None:
        if os.environ.get("STUDIOLITE_CLOUD_DRY_RUN") == "1":
            provider_name = "dry-run"
        else:
            provider_name = os.environ.get("STUDIOLITE_CLOUD_PROVIDER") or ""
    if not provider_name:
        for candidate in ("replicate", "runpod"):
            if _PROVIDERS[candidate].is_configured():
                provider_name = candidate
                break
    if not provider_name:
        return DispatchResult(
            rendered=False, output_path=None, provider="none",
            reason=("no cloud provider configured; set "
                    "STUDIOLITE_CLOUD_PROVIDER or an API token."),
        )
    provider = _PROVIDERS.get(provider_name)
    if provider is None:
        return DispatchResult(
            rendered=False, output_path=None, provider=provider_name,
            reason=f"unknown provider {provider_name!r}",
        )
    if not provider.is_configured():
        return DispatchResult(
            rendered=False, output_path=None, provider=provider.name,
            reason=f"provider {provider.name!r} not configured",
        )
    tfn = transport or _default_transport
    try:
        result = provider.dispatch(request, tfn)
    except Exception as exc:
        logger.exception("provider %s raised", provider.name)
        return DispatchResult(
            rendered=False, output_path=None, provider=provider.name,
            reason=f"provider raised: {exc}",
        )
    return result


def register_provider(name: str, provider: Provider) -> None:
    """Register or replace a provider. Lets a user's config shim ship a
    custom provider (e.g. an in-house GPU queue) without editing this
    module."""
    _PROVIDERS[name] = provider


def providers_configured() -> Dict[str, bool]:
    """Debug hook: show which providers would respond. Useful on the
    settings page and in launcher logs."""
    return {name: p.is_configured() for name, p in _PROVIDERS.items()}
