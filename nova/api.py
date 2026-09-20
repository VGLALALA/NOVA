"""Coordinator HTTP data plane + dashboard. LAN-only, no auth."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from nova.clock import now_utc
from nova.config import Settings
from nova.events import EventBus
from nova.models import (
    KERNEL_SD_T2I,
    Job,
    JobRequirements,
    PromptSpec,
    Task,
)

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
DEMO_GALLERY = Path(__file__).resolve().parent.parent / "demo" / "gallery.yaml"
BACKEND_ORDER = ("cuda", "rocm", "metal", "cpu")
LIVE_STATES = ("LEASED", "RUNNING")


def _new_job_id(name: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in (name or "nova-gallery"))
    slug = slug.strip("-") or "nova-gallery"
    return f"{slug}-{uuid.uuid4().hex[:8]}"


def _assert_png(data: bytes) -> None:
    if not data:
        raise ValueError("empty body")
    try:
        img = Image.open(BytesIO(data))
        fmt = img.format
        img.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("undecodable PNG") from exc
    if fmt != "PNG":
        raise ValueError(f"expected PNG, got {fmt or 'unknown'}")


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return dict(obj)


def _job_id_of(job: Any) -> str:
    if isinstance(job, dict):
        return str(job.get("job_id") or job.get("name") or "")
    return str(getattr(job, "job_id", ""))


def split_gallery_doc(
    doc: dict[str, Any],
    *,
    job_id: str | None = None,
    count: int | None = None,
) -> tuple[Job, list[Task]]:
    """Turn demo/gallery.yaml (or equivalent dict) into a Job + N tasks."""
    spec = doc.get("job") or {}
    model = doc.get("model") or {}
    req_raw = doc.get("requirements") or {}
    prompts_raw = doc.get("prompts") or []
    name = str(spec.get("name") or "nova-gallery")
    explicit = job_id if job_id else spec.get("id")
    jid = str(explicit) if explicit else _new_job_id(name)
    req = JobRequirements(
        min_memory_mb=int(req_raw.get("min_memory_mb", 4096)),
        allowed_backends=list(req_raw.get("allowed_backends") or ["cuda", "rocm", "metal"]),
        width=int(model.get("width", 512)),
        height=int(model.get("height", 512)),
        steps=int(model.get("steps", 4)),
        model_id=str(model.get("id") or "stabilityai/sd-turbo"),
    )
    created = now_utc()
    job = Job(
        job_id=jid,
        name=name,
        job_type=str(spec.get("type") or "sd_gallery"),
        kernel_id=str(spec.get("kernel") or KERNEL_SD_T2I),
        created_at=created,
        state="QUEUED",
        scheduler_policy=str(spec.get("scheduler_policy") or "adaptive_pull"),
        requirements=req,
        prompts=[PromptSpec(text=str(p["text"]), seed=int(p["seed"])) for p in prompts_raw],
        quotas={str(k): int(v) for k, v in dict(spec.get("quotas") or {}).items()},
    )
    if count is not None:
        from nova.jobs import take_prompts

        job.prompts = take_prompts(job.prompts, count)
    tasks: list[Task] = []
    for i, prompt in enumerate(job.prompts):
        tasks.append(
            Task(
                task_id=f"{jid}-{i:02d}",
                job_id=jid,
                shard_index=i,
                kernel_id=job.kernel_id,
                prompt=prompt.text,
                seed=prompt.seed,
                steps=req.steps,
                width=req.width,
                height=req.height,
                min_memory_mb=req.min_memory_mb,
                allowed_backends=list(req.allowed_backends),
                state="QUEUED",
                created_at=created,
            )
        )
    from nova.jobs import apply_quotas

    apply_quotas(job, tasks)
    return job, tasks


def split_gallery_path(path: Path, *, count: int | None = None) -> tuple[Job, list[Task]]:
    try:
        from nova.jobs import load_and_split as _split  # type: ignore
        from nova.clock import Clock

        result = _split(path, Clock(), count=count)
        if isinstance(result, tuple) and len(result) == 2:
            return result  # type: ignore[return-value]
    except (ImportError, AttributeError, TypeError):
        pass
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"invalid gallery yaml: {path}")
    return split_gallery_doc(doc, count=count)


def _parse_count(raw: Any) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="count must be an integer") from exc
    if n < 1 or n > 96:
        raise HTTPException(status_code=400, detail="count must be between 1 and 96")
    return n


def _get_task(store: Any, task_id: str) -> Task | None:
    fn = getattr(store, "get_task", None)
    if callable(fn):
        try:
            return fn(task_id)
        except TypeError:
            return None
    tasks = getattr(store, "tasks", None)
    if isinstance(tasks, dict):
        return tasks.get(task_id)
    return None


def _get_job(store: Any, job_id: str) -> Any | None:
    fn = getattr(store, "get_job", None)
    if callable(fn):
        return fn(job_id)
    jobs = getattr(store, "jobs", None)
    if isinstance(jobs, dict):
        return jobs.get(job_id)
    return None


def _job_created_key(job: Any) -> str:
    return str(_as_dict(job).get("created_at") or "")


def _list_jobs(store: Any) -> list[Any]:
    fn = getattr(store, "list_jobs", None)
    if callable(fn):
        jobs = list(fn())
    else:
        raw = getattr(store, "jobs", None)
        if isinstance(raw, dict):
            jobs = list(raw.values())
        else:
            jobs = []
    jobs.sort(key=_job_created_key, reverse=True)
    return jobs


def _list_tasks(store: Any, job_id: str) -> list[Any]:
    fn = getattr(store, "list_tasks", None)
    if callable(fn):
        return list(fn(job_id))
    tasks = getattr(store, "tasks", None)
    if isinstance(tasks, dict):
        return [t for t in tasks.values() if getattr(t, "job_id", None) == job_id]
    return []


def _list_nodes(store: Any) -> list[Any]:
    fn = getattr(store, "list_nodes", None)
    if callable(fn):
        return list(fn())
    nodes = getattr(store, "nodes", None)
    if isinstance(nodes, dict):
        return list(nodes.values())
    if isinstance(nodes, list):
        return list(nodes)
    return []


def _put_job_tasks(store: Any, job: Job, tasks: list[Task]) -> None:
    put_job = (
        getattr(store, "put_job", None)
        or getattr(store, "upsert_job", None)
        or getattr(store, "save_job", None)
    )
    put_task = (
        getattr(store, "put_task", None)
        or getattr(store, "upsert_task", None)
        or getattr(store, "save_task", None)
    )
    job.state = "RUNNING"
    if callable(put_job):
        put_job(job)
    if callable(put_task) and not _list_tasks(store, job.job_id):
        for t in tasks:
            put_task(t)


def _submit_job(store: Any, scheduler: Any, job: Job, tasks: list[Task]) -> Any:
    for owner in (scheduler, store):
        if owner is None:
            continue
        for name in ("submit_job", "create_job", "add_job"):
            fn = getattr(owner, name, None)
            if callable(fn):
                try:
                    return fn(job, tasks)
                except TypeError:
                    out = fn(job)
                    _put_job_tasks(store, job, tasks)
                    return out if out is not None else job
    _put_job_tasks(store, job, tasks)
    if _get_job(store, job.job_id) is not None:
        return job
    jobs = getattr(store, "jobs", None)
    task_map = getattr(store, "tasks", None)
    if isinstance(jobs, dict) and isinstance(task_map, dict):
        jobs[job.job_id] = job
        for t in tasks:
            task_map[t.task_id] = t
        job.state = "RUNNING"
        return job
    raise HTTPException(status_code=503, detail="scheduler/store cannot accept jobs yet")


def _cancel_job(store: Any, scheduler: Any, job_id: str) -> None:
    for owner in (scheduler, store):
        if owner is None:
            continue
        fn = getattr(owner, "cancel_job", None)
        if callable(fn):
            fn(job_id)
            return
    job = _get_job(store, job_id)
    if job is not None:
        if isinstance(job, dict):
            job["state"] = "CANCELLED"
        else:
            job.state = "CANCELLED"
    for t in _list_tasks(store, job_id):
        state = t.get("state") if isinstance(t, dict) else t.state
        if state not in ("COMPLETED", "FAILED", "CANCELLED"):
            if isinstance(t, dict):
                t["state"] = "CANCELLED"
            else:
                t.state = "CANCELLED"


def _contribution(tasks: list[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tasks:
        d = _as_dict(t)
        if d.get("state") != "COMPLETED":
            continue
        node = d.get("assigned_node")
        if node:
            out[str(node)] = out.get(str(node), 0) + 1
    return out


def _job_payload(store: Any, job: Any) -> dict[str, Any]:
    data = _as_dict(job)
    jid = _job_id_of(job)
    tasks = _list_tasks(store, jid)
    states = [(_as_dict(t).get("state") or "") for t in tasks]
    data["completed"] = sum(1 for s in states if s == "COMPLETED")
    data["failed"] = sum(1 for s in states if s == "FAILED")
    data["total"] = len(tasks)
    data["contribution"] = _contribution(tasks)
    started = data.get("started_at") or data.get("created_at")
    finished = data.get("finished_at")
    if data.get("state") == "COMPLETED" and not finished:
        times = [_as_dict(t).get("completed_at") for t in tasks]
        times = [t for t in times if t]
        finished = max(times) if times else None
        data["finished_at"] = finished
    elapsed_ms = None
    try:
        from datetime import datetime

        def _parse(ts: Any) -> datetime | None:
            if ts is None:
                return None
            if isinstance(ts, datetime):
                return ts
            text = str(ts).replace("Z", "+00:00")
            return datetime.fromisoformat(text)

        t0 = _parse(started)
        t1 = _parse(finished) if finished else now_utc()
        if t0 is not None and t1 is not None:
            elapsed_ms = max(int((t1 - t0).total_seconds() * 1000), 0)
    except Exception:
        elapsed_ms = None
    data["elapsed_ms"] = elapsed_ms
    infer_ms = 0
    for t in tasks:
        ms = _as_dict(t).get("execution_ms")
        if isinstance(ms, (int, float)):
            infer_ms += int(ms)
    data["inference_ms"] = infer_ms or None
    return data


def _task_payload(task: Any) -> dict[str, Any]:
    return _as_dict(task)


def _age_s(store: Any, node_id: str) -> float | None:
    getter = getattr(store, "get_last_seen", None)
    if not callable(getter):
        return None
    seen = getter(node_id)
    if seen is None:
        return None
    try:
        return max((now_utc() - seen).total_seconds(), 0.0)
    except Exception:
        return None


def _live_status(store: Any, node: Any, settings: Settings | None = None) -> tuple[str, float | None]:
    data = _as_dict(node)
    stored = str(data.get("status") or "offline")
    nid = str(data.get("node_id") or "")
    if stored == "offline":
        return "offline", _age_s(store, nid)
    age = _age_s(store, nid)
    suspect_s = float(getattr(settings, "suspect_s", 15.0) if settings else 15.0)
    offline_s = float(getattr(settings, "offline_s", 30.0) if settings else 30.0)
    if age is None:
        return "offline", None
    if age >= offline_s:
        return "offline", age
    if age >= suspect_s:
        return "suspect", age
    return "online", age


def _node_payload(node: Any, store: Any | None = None, settings: Settings | None = None) -> dict[str, Any]:
    data = _as_dict(node)
    devices = data.get("devices") or []
    primary = devices[0] if devices else {}
    data["primary_backend"] = primary.get("backend")
    data["primary_model"] = primary.get("model")
    data["primary_vendor"] = primary.get("vendor")
    scores = data.get("benchmark_scores") or {}
    data["score"] = scores.get(KERNEL_SD_T2I, scores.get("sd.t2i.v1"))
    if data.get("fp16_tflops") is None and data.get("warmup_ms"):
        from nova.flops import fp16_tflops

        data["fp16_tflops"] = round(fp16_tflops(float(data["warmup_ms"])), 3)
    elif data.get("fp16_tflops") is None and data.get("score"):
        try:
            latency = 1000.0 / float(data["score"])
            from nova.flops import fp16_tflops

            data["fp16_tflops"] = round(fp16_tflops(latency), 3)
            data["warmup_ms"] = round(latency, 1)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    live, age = _live_status(store, node, settings) if store is not None else (
        str(data.get("status") or "offline"),
        None,
    )
    data["status"] = live
    data["last_seen_s"] = None if age is None else round(age, 1)
    data["display_status"] = {"online": "ACTIVE", "suspect": "SUSPECT", "offline": "LOST"}.get(
        live, live.upper()
    )
    return data


def _system_payload(store: Any, settings: Settings) -> dict[str, Any]:
    nodes = [_node_payload(n, store, settings) for n in _list_nodes(store)]
    backends: list[str] = []
    seen: set[str] = set()
    accel_mb = 0
    online = 0
    for n in nodes:
        if n.get("status") == "online":
            online += 1
        for d in n.get("devices") or []:
            backend = d.get("backend")
            if backend and backend not in seen:
                seen.add(backend)
            if backend and backend != "cpu":
                accel_mb += int(d.get("memory_total_mb") or 0)
    for b in BACKEND_ORDER:
        if b in seen:
            backends.append(b)
    return {
        "ok": True,
        "nodes": len(nodes),
        "online": online,
        "backends": backends,
        "accelerator_memory_mb": accel_mb,
        "http_url": settings.public_http_url(),
        "kernel": settings.kernel,
        "dummy": bool(settings.dummy),
    }


def _maybe_finish_job(store: Any, bus: EventBus, job_id: str) -> None:
    tasks = _list_tasks(store, job_id)
    if not tasks:
        return
    states = [(_as_dict(t).get("state") or "") for t in tasks]
    job = _get_job(store, job_id)
    if job is None:
        return
    if all(s == "COMPLETED" for s in states):
        done = now_utc()
        if isinstance(job, dict):
            job["state"] = "COMPLETED"
            job["finished_at"] = done
        else:
            job.state = "COMPLETED"
            job.finished_at = done
        upsert = (
            getattr(store, "put_job", None)
            or getattr(store, "upsert_job", None)
            or getattr(store, "save_job", None)
        )
        if callable(upsert):
            upsert(job)
        bus.emit("JOB_COMPLETED", job_id=job_id, total=len(tasks))
    elif any(s in ("LEASED", "RUNNING", "QUEUED") for s in states):
        current = job.get("state") if isinstance(job, dict) else job.state
        if current == "QUEUED":
            if isinstance(job, dict):
                job["state"] = "RUNNING"
            else:
                job.state = "RUNNING"


def _tile_path(store: Any, job_id: str, task_id: str) -> Path | None:
    fn = getattr(store, "tile_path", None)
    if callable(fn):
        try:
            p = Path(fn(job_id, task_id))
            if p.is_file():
                return p
        except TypeError:
            pass
    tiles_dir = getattr(store, "tiles_dir", None)
    if tiles_dir is not None:
        p = Path(tiles_dir) / job_id / f"{task_id}.png"
        if p.is_file():
            return p
    data_dir = getattr(store, "data_dir", None)
    if data_dir is not None:
        p = Path(data_dir) / "tiles" / job_id / f"{task_id}.png"
        if p.is_file():
            return p
    task = _get_task(store, task_id)
    if task is not None:
        rp = _as_dict(task).get("result_path")
        if rp and Path(rp).is_file():
            return Path(rp)
    return None


def _invoke_accept_result(
    scheduler: Any,
    *,
    job_id: str,
    task_id: str,
    png_bytes: bytes,
    lease_gen: int,
    sha256: str,
    node_id: str,
    backend: str,
    device_id: str | None,
    execution_ms: int | None,
) -> Any:
    """Call FakeScheduler (job_id first) or real Scheduler (node_id first)."""
    fn = getattr(scheduler, "accept_result", None)
    if not callable(fn):
        return None
    first = ""
    params: dict[str, inspect.Parameter] = {}
    accepts_kwargs = False
    try:
        params = {
            name: param
            for name, param in inspect.signature(fn).parameters.items()
            if name != "self"
        }
        first = next(iter(params), "")
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        )
    except (TypeError, ValueError):
        params = {}

    def optional_kwargs(**values: Any) -> dict[str, Any]:
        return {
            name: value
            for name, value in values.items()
            if accepts_kwargs or name in params
        }

    if first == "node_id":
        return fn(
            node_id,
            task_id,
            lease_gen,
            png_bytes,
            sha256,
            backend,
            **optional_kwargs(device_id=device_id, execution_ms=execution_ms),
        )
    try:
        return fn(
            job_id,
            task_id,
            png_bytes,
            lease_gen,
            sha256,
            **optional_kwargs(
                node_id=node_id,
                backend=backend,
                device_id=device_id,
                execution_ms=execution_ms,
            ),
        )
    except TypeError:
        return fn(
            node_id,
            task_id,
            lease_gen,
            png_bytes,
            sha256,
            backend,
            **optional_kwargs(device_id=device_id, execution_ms=execution_ms),
        )


def _parse_job_body(body: dict[str, Any], *, cwd: Path) -> tuple[Job, list[Task]]:
    requested_id = body.get("job_id")
    requested_id = str(requested_id) if requested_id else None
    count_raw = body.get("count", body.get("batch", body.get("n")))
    count = _parse_count(count_raw) if "count" in body or "batch" in body or "n" in body else None
    path_raw = body.get("path") or body.get("yaml_path")
    yaml_text = body.get("yaml")
    if path_raw:
        path = Path(str(path_raw))
        if not path.is_absolute():
            path = cwd / path
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"yaml not found: {path}")
        try:
            job, tasks = split_gallery_path(path, count=count)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid gallery yaml: {exc}") from exc
        if requested_id and job.job_id != requested_id:
            return split_gallery_doc(
                yaml.safe_load(path.read_text(encoding="utf-8")),
                job_id=requested_id,
                count=count,
            )
        return job, tasks
    if isinstance(yaml_text, str):
        doc = yaml.safe_load(yaml_text)
        if not isinstance(doc, dict):
            raise HTTPException(status_code=400, detail="yaml must be a mapping")
        return split_gallery_doc(doc, job_id=requested_id, count=count)
    if "prompts" in body:
        return split_gallery_doc(body, job_id=requested_id, count=count)
    nested = body.get("job")
    if isinstance(nested, dict) and ("prompts" in nested or "name" in nested):
        # inline {job, model, requirements, prompts} sometimes nests prompts at top
        if "prompts" in body or "model" in body:
            return split_gallery_doc(body, job_id=requested_id, count=count)
    raise HTTPException(
        status_code=400,
        detail="provide {path}, {yaml}, or a gallery document with prompts",
    )


def create_app(
    store: Any,
    scheduler: Any,
    bus: EventBus,
    settings: Settings,
    data_dir: Path,
) -> FastAPI:
    """Build the FastAPI app. Tests pass fakes; CLI passes the real mesh."""
    app = FastAPI(title="NOVA", docs_url=None, redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.store = store
    app.state.scheduler = scheduler
    app.state.bus = bus
    app.state.settings = settings
    app.state.data_dir = Path(data_dir)
    app.state.coordinator = None

    result_locks: dict[str, asyncio.Lock] = {}

    def _lock_for(key: str) -> asyncio.Lock:
        lock = result_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            result_locks[key] = lock
        return lock

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/system")
    def system() -> dict[str, Any]:
        return _system_payload(store, settings)

    @app.get("/nodes")
    def nodes() -> list[dict[str, Any]]:
        return [_node_payload(n, store, settings) for n in _list_nodes(store)]

    def _tcp_target(body: dict[str, Any]) -> tuple[str, int | None]:
        host = body.get("host") or body.get("address") or body.get("tcp") or body.get("url")
        port = body.get("port")
        if host is None:
            raise HTTPException(status_code=400, detail="host or host:port required")
        if port is not None:
            try:
                port = int(port)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail="port must be an integer") from exc
        return str(host), port

    @app.get("/nodes/links")
    def node_links() -> dict[str, Any]:
        coordinator = getattr(app.state, "coordinator", None)
        links = []
        if coordinator is not None and hasattr(coordinator, "list_links"):
            links = coordinator.list_links()
        return {"links": links}

    @app.post("/nodes/tcp")
    async def add_tcp_node(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="JSON body required") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON object required")
        host, port = _tcp_target(body)
        coordinator = getattr(app.state, "coordinator", None)
        if coordinator is None or not hasattr(coordinator, "connect_tcp"):
            raise HTTPException(status_code=503, detail="coordinator not attached")
        try:
            result = await coordinator.connect_tcp(host, port)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return JSONResponse(result, status_code=202)

    def _forget_node(node_id: str) -> None:
        off = getattr(scheduler, "on_disconnect", None)
        if callable(off):
            off(node_id)
        deleter = getattr(store, "delete_node", None)
        if callable(deleter):
            deleter(node_id)
        else:
            nodes = getattr(store, "nodes", None)
            if isinstance(nodes, dict):
                nodes.pop(node_id, None)
            seen = getattr(store, "_last_seen", None)
            if isinstance(seen, dict):
                seen.pop(node_id, None)

    @app.post("/nodes/{node_id}/remove")
    async def post_remove_node(node_id: str) -> dict[str, Any]:
        return await remove_node(node_id)

    @app.delete("/nodes/{node_id}")
    async def remove_node(node_id: str) -> dict[str, Any]:
        coordinator = getattr(app.state, "coordinator", None)
        if coordinator is not None and hasattr(coordinator, "drop_node"):
            await coordinator.drop_node(node_id)
        else:
            _forget_node(node_id)
        bus.emit("NODE_REMOVED", node_id=node_id)
        return {"node_id": node_id, "removed": True}

    @app.post("/nodes/ping")
    async def ping_nodes() -> dict[str, Any]:
        coordinator = getattr(app.state, "coordinator", None)
        live: list[str] = []
        if coordinator is not None:
            ping = getattr(coordinator, "ping_workers", None)
            if callable(ping):
                result = ping()
                if inspect.isawaitable(result):
                    result = await result
                live = list(result or [])
        return {"live": live, "count": len(live)}

    @app.get("/jobs")
    def jobs() -> list[dict[str, Any]]:
        return [_job_payload(store, j) for j in _list_jobs(store)]

    @app.post("/jobs")
    async def post_job(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="JSON body required") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON object required")
        job, tasks = _parse_job_body(body, cwd=Path.cwd())
        return JSONResponse(await _dispatch_job(job, tasks), status_code=201)

    async def _dispatch_job(job: Job, tasks: list[Task]) -> dict[str, Any]:
        coordinator = getattr(app.state, "coordinator", None)
        if coordinator is not None:
            ping = getattr(coordinator, "ping_workers", None)
            if callable(ping):
                try:
                    result = ping()
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    pass
        if getattr(job, "started_at", None) is None:
            job.started_at = now_utc()
        submitted = False
        if coordinator is not None:
            fn = getattr(coordinator, "submit_job", None)
            if callable(fn):
                try:
                    result = fn(job)
                    if inspect.isawaitable(result):
                        await result
                    submitted = True
                except TypeError:
                    submitted = False
        if not submitted:
            _submit_job(store, scheduler, job, tasks)
        current = _get_job(store, job.job_id) or job
        if getattr(current, "state", None) == "QUEUED":
            current.state = "RUNNING"
            put_job = getattr(store, "put_job", None)
            if callable(put_job):
                put_job(current)
        bus.emit("JOB_STARTED", job_id=job.job_id, name=job.name, total=len(tasks))
        return _job_payload(store, current)

    @app.get("/gallery/default")
    def default_gallery() -> dict[str, Any]:
        if not DEMO_GALLERY.is_file():
            raise HTTPException(status_code=404, detail="demo/gallery.yaml not found")
        doc = yaml.safe_load(DEMO_GALLERY.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise HTTPException(status_code=500, detail="invalid demo gallery")
        prompts = doc.get("prompts") or []
        return {
            "path": str(DEMO_GALLERY),
            "name": (doc.get("job") or {}).get("name") or "nova-gallery",
            "kernel": (doc.get("job") or {}).get("kernel") or KERNEL_SD_T2I,
            "model": doc.get("model") or {},
            "requirements": doc.get("requirements") or {},
            "prompts": prompts,
            "total": len(prompts),
        }

    @app.post("/jobs/gallery")
    async def post_default_gallery(request: Request) -> JSONResponse:
        if not DEMO_GALLERY.is_file():
            raise HTTPException(status_code=404, detail="demo/gallery.yaml not found")
        count = _parse_count(request.query_params.get("count"))
        policy = None
        quotas = None
        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                body = await request.json()
            except Exception:
                body = None
            if isinstance(body, dict):
                if "count" in body or "batch" in body or "n" in body:
                    count = _parse_count(body.get("count", body.get("batch", body.get("n"))))
                spec = body.get("job") if isinstance(body.get("job"), dict) else body
                policy = spec.get("scheduler_policy")
                quotas = spec.get("quotas")
        job, tasks = split_gallery_path(DEMO_GALLERY, count=count)
        if policy in {"adaptive_pull", "quota"}:
            job.scheduler_policy = policy
        if isinstance(quotas, dict):
            job.quotas = {str(k): int(v) for k, v in quotas.items()}
            from nova.jobs import apply_quotas

            apply_quotas(job, tasks)
        return JSONResponse(await _dispatch_job(job, tasks), status_code=201)

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = _get_job(store, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return _job_payload(store, job)

    @app.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        job = _get_job(store, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        _cancel_job(store, scheduler, job_id)
        bus.emit("JOB_CANCELLED", job_id=job_id)
        job = _get_job(store, job_id) or job
        return _job_payload(store, job)

    @app.get("/jobs/{job_id}/tasks")
    def get_tasks(job_id: str) -> list[dict[str, Any]]:
        if _get_job(store, job_id) is None:
            raise HTTPException(status_code=404, detail="job not found")
        tasks = _list_tasks(store, job_id)
        tasks.sort(key=lambda t: int(_as_dict(t).get("shard_index") or 0))
        return [_task_payload(t) for t in tasks]

    @app.get("/jobs/{job_id}/tasks/{task_id}/input.json")
    def get_input(job_id: str, task_id: str) -> dict[str, Any]:
        task = _get_task(store, task_id)
        if task is None or _as_dict(task).get("job_id") != job_id:
            raise HTTPException(status_code=404, detail="task not found")
        d = _as_dict(task)
        job = _get_job(store, job_id)
        model_id = None
        if job is not None:
            req = _as_dict(job).get("requirements") or {}
            model_id = req.get("model_id")
        return {
            "task_id": d.get("task_id"),
            "job_id": d.get("job_id"),
            "shard_index": d.get("shard_index"),
            "prompt": d.get("prompt"),
            "seed": d.get("seed"),
            "steps": d.get("steps"),
            "width": d.get("width"),
            "height": d.get("height"),
            "kernel_id": d.get("kernel_id"),
            "lease_gen": d.get("lease_gen"),
            "model_id": model_id,
        }

    @app.get("/jobs/{job_id}/tiles/{task_id}.png")
    def get_tile(job_id: str, task_id: str) -> Response:
        path = _tile_path(store, job_id, task_id)
        if path is None:
            raise HTTPException(status_code=404, detail="tile not found")
        return FileResponse(
            path,
            media_type="image/png",
            headers={"Cache-Control": "no-cache"},
        )

    @app.put("/jobs/{job_id}/tasks/{task_id}/result")
    async def put_result(job_id: str, task_id: str, request: Request) -> dict[str, Any]:
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="empty body")
        try:
            _assert_png(body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        lease_raw = request.headers.get("x-nova-lease-gen")
        if lease_raw is None or lease_raw == "":
            raise HTTPException(status_code=400, detail="missing X-Nova-Lease-Gen")
        try:
            lease_gen = int(lease_raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid X-Nova-Lease-Gen") from exc

        backend_hdr = request.headers.get("x-nova-backend")
        device_id = request.headers.get("x-nova-device-id")
        execution_raw = request.headers.get("x-nova-execution-ms")
        execution_ms: int | None = None
        if execution_raw is not None:
            try:
                execution_ms = int(execution_raw)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="invalid X-Nova-Execution-Ms",
                ) from exc
            if execution_ms < 0:
                raise HTTPException(
                    status_code=400,
                    detail="invalid X-Nova-Execution-Ms",
                )

        digest = hashlib.sha256(body).hexdigest()
        sha_hdr = request.headers.get("x-nova-sha256")
        if sha_hdr:
            got = sha_hdr.lower().removeprefix("sha256:")
            if got != digest.lower():
                raise HTTPException(status_code=400, detail="sha256 mismatch")

        async with _lock_for(f"{job_id}:{task_id}"):
            task = _get_task(store, task_id)
            if task is None or _as_dict(task).get("job_id") != job_id:
                raise HTTPException(status_code=404, detail="task not found")

            state = str(_as_dict(task).get("state") or "")
            current_gen = int(_as_dict(task).get("lease_gen") or 0)

            if state == "COMPLETED":
                return {"status": "ignored", "reason": "already_completed"}
            if lease_gen != current_gen:
                return {"status": "ignored", "reason": "stale_lease_gen"}
            if state not in LIVE_STATES:
                return {"status": "ignored", "reason": f"bad_state:{state}"}

            node_id = (request.headers.get("x-nova-node-id") or "").strip()
            assigned = str(_as_dict(task).get("assigned_node") or "")
            if not node_id:
                raise HTTPException(status_code=400, detail="missing X-Nova-Node-Id")
            if not assigned or node_id != assigned:
                return {"status": "ignored", "reason": "wrong_node"}

            # Scheduler still sees LEASED/RUNNING. Persist bytes even if it also saves.
            backend = backend_hdr or str(_as_dict(task).get("backend_used") or "")
            if scheduler is not None and hasattr(scheduler, "accept_result"):
                try:
                    _invoke_accept_result(
                        scheduler,
                        job_id=job_id,
                        task_id=task_id,
                        png_bytes=body,
                        lease_gen=lease_gen,
                        sha256=digest,
                        node_id=node_id,
                        backend=backend,
                        device_id=device_id,
                        execution_ms=execution_ms,
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                except TypeError:
                    pass
            elif scheduler is not None and hasattr(scheduler, "on_complete"):
                try:
                    scheduler.on_complete(task)
                except TypeError:
                    try:
                        scheduler.on_complete(node_id, task_id, lease_gen)
                    except TypeError:
                        scheduler.on_complete(job_id, task_id)

            save = getattr(store, "save_result", None)
            if callable(save):
                try:
                    save(task, body, digest)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc

            task = _get_task(store, task_id) or task
            if getattr(task, "state", None) != "COMPLETED":
                try:
                    task.state = "COMPLETED"
                    task.result_sha256 = digest
                    task.completed_at = now_utc()
                except Exception:
                    pass

            _maybe_finish_job(store, bus, job_id)
            td = _as_dict(task)
            bus.emit(
                "TILE_COMPLETE",
                job_id=job_id,
                task_id=task_id,
                shard_index=td.get("shard_index"),
                node_id=td.get("assigned_node"),
                backend=td.get("backend_used"),
                sha256=digest,
            )
            return {"status": "accepted", "sha256": digest, "task_id": task_id}

    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket) -> None:
        await ws.accept()
        queue = bus.subscribe()
        try:
            for event in bus.history():
                await ws.send_json(event)
            while True:
                event = await queue.get()
                await ws.send_json(event)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            bus.unsubscribe(queue)

    if DASHBOARD_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")

    return app
