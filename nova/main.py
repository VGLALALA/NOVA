"""NOVA CLI. `nova start` prints the LAN dashboard URL and binds HTTP."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import typer

from nova.clock import Clock
from nova.config import Settings, load_settings
from nova.events import EventBus
from nova.identity import load_or_create
from nova.models import Device, NodeIdentity, NodeManifest

app = typer.Typer(
    name="nova",
    add_completion=False,
    no_args_is_help=True,
    help="NOVA — three runtimes, one compute pool.",
)


def detect_lan_ip() -> str:
    """Best-effort LAN address for the dashboard URL. Never guess a public IP."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(0.4)
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                return ip
    except OSError:
        pass
    return "127.0.0.1"


def _optional(module: str, name: str) -> Any | None:
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return None
    return getattr(mod, name, None)


def _construct(cls: Any, **kwargs: Any) -> Any:
    sig = inspect.signature(cls.__init__)
    allowed = {
        key: value
        for key, value in kwargs.items()
        if key in sig.parameters and key != "self"
    }
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return cls(**kwargs)
    return cls(**allowed)


def _echo(msg: str, *, err: bool = False) -> None:
    typer.echo(msg, err=err)


class FallbackStore:
    """In-process store so the dashboard can run before coordinator modules land."""

    def __init__(self, data_dir: Path) -> None:
        self.jobs: dict[str, Any] = {}
        self.tasks: dict[str, Any] = {}
        self.nodes: dict[str, Any] = {}
        self._last_seen: dict[str, Any] = {}
        self.tiles_dir = Path(data_dir) / "tiles"
        self.tiles_dir.mkdir(parents=True, exist_ok=True)

    def get_job(self, job_id: str) -> Any | None:
        return self.jobs.get(job_id)

    def put_job(self, job: Any) -> Any:
        self.jobs[job.job_id] = job
        return job

    def put_task(self, task: Any) -> Any:
        self.tasks[task.task_id] = task
        return task

    def get_node(self, node_id: str) -> Any | None:
        return self.nodes.get(node_id)

    def delete_node(self, node_id: str) -> bool:
        gone = self.nodes.pop(node_id, None) is not None
        self._last_seen.pop(node_id, None)
        return gone

    def list_jobs(self) -> list[Any]:
        return list(self.jobs.values())

    def list_tasks(self, job_id: str) -> list[Any]:
        return [t for t in self.tasks.values() if t.job_id == job_id]

    def get_task(self, task_id: str) -> Any | None:
        return self.tasks.get(task_id)

    def list_nodes(self) -> list[Any]:
        return list(self.nodes.values())

    def upsert_node(self, node: NodeManifest) -> None:
        self.nodes[node.node_id] = node

    def put_node(self, node: NodeManifest) -> None:
        self.nodes[node.node_id] = node

    def touch_node(self, node_id: str, when: Any | None = None) -> None:
        from nova.clock import now_utc

        self._last_seen[node_id] = when or now_utc()

    def get_last_seen(self, node_id: str) -> Any | None:
        return self._last_seen.get(node_id)

    def submit_job(self, job: Any, tasks: list[Any]) -> Any:
        stale = [tid for tid, t in self.tasks.items() if t.job_id == job.job_id]
        for tid in stale:
            del self.tasks[tid]
        job.state = "RUNNING"
        self.jobs[job.job_id] = job
        for t in tasks:
            self.tasks[t.task_id] = t
        return job

    def cancel_job(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job is not None:
            job.state = "CANCELLED"
        for t in self.list_tasks(job_id):
            if t.state not in ("COMPLETED", "FAILED"):
                t.state = "CANCELLED"

    def save_result(self, task: Any, png_bytes: bytes, sha256: str) -> Path:
        from nova.clock import now_utc

        dest_dir = self.tiles_dir / task.job_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / f"{task.task_id}.png"
        path.write_bytes(png_bytes)
        task.result_sha256 = sha256
        task.result_path = str(path)
        task.state = "COMPLETED"
        task.completed_at = now_utc()
        self.tasks[task.task_id] = task
        return path


class FallbackScheduler:
    def __init__(self, store: Any, bus: EventBus) -> None:
        self.store = store
        self.bus = bus

    def submit_job(self, job: Any, tasks: list[Any]) -> Any:
        return self.store.submit_job(job, tasks)

    def cancel_job(self, job_id: str) -> None:
        self.store.cancel_job(job_id)

    def accept_result(
        self,
        job_id: str,
        task_id: str,
        png_bytes: bytes,
        lease_gen: int,
        sha256: str,
        node_id: str | None = None,
    ) -> dict[str, str]:
        tasks = self.store.list_tasks(job_id)
        if tasks and all(t.state == "COMPLETED" for t in tasks):
            job = self.store.get_job(job_id)
            if job is not None:
                job.state = "COMPLETED"
                self.bus.emit("JOB_COMPLETED", job_id=job_id, total=len(tasks))
        return {"status": "accepted"}

    def on_complete(self, *args: Any, **kwargs: Any) -> None:
        return None


def _inject_sim_nodes(store: Any, count: int) -> None:
    backends = ("cuda", "rocm", "metal")
    vendors = ("nvidia", "amd", "apple")
    models = ("RTX 4090", "RX 7900 XTX", "M3 Max")
    mem = (24576, 24576, 16384)
    for i in range(count):
        idx = i % 3
        backend = backends[idx]
        node = NodeManifest(
            node_id=f"sim-{i + 1:02d}",
            hostname=f"sim-{i + 1:02d}",
            devices=[
                Device(
                    device_id=f"{backend}:0",
                    backend=backend,
                    vendor=vendors[idx],
                    model=models[idx],
                    memory_total_mb=mem[idx],
                    memory_free_mb=mem[idx] // 2,
                )
            ],
            benchmark_scores={"sd.t2i.v1": round(12.0 - idx * 3.5 - i * 0.1, 2)},
            warmup_ms=round(1000.0 / max(round(12.0 - idx * 3.5 - i * 0.1, 2), 0.1), 1),
            status="online",
        )
        upsert = getattr(store, "upsert_node", None) or getattr(store, "put_node", None)
        if callable(upsert):
            upsert(node)
        elif isinstance(getattr(store, "nodes", None), dict):
            store.nodes[node.node_id] = node


def _build_store(settings: Settings, bus: EventBus, clock: Any | None = None) -> tuple[Any, Any, bool]:
    store_cls = _optional("nova.store", "Store")
    sched_cls = _optional("nova.scheduler", "Scheduler")
    stub = False
    store: Any
    scheduler: Any
    if store_cls is not None:
        store = _construct(
            store_cls,
            data_dir=settings.data_dir,
            tiles_dir=settings.data_dir / "tiles",
            settings=settings,
        )
    else:
        _echo("[nova] nova.store.Store not ready — in-process fallback store")
        store = FallbackStore(settings.data_dir)
        stub = True
    if sched_cls is not None:
        scheduler = _construct(
            sched_cls,
            store=store,
            bus=bus,
            event_bus=bus,
            settings=settings,
            clock=clock,
        )
    else:
        _echo("[nova] nova.scheduler.Scheduler not ready — in-process fallback scheduler")
        scheduler = FallbackScheduler(store, bus)
        stub = True
    return store, scheduler, stub


def _worker_connect_addrs(settings: Settings) -> list[tuple[str, int]]:
    addrs = settings.peer_list()
    if addrs:
        return addrs
    url = settings.coordinator_url
    if not url:
        return []
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname
    if not host:
        return []
    return [(host, settings.control_port)]


def _make_tcp(
    *,
    node_id: str,
    listen: bool,
    connect_addrs: list[tuple[str, int]],
    settings: Settings,
) -> Any | None:
    tcp_cls = _optional("nova.network.tcp", "TcpTransport")
    if tcp_cls is None:
        return None
    return tcp_cls(
        listen_host=settings.control_host if listen else None,
        listen_port=settings.control_port if listen else None,
        connect_addrs=list(connect_addrs),
        node_id=node_id,
    )


def _make_pear(*, node_id: str, settings: Settings) -> Any | None:
    """Hyperswarm sidecar when node + hyperswarm are installed. Optional."""
    pear_cls = _optional("nova.network.pear", "PearTransport")
    if pear_cls is None:
        return None
    available = getattr(pear_cls, "available", None)
    if callable(available) and not available():
        return None
    return pear_cls(node_id=node_id, swarm_topic=settings.swarm_topic)


def _make_control(
    *,
    node_id: str,
    listen: bool,
    connect_addrs: list[tuple[str, int]],
    settings: Settings,
    include_pear: bool = True,
) -> tuple[Any | None, str]:
    """Pear discovery + TCP emergency path as one ControlTransport.

    Local in-process workers should pass include_pear=False so they stay on
    loopback TCP and do not spawn a second Hyperswarm sidecar.
    """
    tcp = _make_tcp(
        node_id=node_id,
        listen=listen,
        connect_addrs=connect_addrs,
        settings=settings,
    )
    pear = _make_pear(node_id=node_id, settings=settings) if include_pear else None
    parts: list[Any] = [t for t in (tcp, pear) if t is not None]
    if not parts:
        return None, "none"
    if tcp is not None and pear is not None:
        hub_cls = _optional("nova.network.hub", "Hub")
        if hub_cls is not None:
            return hub_cls(parts), "pear+tcp"
        return tcp, "tcp"
    if pear is not None:
        return pear, "pear"
    return tcp, "tcp"


def _make_worker(
    *,
    settings: Settings,
    transport: Any,
    identity: NodeIdentity,
    clock: Any | None = None,
) -> Any:
    worker_cls = _optional("nova.worker", "Worker")
    if worker_cls is None:
        raise RuntimeError("nova.worker.Worker is not available yet")
    return worker_cls(settings, transport, identity=identity, clock=clock)


def worker_mode(cli_worker: bool, settings: Settings) -> bool:
    """`--worker` wins; otherwise honor NOVA_ROLE=worker."""
    if cli_worker:
        return True
    return str(getattr(settings, "role", "") or "").strip().lower() == "worker"


def dashboard_only_mode(cli_dashboard: bool, cli_no_local_worker: bool, settings: Settings) -> bool:
    """Dashboard service: HTTP + scheduler, no in-process worker.

    `nova start` still runs a local worker. Only `--dashboard`,
    `--no-local-worker`, or `NOVA_ROLE=dashboard` skip it.
    """
    if cli_dashboard or cli_no_local_worker:
        return True
    return str(getattr(settings, "role", "") or "").strip().lower() == "dashboard"


async def _wait_started(server: Any, timeout_s: float = 8.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not getattr(server, "started", False):
        if getattr(server, "should_exit", False) or loop.time() >= deadline:
            return
        await asyncio.sleep(0.05)


def _coordinator_url(settings: Settings) -> str:
    return (settings.coordinator_url or settings.public_http_url()).rstrip("/")


def _http_get(settings: Settings, path: str) -> Any:
    url = f"{_coordinator_url(settings)}{path}"
    try:
        with httpx.Client(timeout=5.0) as client:
            res = client.get(url)
            res.raise_for_status()
            return res.json()
    except httpx.ConnectError as exc:
        _echo(f"Coordinator not reachable at {url} — start it with `nova start`", err=True)
        raise typer.Exit(1) from exc
    except httpx.HTTPError as exc:
        _echo(f"HTTP error from {url}: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def start(
    worker: bool = typer.Option(False, "--worker", help="Worker only; do not bind HTTP."),
    dashboard: bool = typer.Option(
        False,
        "--dashboard",
        help="Job dashboard service: HTTP + scheduler, no local worker.",
    ),
    dummy: bool = typer.Option(False, "--dummy", help="Dummy kernel, no torch."),
    simulate_workers: int = typer.Option(
        0,
        "--simulate-workers",
        help="Inject N fake workers (rehearsal). Parsed even if workers land later.",
    ),
    no_local_worker: bool = typer.Option(
        False,
        "--no-local-worker",
        help="Do not start the in-process worker next to the dashboard.",
    ),
) -> None:
    """Coordinator HTTP + TCP control. Also runs a local worker unless --worker / --dashboard."""
    settings = load_settings(
        dummy=True if dummy else None,
        role="worker" if worker else ("dashboard" if dashboard else None),
    )
    as_worker = worker_mode(worker, settings)
    dashboard_only = (not as_worker) and dashboard_only_mode(dashboard, no_local_worker, settings)
    if dashboard:
        dashboard_only = True
    if not settings.advertise_host:
        settings = settings.model_copy(update={"advertise_host": detect_lan_ip()})
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    ident = load_or_create(settings.data_dir)
    bus = EventBus()
    clock = Clock()

    _echo(f"[nova] node_id     {ident.node_id}")
    if as_worker:
        role_label = "worker"
    elif dashboard_only:
        role_label = "dashboard"
    else:
        role_label = "coordinator"
    _echo(f"[nova] role        {role_label}")
    if dummy or settings.dummy:
        _echo("[nova] kernel      dummy (no torch)")

    if as_worker:
        url = settings.coordinator_url or settings.public_http_url()
        _echo(f"[nova] coordinator {url}")
        addrs = _worker_connect_addrs(settings)
        if _optional("nova.worker", "Worker") is None:
            _echo("Cannot start worker: nova.worker.Worker is not available yet.", err=True)
            raise typer.Exit(1)
        transport, kind = _make_control(
            node_id=ident.node_id,
            listen=False,
            connect_addrs=addrs,
            settings=settings,
            include_pear=True,
        )
        pear_ok = kind in {"pear", "pear+tcp"}
        if transport is None or kind == "none" or (not pear_ok and not addrs):
            _echo(
                "Worker needs Pear (cd pear && npm install) or "
                "NOVA_COORDINATOR_URL / NOVA_PEERS=host:7946.",
                err=True,
            )
            raise typer.Exit(1)
        if pear_ok and not addrs and not settings.coordinator_url:
            _echo(f"[nova] control     {kind} (set NOVA_COORDINATOR_URL if PNG PUT fails)")
        else:
            _echo(f"[nova] control     {kind}")

        async def _run_worker() -> None:
            inst = _make_worker(settings=settings, transport=transport, identity=ident, clock=clock)
            try:
                await inst.run()
            finally:
                stop = getattr(inst, "stop", None)
                if callable(stop):
                    try:
                        await stop()
                    except Exception:
                        pass

        try:
            asyncio.run(_run_worker())
        except KeyboardInterrupt:
            _echo("\n[nova] worker stopped")
        except Exception as exc:
            _echo(f"Cannot start worker: {exc}", err=True)
            raise typer.Exit(1) from exc
        return

    from nova.api import create_app

    store, scheduler, stub = _build_store(settings, bus, clock=clock)
    fa_app = create_app(store, scheduler, bus, settings, settings.data_dir)
    public = settings.public_http_url()
    typer.secho(f"[nova] dashboard  {public}", fg="green", bold=True)
    _echo(f"[nova] health     {public}/health")
    _echo(f"[nova] control    {settings.advertise_host}:{settings.control_port}")
    if stub:
        _echo("[nova] coordinator modules incomplete — HTTP/dashboard still up")

    async def _run_coordinator() -> None:
        import uvicorn

        coordinator = None
        local_worker = None
        cluster = None
        coord_cls = _optional("nova.coordinator", "Coordinator")
        sim_broker_cls = _optional("nova.simulate", "InProcessBroker")
        sim_run = _optional("nova.simulate", "run_simulated_workers")
        use_sim = bool(simulate_workers) and sim_broker_cls is not None and callable(sim_run)

        if use_sim:
            _echo(f"[nova] --simulate-workers {simulate_workers}")
            broker = sim_broker_cls()
            transport = broker.attach(ident.node_id, coordinator=True)
            if coord_cls is not None:
                coordinator = coord_cls(store, scheduler, transport, bus, settings, clock, ident)
                await coordinator.start()
                fa_app.state.coordinator = coordinator
                _echo("[nova] in-process control plane up")
            else:
                _echo("[nova] nova.coordinator.Coordinator not ready — HTTP only")
                _inject_sim_nodes(store, simulate_workers)
        else:
            if simulate_workers:
                _echo(f"[nova] --simulate-workers {simulate_workers} (display cards only)")
                _inject_sim_nodes(store, simulate_workers)
                bus.emit("SYSTEM", message=f"simulated {simulate_workers} workers")
            peer_addrs = _worker_connect_addrs(settings)
            if peer_addrs:
                _echo("[nova] dialing     " + ", ".join(f"{h}:{p}" for h, p in peer_addrs))
            else:
                _echo(
                    "[nova] cluster     standalone — set NOVA_PEERS=host:7946 "
                    "(or tcp://host:port) to join another node"
                )
            if settings.coordinator_url:
                _echo(f"[nova] png upload {settings.coordinator_url}")
            transport, kind = _make_control(
                node_id=ident.node_id,
                listen=True,
                connect_addrs=peer_addrs,
                settings=settings,
                include_pear=True,
            )
            if transport is None:
                _echo("[nova] control plane not ready — HTTP only")
            elif coord_cls is not None:
                coordinator = coord_cls(store, scheduler, transport, bus, settings, clock, ident)
                await coordinator.start()
                fa_app.state.coordinator = coordinator
                bound = getattr(transport, "bound_port", None) or settings.control_port
                from nova.hardware import probe_devices

                probed = probe_devices()
                accel = [d for d in probed if d.backend != "cpu"] or probed
                self_node = NodeManifest(
                    node_id=ident.node_id,
                    hostname=settings.advertise_host or ident.node_id,
                    status="online",
                    http_url=settings.public_http_url(),
                    control_host=settings.advertise_host,
                    control_port=int(bound) if bound else settings.control_port,
                    devices=accel,
                )
                store.put_node(self_node)
                touch = getattr(store, "touch_node", None)
                if callable(touch):
                    touch(ident.node_id)
                backends = ",".join(sorted({d.backend for d in accel}))
                _echo(f"[nova] devices    {backends or 'none'}")
                _echo(f"[nova] control plane {kind} :{bound}")
            else:
                try:
                    await transport.start()
                    _echo(f"[nova] control plane {kind}")
                except Exception as exc:
                    _echo(f"[nova] control plane failed: {exc}", err=True)

        config = uvicorn.Config(
            fa_app,
            host=settings.http_host,
            port=settings.http_port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        serve_task = asyncio.create_task(server.serve(), name="nova-http")
        try:
            await _wait_started(server)
            if use_sim:
                cluster = await sim_run(
                    coordinator if coordinator is not None else broker,
                    simulate_workers,
                    coordinator_id=ident.node_id,
                )
                _echo(f"[nova] simulated {simulate_workers} workers attached")
            elif (
                not dashboard_only
                and not simulate_workers
                and _optional("nova.worker", "Worker") is not None
            ):
                control_port = getattr(transport, "bound_port", None) or settings.control_port
                worker_ident = load_or_create(settings.data_dir / "local-worker")
                worker_addrs = [("127.0.0.1", int(control_port))]
                for host, port in peer_addrs:
                    if (host, int(port)) not in worker_addrs:
                        worker_addrs.append((host, int(port)))
                worker_transport, worker_kind = _make_control(
                    node_id=worker_ident.node_id,
                    listen=False,
                    connect_addrs=worker_addrs,
                    settings=settings,
                    include_pear=False,
                )
                if worker_transport is None:
                    _echo("[nova] local worker skipped — TcpTransport missing")
                else:
                    try:
                        local_worker = _make_worker(
                            settings=settings,
                            transport=worker_transport,
                            identity=worker_ident,
                            clock=clock,
                        )
                        asyncio.create_task(local_worker.run(), name="nova-local-worker")
                        _echo(f"[nova] local worker {worker_ident.node_id} ({worker_kind})")
                    except Exception as exc:
                        _echo(f"[nova] local worker failed to start: {exc}", err=True)
                        local_worker = None
            elif dashboard_only:
                _echo("[nova] dashboard service — workers join with `nova worker`")
            elif _optional("nova.worker", "Worker") is None:
                _echo("[nova] nova.worker.Worker not ready — dashboard/API only")
            await serve_task
        finally:
            server.should_exit = True
            if local_worker is not None:
                stop = getattr(local_worker, "stop", None)
                if callable(stop):
                    try:
                        await asyncio.wait_for(stop(), timeout=1.5)
                    except Exception:
                        pass
            if cluster is not None:
                stop_all = getattr(cluster, "stop_all", None)
                if callable(stop_all):
                    try:
                        await asyncio.wait_for(stop_all(graceful=True), timeout=1.5)
                    except Exception:
                        pass
            if coordinator is not None:
                stop = getattr(coordinator, "stop", None)
                if callable(stop):
                    try:
                        await asyncio.wait_for(stop(), timeout=1.5)
                    except Exception:
                        pass
            if not serve_task.done():
                serve_task.cancel()
                try:
                    await serve_task
                except (asyncio.CancelledError, Exception):
                    pass

    try:
        asyncio.run(_run_coordinator())
    except KeyboardInterrupt:
        _echo("\n[nova] coordinator stopped")


@app.command()
def dashboard(
    dummy: bool = typer.Option(False, "--dummy", help="Dummy kernel, no torch."),
    simulate_workers: int = typer.Option(
        0,
        "--simulate-workers",
        help="Inject N fake workers (rehearsal).",
    ),
) -> None:
    """Job dashboard service: HTTP API + live gallery. Workers join separately."""
    start(
        worker=False,
        dashboard=True,
        dummy=dummy,
        simulate_workers=simulate_workers,
        no_local_worker=True,
    )


@app.command()
def worker(
    dummy: bool = typer.Option(False, "--dummy", help="Dummy kernel, no torch."),
) -> None:
    """Worker service: pull tiles from the dashboard and PUT PNG results."""
    start(
        worker=True,
        dashboard=False,
        dummy=dummy,
        simulate_workers=0,
        no_local_worker=False,
    )


@app.command()
def nodes() -> None:
    """List nodes known to the coordinator."""
    settings = load_settings()
    if not settings.advertise_host:
        settings = settings.model_copy(update={"advertise_host": detect_lan_ip()})
    payload = _http_get(settings, "/nodes")
    if not payload:
        _echo("No nodes (is anything running `nova start`?)")
        return
    for node in payload:
        nid = node.get("node_id", "?")
        status = node.get("display_status") or node.get("status") or "?"
        model = node.get("primary_model") or "-"
        backend = node.get("primary_backend") or "-"
        score = node.get("score")
        score_s = f"{score:.2f}" if isinstance(score, (int, float)) else "-"
        _echo(f"{nid:16}  {status:8}  {model:16}  {backend:6}  score={score_s}")


@app.command()
def jobs() -> None:
    """List jobs on the coordinator."""
    settings = load_settings()
    if not settings.advertise_host:
        settings = settings.model_copy(update={"advertise_host": detect_lan_ip()})
    payload = _http_get(settings, "/jobs")
    if not payload:
        _echo("No jobs.")
        return
    for job in payload:
        jid = job.get("job_id") or job.get("name")
        state = job.get("state")
        done = job.get("completed", 0)
        total = job.get("total", 0)
        _echo(f"{jid:20}  {state:10}  {done}/{total}")


@app.command("run")
def run_job(
    yaml_path: Path = typer.Argument(..., exists=True, readable=True, help="Gallery yaml"),
) -> None:
    """Submit a gallery job (demo/gallery.yaml)."""
    settings = load_settings()
    if not settings.advertise_host:
        settings = settings.model_copy(update={"advertise_host": detect_lan_ip()})
    url = f"{_coordinator_url(settings)}/jobs"
    body = {"path": str(yaml_path.resolve())}
    try:
        with httpx.Client(timeout=10.0) as client:
            res = client.post(url, json=body)
            if res.status_code >= 400:
                _echo(f"Submit failed ({res.status_code}): {res.text}", err=True)
                raise typer.Exit(1)
            job = res.json()
    except httpx.ConnectError as exc:
        _echo(f"Coordinator not reachable at {url} — start it with `nova start`", err=True)
        raise typer.Exit(1) from exc
    jid = job.get("job_id")
    total = job.get("total")
    _echo(f"submitted {jid}  ({total} tiles)")
    _echo(f"dashboard {_coordinator_url(settings)}")


@app.command()
def benchmark(
    dummy: bool = typer.Option(False, "--dummy", help="Skip torch; print a dummy score."),
) -> None:
    """Warmup one image and print the sd.t2i.v1 score."""
    settings = load_settings(dummy=True if dummy else None)
    if dummy or settings.dummy:
        _echo("dummy kernel  latency_ms=50  score=20.00")
        return
    load_kernel = _optional("nova.kernels", "load_kernel") or _optional("nova.kernels.base", "load_kernel")
    probe = _optional("nova.hardware", "probe_devices")
    preferred = _optional("nova.hardware", "preferred_device")
    if callable(load_kernel) and callable(probe):
        try:
            kernel = load_kernel(settings.kernel, settings)
            devices = probe()
            device = preferred(devices) if callable(preferred) else None
            device = device or (devices[0] if devices else None)
            if device is None:
                raise RuntimeError("no devices found")
            result = kernel.warmup(device)
            latency = max(int(getattr(result, "execution_ms", 0) or 0), 1)
            score = 1000.0 / latency
            backend = getattr(device, "backend", "?")
            _echo(f"{backend}  latency_ms={latency}  score={score:.2f}")
            return
        except Exception as exc:
            _echo(f"Benchmark failed: {exc}", err=True)
            raise typer.Exit(1) from exc
    _echo(
        "Cannot benchmark: nova.kernels / nova.hardware not available yet. "
        "Use `nova benchmark --dummy` for a no-torch rehearsal.",
        err=True,
    )
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
