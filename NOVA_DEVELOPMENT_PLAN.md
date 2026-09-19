# NOVA Development Plan

**Project:** NOVA — Networked On-demand Virtual Acceleration
**Hackathon:** Hack the North 2026
**Primary sponsor lane:** Tether — Best Sovereign App
**Secondary:** Sentry, Warp
**MVP workload:** parallel Stable Diffusion image batch
**Primary demo message:** **Three runtimes. One compute pool.**

This is the implementation contract for a 36-hour live demo. Build the timeline, not a platform.

---

# 0. What we are actually building

A **coordinator-led P2P rendering runtime** that turns heterogeneous idle GPUs into one job pool.

For judging, there is **one workload**:

```text
24 prompts  →  24 independent SD images  →  live gallery
```

A user submits a gallery job. NOVA:

1. discovers workers over Pear
2. reads each worker's devices and backend
3. benchmarks a single warmup image
4. leases one prompt per free slot
5. worker generates a PNG and uploads the bytes
6. dashboard tiles fill in live, tagged by GPU vendor
7. if a worker dies, unfinished tiles requeue
8. job completes when every tile has an image

That is the whole MVP.

**Not in the live demo:** video frames, FFmpeg, interpolation, per-frame SD, Flux, SD3, credits, ledgers, signatures, Docker GPUs.

---

# 1. Honest topology

Do not pretend there is no coordinator. For this hackathon:

```text
Submitter machine  =  coordinator + scheduler + gallery store + HTTP API + dashboard
Worker machines    =  model runtime + pull client
Pear               =  discovery + control-plane messages
HTTP               =  image bytes (data plane)
```

```text
                    ┌─────────────────────────┐
                    │   Coordinator           │
                    │   CLI / Dashboard / API │
                    │   scheduler + gallery   │
                    └────────────┬────────────┘
                         control │ Pear
                         bytes   │ HTTP
                    ┌────────────┴────────────┐
                    ▼                         ▼
           ┌────────────────┐        ┌────────────────┐
           │ Worker A       │        │ Worker B       │
           │ RTX 4090 CUDA  │        │ RX 7900  ROCm  │
           └────────────────┘        └────────────────┘
                    │
                    ▼
           ┌────────────────┐
           │ Worker C       │
           │ M3 Max   MPS   │
           └────────────────┘
```

The coordinator is allowed to also be a worker (the NVIDIA box usually should). Pear finds nodes. HTTP moves PNGs. One scheduler per job: the submitter.

Pitch this accurately:

> Workers discover each other peer-to-peer. The job owner coordinates the batch. There is no cloud GPU provider.

Do not pitch: coordinatorless consensus, compute marketplace, or a generic cloud replacement.

---

# 2. Why Stable Diffusion batch (not video)

Each image is an independent task. That matches pull scheduling, failover, and heterogeneous GPUs.

Cross-vendor numeric drift is **fine** here: tiles are different prompts, not adjacent video frames. We will not stitch CUDA frame 41 next to Metal frame 42.

| Workload | Use? |
|---|---|
| SD image batch (this MVP) | **Yes. Only demo.** |
| Deterministic video filter / upscale | Future. Not 36h. |
| SD img2img on every video frame | No. Flickers even on one GPU. |
| SVD / AnimateDiff / video diffusion | No. Not parallel, too slow. |

---

# 3. Live demo contract

**Duration:** 2–3 minutes talking. Generation happens **during** the talk.

**Hardware target:** 3 physical machines, ≥2 GPU vendors. Ideal: CUDA + ROCm + Metal.

**Job:**

```text
24 images
512×512
SD-Turbo or SD 1.5 + LCM, 4 steps
guidance as required by the turbo recipe (often 0 for SD-Turbo)
one seed per tile, unique
```

**Timing budget:** wall clock 20–45 seconds on mixed hardware if models are already in VRAM.

## 3.1 What judges see

1. Three machines on a table. Dashboard shows CUDA / ROCm / Metal badges.
2. `nova run demo/gallery.yaml`
3. Empty 4×6 grid.
4. Tiles appear. Fast GPU fills more squares. Each tile labeled with node + backend.
5. **Ctrl+C the AMD worker** (do not close the lid). Pear socket drops.
6. Event feed within 1–2s: `NODE_DISCONNECTED` → tiles requeued → NVIDIA/Apple finish them.
7. Full gallery. Tagline: **Three runtimes. One compute pool.**

## 3.2 What must be true before judging

- [ ] Model weights already on disk on every worker. No Hugging Face during demo.
- [ ] One warmup image already run on each worker (`nova start` does this).
- [ ] Coordinator bound to LAN IP, dashboard open on projector.
- [ ] Ctrl+C failover rehearsed 10 times (must requeue in 1–2s, not 30s).
- [ ] Recorded backup of the same flow.

---

# 4. Model policy (non-negotiable)

**One pipeline. One set of weights. Every worker.**

Preference order:

1. **SD-Turbo** (stabilityai/sd-turbo) — 1–4 steps, 512². Best live-demo speed.
2. **SDXL-Turbo** — only if every worker can hold it and hit <8s/image.
3. **SD 1.5 + LCM-LoRA** — fallback if Turbo is broken on MPS or ROCm.
4. Last resort: **SD 1.5, 8 steps, 512²**.

**Do not use:** Flux, SD3.x, video models, ControlNet, refiner, img2img, LoRA stacks beyond the one LCM if needed.

Load once at worker startup, keep on device for the process lifetime.

Pin **device and dtype per backend**. Do not share one universal dtype config.

```text
NVIDIA  →  torch.device("cuda")   dtype=float16
AMD     →  torch.device("cuda")   dtype=float16   # HIP/ROCm
Apple   →  torch.device("mps")    dtype=float16   + attention slicing if warmup OOMs
CPU     →  torch.device("cpu")    dtype=float32   # last resort only
```

```python
if backend == "metal":
    pipe.enable_attention_slicing()
```

RX 7900 XTX can run PyTorch on Windows 11 with current ROCm components; the full ROCm stack is still not on Windows. Pin the exact Python / torch / ROCm (or CUDA / MPS) versions during Hour 0 and freeze them. If AMD cannot produce one image in Hour 0, that worker is CPU or recorded — do not spend the afternoon debugging ROCm.

Same prompt + seed **will not** match across vendors. Do not verify pixels. Verify: PNG decodes, non-zero size, SHA-256 stored as transport integrity only.

---

# 5. Data plane

There is no shared filesystem. A task never says `frame_0042.png` as if a path existed on the worker.

**Control plane (Pear):** task metadata.
**Data plane (HTTP on the coordinator):** PNG bytes.

Coordinator serves:

```text
GET  /health
GET  /jobs/{job_id}/tasks/{task_id}/input.json     # prompt payload
PUT  /jobs/{job_id}/tasks/{task_id}/result         # body = PNG
     header X-Nova-Lease-Gen: <int>
     header X-Nova-Sha256: <hex>
     header X-Nova-Node-Id: <assigned node_id>
GET  /jobs/{job_id}/tiles/{task_id}.png            # dashboard
```

Worker flow:

```text
WORK_REQUEST
  → TASK_OFFER {task_id, job_id, prompt, seed, steps, width, height,
                fetch_url, upload_url, lease_seconds, lease_gen}
  → TASK_ACCEPT
  → (optional GET input.json if payload was not inlined)
  → run pipeline
  → PUT PNG
  → TASK_COMPLETE {result_url, sha256, execution_ms, lease_gen}
```

For this MVP, **inline the prompt in TASK_OFFER**. The PUT of the PNG is the only bulk transfer. 512² PNG is ~0.5–1.5 MB. 24 images is nothing on LAN.

Coordinator rejects PUT if:

- lease_gen does not match current lease
- `X-Nova-Node-Id` is missing or is not the assigned node
- task is not LEASED/RUNNING to that node
- body empty / not a decodable PNG

Stale winner: **first valid PUT + matching lease_gen wins**. Later duplicates are ACKed as ignored.

---

# 6. Control plane (Pear)

Pear is discovery + messages. It is not the image transport for MVP.

Minimum messages:

```text
HELLO
NODE_MANIFEST
NODE_UPDATE
HEARTBEAT
NODE_DISCONNECTED      # peer socket closed; coordinator marks OFFLINE immediately
NODE_GOODBYE           # explicit graceful leave; same as disconnect

JOB_ANNOUNCE

WORK_REQUEST
TASK_OFFER
TASK_ACCEPT
TASK_REJECT

TASK_STARTED
TASK_PROGRESS
TASK_COMPLETE
TASK_FAILED

RESULT_ACK
```

HELLO carries `protocol_version: 1` and `node_id`.

If Pear bulk streams are free later, they can replace HTTP. Do not block the demo on that.

### Discovery (concrete, not hand-wavy)

Sharing a Hyperswarm topic is **not** LAN-only discovery. Topic lookup goes through HyperDHT. Venue WiFi / NAT can delay or fail it.

**Primary judging path:** Pear / Hyperswarm on a **phone hotspot** (or venue net if it works in rehearsal). Coordinator prints `http://<lan-ip>:8080` and announces it in JOB_ANNOUNCE.

**Emergency path (must exist before judging):**

```text
NOVA_COORDINATOR_URL=http://192.168.x.x:8080
NOVA_PEERS=192.168.x.x:port   # optional direct connect
```

Worker connects to that HTTP URL for bytes and, if Pear discovery failed, opens a **direct** control socket to the coordinator. No DHT, no mDNS required for the emergency path.

**Optional extra, not a dependency:** mDNS (`_nova._tcp`) to find the coordinator on LAN if someone has 30 spare minutes after failover is green.

Rehearse both paths. Never write “local bootstrap” without a typed-IP fallback that you have actually run.

---

# 7. Node identity (mandatory)

Persistent identity is **required**, not optional.

On first start, write:

```text
.nova/node.json
```

```json
{
  "node_id": "nova-7f4a9c2e",
  "created_at": "2026-09-19T00:00:00Z"
}
```

`node_id` never changes across `nova start`. Rejoin restores the same id. Leases always name this id. Random ids on restart are a demo bug.

---

# 8. Hardware model

A node is not one `backend: str`. A node has **devices**.

```python
@dataclass
class Device:
    device_id: str          # "cuda:0" | "mps:0" | "cpu:0"
    backend: str            # "cuda" | "rocm" | "metal" | "cpu"
    vendor: str             # "nvidia" | "amd" | "apple" | "cpu"
    model: str
    memory_total_mb: int
    memory_free_mb: int     # hint only, never the scheduler lock
    load: float             # 0..1, informational

@dataclass
class NodeManifest:
    node_id: str
    hostname: str
    os: str
    architecture: str
    devices: list[Device]
    supported_kernels: list[str]      # ["sd.t2i.v1"]
    benchmark_scores: dict[str, float]
    max_concurrency: int              # usually 1 for SD
    current_slots_used: int
    status: str                       # online | suspect | offline
```

Probe **all** devices. Prefer the fastest compatible accelerator. CPU is fallback, not advertised as CUDA.

Apple memory is unified RAM. Report a conservative `memory_free_mb` haircut (e.g. 50% of total). Do not claim 36 GB VRAM.

Dashboard memory total is the sum of **accelerator** memories we actually run on, with Apple labeled “unified”.

---

# 9. Kernel

One logical kernel:

```text
sd.t2i.v1
```

```python
class NovaKernel(ABC):
    kernel_id: str

    @classmethod
    def compatible(cls, device) -> bool: ...

    @classmethod
    def estimate(cls, task, device) -> float: ...   # seconds

    async def execute(self, task) -> KernelResult: ...
```

`sd.t2i.v1` implementation: Hugging Face `diffusers` pipeline, local files only (`local_files_only=True`).

`KernelResult`: PNG bytes, sha256, execution_ms, device_id, backend.

Worker rejects the offer if no compatible device. Do not silently run a different kernel.

---

# 10. Job and task models

```python
@dataclass
class Job:
    job_id: str
    job_type: str                 # "sd_gallery"
    kernel_id: str                # "sd.t2i.v1"
    created_at: datetime          # UTC
    state: str                    # QUEUED|RUNNING|COMPLETED|FAILED|CANCELLED
    scheduler_policy: str         # "adaptive_pull"
    requirements: JobRequirements
    prompts: list[PromptSpec]

@dataclass
class JobRequirements:
    min_memory_mb: int            # 4096
    allowed_backends: list[str]   # ["cuda", "rocm", "metal"]  # cpu only if desperate
    width: int                    # 512
    height: int
    steps: int
    model_id: str

@dataclass
class Task:
    task_id: str
    job_id: str
    shard_index: int
    kernel_id: str
    prompt: str
    seed: int
    steps: int
    width: int
    height: int
    min_memory_mb: int            # copied from job at split time
    allowed_backends: list[str]   # copied from job
    state: str
    assigned_node: str | None
    lease_gen: int
    lease_expires_at: datetime | None
    attempt_count: int            # starts at 0; incremented when a lease is granted
    max_attempts: int             # 3  →  at most 3 leases, not 1 try + 3 retries
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    result_sha256: str | None
    result_path: str | None
    backend_used: str | None
```

Task states:

```text
QUEUED → LEASED → RUNNING → COMPLETED
                 ↘ FAILED → QUEUED if attempt_count < max_attempts else FAILED
LEASED/RUNNING → QUEUED on lease expiry, NODE_DISCONNECTED, or NODE_OFFLINE
                 (attempt_count already includes this lease)
any → CANCELLED
```

Semantics:

```text
attempt_count += 1   # when the lease is granted
on fail/expire/disconnect:
    if attempt_count < max_attempts:  requeue (QUEUED)
    else:                             FAILED
```

`max_attempts = 3` means three leases total, not “1 try + 3 retries”. There is no `FAILED_PERMANENT` name. Exhausted tiles stay `FAILED` (red square). Do not hang the job forever.

---

# 11. Gallery splitter

```text
demo/gallery.yaml  →  24 tasks, shard_index 0..23
```

Example:

```yaml
job:
  name: nova-gallery
  type: sd_gallery
  kernel: sd.t2i.v1

model:
  id: stabilityai/sd-turbo
  steps: 4
  width: 512
  height: 512

requirements:
  min_memory_mb: 4096
  allowed_backends: [cuda, rocm, metal]

prompts:
  - text: "neon street at night, cinematic, 35mm"
    seed: 1000
  - text: "ancient library, volumetric light"
    seed: 1001
  # ... 24 total, checked into git
```

Seeds and prompts are **deterministic and committed**. Same yaml always produces the same 24 slots (not the same pixels across vendors).

---

# 12. Scheduler

Policy: **`adaptive_pull` only.** No work stealing. No push except answering a pull.

Worker sends `WORK_REQUEST {node_id, available_slots, device_ids}`.

Coordinator:

1. drop request if node not `online`
2. filter queued tasks with `is_compatible`
3. pick the oldest queued task (FIFO is enough)
4. `lease_gen += 1`, `attempt_count += 1`, set `LEASED`, `assigned_node`, `lease_expires_at = now_utc + lease_seconds`
5. `TASK_OFFER`

### Compatibility (single source of truth)

A node is eligible iff **all** of:

```text
status == online
kernel_id in supported_kernels
some device.backend in task.allowed_backends
that device.memory_total_mb >= task.min_memory_mb
current_slots_used < max_concurrency
```

Do not schedule on `memory_free_mb`. Reserve **slots**, not VRAM snapshots. SD MVP `max_concurrency = 1` per worker unless a 4090 warmup proves 2 is stable.

### Why the 4090 gets more tiles

Not because we “distribute by benchmark score.”

```text
4090 finishes → WORK_REQUEST → next tile
4090 finishes → WORK_REQUEST → next tile
M3 still on tile 3
```

Capability filter decides **who may** run a tile. Pull decides **who gets** the next one. Faster devices naturally consume more work.

Benchmarks are used for:

- lease length (`estimate * 2.5`)
- dashboard numbers
- rare tie-break if two WORK_REQUESTs sit in the coordinator queue at once

```python
def score_node(node, task):
    return node.benchmark_scores.get(task.kernel_id, 0.1)
```

Missing score: still eligible, treated as slow (0.1). Never KeyError. Do **not** penalize `gpu_load`.

### Benchmark

On join, after pipeline load, generate **one** discarded warmup image. Score = `1000 / latency_ms` (higher is faster). Broadcast via NODE_MANIFEST.

Do not use TFLOPS. Do not pitch “weighted score scheduling.” Pitch: **filter + pull.**

---

# 13. Leases (fenced)

Never assign a task forever.

```text
lease_seconds = clamp(estimate * 2.5, min=20, max=90)
```

`estimate` comes from the worker's benchmark for `sd.t2i.v1` (or 15s default).

Rules:

- Offer includes `lease_gen` (monotonic int per task).
- `TASK_PROGRESS` from the **assigned node** with the **current lease_gen** renews `lease_expires_at`.
- Heartbeat does **not** renew the task lease. A hung GPU that still pings must still expire.
- On expiry, NODE_DISCONNECTED, or NODE_OFFLINE: if state is LEASED or RUNNING, clear assigned_node; if `attempt_count < max_attempts` → QUEUED, else FAILED. Do not increment `attempt_count` again here (it incremented when the lease was granted).
- `TASK_COMPLETE` / result PUT with stale `lease_gen` is ignored.
- Coordinator clock is the only clock. Workers do not decide expiry. Timestamps are UTC.

If the slow Metal box is still generating when the lease expires, CUDA may take the tile. Metal's late PUT is discarded. That is correct.

Poison tile: do not immediately re-offer the same task to the node that just FAILED it (one-cycle blacklist).

---

# 14. Liveness: disconnect first, heartbeat as fallback

Two paths. The demo uses the first.

```text
1. Socket closed / Ctrl+C / NODE_GOODBYE
       → NODE_DISCONNECTED immediately
       → status = offline
       → requeue that node's LEASED/RUNNING tasks
       → dashboard LOST in 1–2 seconds

2. Heartbeat timeout (fallback only)
       interval  5s
       suspect  15s
       offline  30s
```

Do **not** close a laptop lid for the live demo. Lid-close can sleep the NIC, stall TCP, and force the 30s heartbeat path. Judges will watch a spinner.

Demo kill: **Ctrl+C the worker process** so Pear’s peer-close fires.

Heartbeat 2/4/6 is too tight for venue WiFi. 5/15/30 stays as the backup when a machine hard-freezes without a FIN.

Suspect: keep leases, dashboard yellow.
Offline (either path): requeue, emit NODE_OFFLINE / NODE_DISCONNECTED.

NODE_RECOVERED / rejoin: same `node_id`, manifest refresh, worker may pull again. In-flight tasks were already requeued; do not resurrect them on the old lease_gen.

**Mandatory:** inference must not block this loop. See §16.

---

# 15. Failure

Normal events: Ctrl+C, socket drop, WiFi blip, OOM, pipeline exception, lease timeout.

Live-demo flow:

```text
worker generating tile 14
    → Ctrl+C
    → Pear socket closed
    → NODE_DISCONNECTED (immediate)
    → requeue tile 14 (new lease_gen)
    → another worker pulls
    → tile 14 appears on a different GPU   # 1–2 seconds, not 30
```

`max_attempts = 3` per tile. After that, red tile, job still shows the rest. Demo success means 24/24; keep prompts easy and steps low so this does not happen.

---

# 16. Worker runtime

## 16.1 Process shape (mandatory)

`pipe(...)` is a **blocking** PyTorch call. If it runs on the asyncio thread, HEARTBEAT and TASK_PROGRESS stop, the coordinator thinks the worker died, and the tile is double-assigned.

```text
Worker process
├── Async control loop          # never blocked by GPU
│   ├── Pear / control messages
│   ├── HEARTBEAT every 5s
│   ├── TASK_PROGRESS (~every 2s while a job is leased)
│   └── HTTP PUT
│
└── Compute executor            # thread or process
    └── blocking Diffusers inference
```

```python
result = await asyncio.to_thread(kernel.execute_sync, task)
```

`execute_sync` is the only place `pipe()` runs. Periodic TASK_PROGRESS is emitted from the control loop while the thread is alive, not from inside `pipe()`.

Do not use a `progress` callback inside Diffusers as the heartbeat. If the callback is starved, you still need the independent 5s HEARTBEAT.

Warmup also runs in the compute thread so startup cannot stall HELLO.

## 16.2 Main loop

```text
start
  load node.json (or create)
  probe devices
  load sd.t2i.v1 weights from local disk
  warmup 1 image          # in compute thread
  join Pear swarm
  broadcast NODE_MANIFEST
  loop (async):
      if slots free: WORK_REQUEST
      on TASK_OFFER: TASK_ACCEPT or TASK_REJECT
      TASK_STARTED
      result = await asyncio.to_thread(execute_sync, task)
      PUT png
      TASK_COMPLETE
```

Executor: validate offer, pick compatible device, run pipeline, encode PNG, release slot. Catch OOM and send TASK_FAILED.

---

# 17. Verification (honest)

MVP checks:

- PNG decodes
- width/height match
- sha256 stored

This is **not** correctness of the diffusion. Do not tell judges we cryptographically verify the art. Do not run SSIM across vendors.

---

# 18. API and dashboard

FastAPI on the coordinator. Bind to `127.0.0.1` plus the LAN interface we need. No auth for MVP, **but do not port-forward to the internet**. LAN-only.

```text
GET  /health
GET  /system
GET  /nodes
GET  /jobs
POST /jobs                 # from nova run
GET  /jobs/{job_id}
POST /jobs/{job_id}/cancel
GET  /jobs/{job_id}/tasks
WS   /ws/events
PUT  /jobs/{job_id}/tasks/{task_id}/result
```

Dashboard is **not** an admin panel. One overview:

```text
NOVA NETWORK
3 nodes · CUDA · ROCm · Metal

[GPU cards with backend, model, score, ACTIVE/LOST]

JOB nova-gallery   18/24
[4x6 tiles, filling in, each with vendor chip]

RTX 4090     11
RX 7900 XTX   5
M3 Max        2

EVENT FEED
12:10:16  nova-amd  OFFLINE
12:10:16  requeued tile 14, 15
12:10:17  tile 14  →  nova-4090
```

No settings page. No login. No extra animations.

---

# 19. CLI

```bash
nova start          # coordinator+worker if this machine owns the job, else worker
nova start --worker # worker only
nova nodes
nova benchmark      # warmup + print score
nova run demo/gallery.yaml
nova jobs
```

`nova join` is not a separate command. `nova start` joins the swarm. Two entrypoints cause double-join bugs.

Config: `NOVA_ROLE=coordinator|worker`, `NOVA_SWARM_TOPIC`, `NOVA_COORDINATOR_URL`.

Workers need the coordinator HTTP base URL (printed by coordinator on start, or in `demo/gallery.yaml`). Pear finds peers; HTTP still needs an address. On LAN, coordinator announces `http://<ip>:8080` inside NODE_MANIFEST / JOB_ANNOUNCE so workers do not type IPs by hand.

---

# 20. Telemetry

Optional Sentry tracing if it takes <1 hour after the gallery works:

- job start/complete
- lease grant/expire
- TASK_FAILED
- node offline
- PUT result

Do not block demo on Sentry.

---

# 21. Repo (MVP only)

Do not create modules we will not run.

```text
nova/
├── README.md
├── NOVA_DEVELOPMENT_PLAN.md
├── LICENSE
├── pyproject.toml
├── .env.example
├── nova/
│   ├── main.py
│   ├── config.py
│   ├── identity.py
│   ├── hardware.py
│   ├── network/          # Pear bridge + messages
│   ├── kernels/sd_t2i.py
│   ├── jobs.py           # models, splitter, store
│   ├── scheduler.py      # pull, compatibility, leases
│   ├── worker.py         # async control loop; compute via to_thread
│   └── api.py            # FastAPI + result PUT + WS
├── pear/                 # discovery + control messages
├── dashboard/            # gallery UI
├── demo/
│   ├── gallery.yaml
│   └── prompts.txt
├── scripts/
│   ├── setup.sh
│   ├── download_model.py
│   └── record_backup.sh
└── tests/
    ├── test_scheduler.py
    ├── test_leases.py
    └── test_compatibility.py
```

Explicitly **out of the tree for MVP:** signatures, ledger, credits, work stealing, video aggregation, numerical verifier, Docker GPU, kernel zoo.

---

# 22. Team split (4 people)

## A — Pear / control plane

`pear/` + `nova/network/`

Milestone: two laptops discover each other on hotspot; HELLO + HEARTBEAT; **Ctrl+C one process → NODE_DISCONNECTED in <2s**. Typed `NOVA_COORDINATOR_URL` emergency path also works.

## B — Worker / model

`nova/hardware.py`, `nova/kernels/sd_t2i.py`, `nova/worker.py`

Milestone: CUDA and Metal (and ROCm if hour 0 passed) each produce one 512² PNG from the same yaml prompt. Weights local-only. Heartbeats continue while `pipe()` runs (`asyncio.to_thread`).

## C — Coordinator / scheduler

`nova/jobs.py`, `nova/scheduler.py`, result store, lease expiry

Milestone: 24 tasks pull-scheduled; kill worker; tiles requeue; first valid lease_gen wins.

## D — Dashboard / API / demo

`nova/api.py`, `dashboard/`, `demo/`

Milestone: projector-readable gallery. A judge understands the system without a terminal.

Coordinator HTTP base URL announcement is A+C together.

---

# 23. 36-hour timeline

## Hours 0–2 — Gate

- repo, Python 3.11+, torch, diffusers
- pin dtype: CUDA/ROCm/MPS fp16; test MPS attention slicing
- write down exact torch/ROCm/CUDA/Python versions per box
- **each machine: `download_model.py` then one local image**
- AMD ROCm go/no-go
- pick the model from the preference list based on what actually ran
- swarm topic string, hotspot check, `NOVA_COORDINATOR_URL` emergency path
- projector machine = coordinator

Deliverable: every box has produced a PNG. If not, that box is not a live backend.

## Hours 2–6 — Control plane

Pear swarm, HELLO, HEARTBEAT, NODE_MANIFEST, Python bridge.

Stop here until A↔B ping/pong is boringly reliable.

## Hours 6–10 — Worker

Load pipeline, warmup, execute one TASK-shaped object via `asyncio.to_thread`, write PNG. Prove HEARTBEAT still fires during `pipe()`.

## Hours 10–14 — Coordinator + HTTP data plane

Split yaml → 24 tasks. TASK_OFFER / ACCEPT. PUT PNG. Persist tiles. lease_gen.

Deliverable: two processes on one machine complete a 4-image job over localhost.

## Hours 14–18 — Multi-worker pull

Two physical machines, 24 images, contribution counts move.

## Hours 18–21 — Failover (hard requirement)

Ctrl+C worker at ~40%. NODE_DISCONNECTED in <2s. Tiles requeue. Job finishes. Dashboard shows LOST + reassigned. Lid-close is not the kill method.

If this is not green, do not add UI chrome.

## Hours 21–25 — Gallery UI

Node cards, 4×6 tiles, event feed, contribution counts. No other pages.

## Hours 25–28 — Third backend + polish

Plug Metal or ROCm if not already. Confirm mixed-vendor gallery. Sentry only if idle.

## Hours 28–31 — Sponsor pass

Pear path is already core (Tether). README + Devpost draft. Warp only if someone already uses it.

## Hours 31–36 — Freeze

Bugs, backup recording, one-machine simulation mode (fake second/third worker profiles **only** if a physical backend dies), rehearse 10 times.

---

# 24. Tests that matter

Unit:

- incompatible backend rejected
- offline node rejected
- slot full rejected
- lease expiry → QUEUED and lease_gen bumps
- attempt_count / max_attempts (3 leases, not 1+3)
- stale lease_gen complete ignored
- same node_id after restart
- NODE_DISCONNECTED requeues immediately without waiting 30s

Integration:

- 2 workers, 12 images, all tiles stored
- Ctrl+C worker at 40%, job completes in seconds not 30s
- heartbeats received while a dummy 10s sleep stands in for pipe()
- rejoin uses same node_id and pulls new work
- PUT of a garbage body is rejected

No video pipeline tests.

---

# 25. Backup demo

Live mesh will fail for reasons that are not our code.

Always have:

- LAN (no internet) path
- phone hotspot path
- recorded dashboard session
- recorded full 2-minute take
- pre-generated 24-tile folder so the UI can play back
- `--simulate-workers` on the coordinator that replays manifests + fills tiles (last resort, say so)

Never depend on Hugging Face, public DHT, or venue WiFi for the core loop.

---

# 26. Success criteria

MVP is done when:

```text
[ ] ≥2 physical workers, ideally 3
[ ] ≥2 accelerator backends live
[ ] persistent node ids
[ ] Pear discovery on hotspot; typed NOVA_COORDINATOR_URL emergency path
[ ] local model weights, warmup on start
[ ] 24-tile job from demo/gallery.yaml
[ ] HTTP PUT results, no shared disk
[ ] pull scheduling (faster node more tiles because it asks again)
[ ] compute off the asyncio thread; heartbeats during pipe()
[ ] Ctrl+C → NODE_DISCONNECTED <2s → requeue → job still finishes
[ ] stale duplicate results ignored
[ ] dashboard gallery + event feed + contribution
[ ] demo completes in one talk without a terminal
```

---

# 27. Priority if time dies

1. Local SD image on each box
2. Pear discovery + manifest
3. HTTP result PUT
4. Pull scheduler + leases
5. Failover
6. Gallery dashboard
7. Third vendor polish
8. Sentry

Never spend remaining hours on video, Flux, credits, or sandboxing.

---

# 28. Positioning

Pitch:

> **A peer-to-peer heterogeneous GPU runtime. Write the job once. Every worker runs it on whatever accelerator it actually has.**

Live:

> **Three runtimes. One compute pool.**

Do not pitch a crypto marketplace, a cloud replacement, or a distributed LLM.

Legal/practical notes for the team (do not put on slides):

- Use committed prompts and generated images we own. No scraped celebrity lists.
- Consumer GeForce clustering is a product-law problem later, not a 36h demo problem. Do not label this a GeForce datacenter.
- FastAPI is LAN-only. Workers execute **our** kernel, not arbitrary code.

---

# 29. Taglines

Primary: **NOVA — Write once. Accelerate anywhere.**

Demo: **Three runtimes. One compute pool.**

Technical: **Hardware-adaptive, peer-to-peer batch generation across CUDA, ROCm, and Metal.**

Pitch scheduling as **capability filter + pull**, not “benchmark-weighted distribution.”
