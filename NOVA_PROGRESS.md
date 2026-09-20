# NOVA Development Progress

Last updated: 2026-09-19 (dashboard tabs + FP16 TFLOPS + quota mode)

This is the living handoff ledger for NOVA. Read this file together with
`NOVA_DEVELOPMENT_PLAN.md`, which remains the implementation contract. Update
this document immediately after finishing each source, test, configuration, or
documentation file so the next contributor can see what changed, what was
verified, and what remains.

## Status vocabulary

- **Implemented** — code exists and is connected to the runtime.
- **Unit verified** — focused automated tests pass.
- **Integration verified** — the relevant multi-component path was exercised.
- **Demo verified** — exercised on the intended physical hardware/network flow.
- **Pending** — required by the development plan but not yet implemented.
- **Unverified** — code exists, but the required runtime or hardware proof has
  not been recorded.

Do not promote an item to **Demo verified** based only on source inspection or
unit tests.

## Current baseline

Recorded before the 2026-09-19 continuation work:

- `python -m pytest -q -p no:cacheprovider`: **127 passed**, 3 third-party
  deprecation warnings, in 2.52s (2026-09-19 after Pear startup wiring).
- `python -m compileall -q nova tests`: passed.
- `node --check pear/bridge.js`: passed.
- `node --check dashboard/app.js`: passed.
- `git diff --check`: passed.
- `python -m nova.main --help`: passed; exposes `start`, `nodes`, `jobs`, `run`,
  and `benchmark`.
- `python -m nova.main benchmark --dummy`: passed.
- `python -m ruff check nova tests --statistics`: **82 pre-existing findings**
  (30 `BLE001`, 18 `S110`, and 34 other findings; 22 mechanically fixable).
  Ruff is a tracked quality backlog, not currently a clean gate.

## Development-plan progress

### Coordinator, scheduler, leases, and result store

Status: **Implemented; unit/integration verified in the local test suite; physical
failover unverified.**

- Job/task models, deterministic gallery splitting, adaptive pull scheduling,
  compatibility filtering, fenced lease generations, three-lease attempt
  semantics, expiry, stale completion rejection, poison-tile avoidance, node
  disconnect requeue, and disk-backed PNG tiles exist under `nova/jobs.py`,
  `nova/scheduler.py`, `nova/store.py`, and `nova/coordinator.py`.
- Scheduler, lease, compatibility, coordinator-message, API PUT, and local
  failover tests are present under `tests/`.
- Result PUT now carries backend/device/execution metadata and `X-Nova-Node-Id`.
  `TASK_COMPLETE` is advisory: the coordinator only ACKs after a stored tile.
  Failed or ignored PUTs emit `TASK_FAILED`, not `TASK_COMPLETE`.

### Worker and kernels

Status: **Implemented; dummy path unit verified; Mac Metal and Linux CUDA
image-generation demo-verified; Windows CUDA still unverified.**

- Hardware probing, backend-specific device selection, dummy kernel, SD-Turbo
  kernel, local-files-only model loading, warmup, independent heartbeat/progress
  loops, and `asyncio.to_thread` inference exist in `nova/hardware.py`,
  `nova/kernels/`, and `nova/worker.py`.
- The dummy kernel and heartbeat-during-compute behavior have automated tests.
- The kernel now accepts either a complete Hugging Face snapshot under
  `models/sd-turbo/` or a lone `sd_turbo.safetensors`. Incomplete snapshots
  fall back to `from_single_file` and pass local `model_index.json` as
  `config` so conversion does not hit gated SD 2.1 repos.
- Mac Metal 512×512 image recorded 2026-09-19. Linux CUDA 512×512 image
  recorded 2026-09-19 on RunPod RTX 4000 Ada. Windows CUDA / ROCm still
  unverified.

### Control plane and discovery

Status: **Implemented for TCP/in-process paths; Pear and physical network paths
unverified.**

- Typed emergency TCP control, in-process simulation, protocol envelopes,
  disconnect handling, and the Python/Pear adapter exist in `nova/network/`.
- The Hyperswarm sidecar exists in `pear/bridge.js` and passes syntax checking.
- Local TCP disconnect tests exist, but phone-hotspot Hyperswarm discovery,
  direct typed-IP fallback between physical machines, graceful Ctrl+C latency
  under two seconds, reconnect, and venue-network behavior are not recorded as
  Demo verified.

### API, dashboard, CLI, and rehearsal

Status: **Implemented; local automated/smoke verification present; projector and
full rehearsal unverified.**

- FastAPI endpoints, WebSocket event stream, static dashboard with **Send 24-tile
  job**, CLI commands (`nova dashboard` / `nova worker`), systemd/launchd/Windows
  service units, deterministic 24-prompt gallery YAML, dummy kernel, and
  three-worker in-process simulation are present.
- API result validation covers empty bodies, garbage PNGs, SHA-256, stale lease
  generations, and first-winner behavior.
- Dashboard **tiles** control (1–96, default 24) dispatches that many prompts
  from `demo/gallery.yaml`. Custom prompt repeats N times with incrementing
  seeds. Gallery/Benchmark tabs, FP16 TFLOPS from warmup, ASAP vs per-node
  quota generation. Projector check still unverified.

## Highest-priority pending work

1. ~~Preserve backend/device/execution metadata atomically with the winning HTTP
   result PUT~~ **done locally** (2026-09-19).
2. ~~Prevent a failed PNG upload from being followed by an authoritative
   `TASK_COMPLETE`~~ **done locally** (2026-09-19).
3. ~~Fence HTTP uploads to the assigned node by carrying and validating
   `X-Nova-Node-Id`~~ **done locally** (2026-09-19).
4. ~~Run the one-machine dummy flow end to end~~ **done locally**
   (2026-09-19): `nova start --dummy --simulate-workers 3` then
   `nova run demo/gallery.yaml` completed 24/24 with CUDA 15 / ROCm 6 /
   Metal 3 contribution. Projector/browser check still unverified.
5. ~~Wire Pear/Hyperswarm into executable startup~~ **done locally**
   (2026-09-19): `nova start` / `nova worker` build a Pear+TCP `Hub` when
   `pear/node_modules/hyperswarm` exists; typed `NOVA_PEERS` /
   `NOVA_COORDINATOR_URL` still open TCP. Local in-process worker stays TCP
   only. Physical hotspot discovery still unverified.
6. ~~Make `NOVA_ROLE=worker` select worker startup as documented~~ **done
   locally** (2026-09-19); `--worker` still wins over env.
7. ~~Fail closed for real GPU startup when torch/diffusers or a CUDA/ROCm/Metal
   accelerator is missing~~ **done in unit tests** (2026-09-19). Mac Metal and
   Linux CUDA warmup are **Demo verified**. Windows CUDA remains unverified.
8. Exercise two real processes over TCP and measure Ctrl+C disconnect-to-requeue
   latency; prove it is under two seconds without waiting for heartbeat timeout.
9. Exercise Pear/Hyperswarm discovery on the intended hotspot and separately
   rehearse the typed `NOVA_COORDINATOR_URL` / `NOVA_PEERS` fallback.
10. ~~Download weights and generate one local 512×512 image on Mac Metal and
    Linux CUDA~~ **done** (2026-09-19). Still need the same proof on Windows
    CUDA. ROCm is opportunistic, not a judging blocker.
11. Run a mixed-worker 24-tile gallery, kill one worker near 40%, verify the job
    still finishes, and repeat the failover rehearsal ten times.
12. Record the backup dashboard session and full two-minute take.
13. ~~Make the repository reproducible~~ **done locally** (2026-09-19):
    implementation, tests, dashboard, pear sidecar, and docs committed;
    `models/`, `.nova/`, `starter/.tools/`, and `.claude/worktrees/` ignored.
14. Address the Ruff backlog after the contract-critical flow is green; do not
    mix broad style rewrites into failover or result-fencing work.
15. ~~Split one 512×512 image across nodes at the same time~~ **won't build**
    (2026-09-19). Gallery sharding (24 independent tiles) is the right split.
    DistriFusion/AsyncDiff/PipeFusion need NVLink, 25–50 steps, and ≥1024px.
    Our 4-step Metal 873 ms / CUDA 312 ms tiles lose to ngrok RTT. Naive
    patches seam. See journal below.

## Handoff rules

After every file edit, append one entry to the change journal with:

- file path;
- what changed and why;
- tests or runtime checks performed;
- remaining risk or follow-up.

When a check was not run, write **not run** rather than implying success. Preserve
unrelated working-tree changes; most of the current implementation is untracked
relative to the initial commit.

## Change journal

### 2026-09-19 — `NOVA_PROGRESS.md`

- Created the living progress and handoff ledger from the development contract,
  current source tree, and clean baseline test/syntax/smoke results.
- Recorded the pre-existing Ruff backlog separately from functional verification.
- Identified result metadata loss across HTTP PUT → `TASK_COMPLETE` as the next
  contract-aligned implementation slice.
- Verification: document reviewed against `NOVA_DEVELOPMENT_PLAN.md`; no source
  behavior changed by this file.

### 2026-09-19 — `nova/worker.py`

- Added `X-Nova-Backend`, `X-Nova-Device-Id`, and `X-Nova-Execution-Ms` to the
  result PUT so execution metadata reaches the same atomic boundary as PNG bytes,
  lease generation, and SHA-256.
- Kept the existing PUT-then-`TASK_COMPLETE` ordering unchanged; failed-upload
  completion remains a separate P0 follow-up recorded above.
- Verification: `python3 -m py_compile nova/worker.py` passed in the worker-agent
  check; direct source review confirmed all three values come from `KernelResult`.
- Remaining risk: the API must validate and consume the new headers before this
  changes persisted task metadata.

### 2026-09-19 — `tests/test_api_put.py`

- Added a regression test that uses the real `Store` and `Scheduler`, registers a
  compatible CUDA node, obtains a real lease through `on_work_request`, uploads a
  correctly sized PNG, and asserts atomic metadata persistence plus slot release.
- The test covers backend, device ID, execution time, assigned node, SHA-256, tile
  bytes, completed state, and `current_slots_used == 0`.
- Verification at file completion: focused test executed and failed only at the
  expected pre-fix assertion `backend_used == "cuda"`; request acceptance, tile
  persistence, and completion otherwise succeeded. This is the intended red
  regression state pending `nova/api.py` ingestion.
- Remaining risk: run the focused test again after the API file is complete, then
  run the full suite to ensure fake-scheduler compatibility remains intact.

### 2026-09-19 — `nova/api.py`

- Added parsing for `X-Nova-Backend`, `X-Nova-Device-Id`, and
  `X-Nova-Execution-Ms` on result PUTs.
- Rejects execution time unless it is a non-negative integer, and forwards all
  supported metadata through the existing real/fake scheduler adapter without
  requiring older fake schedulers to accept new keyword arguments.
- Preserved existing PNG/SHA validation, lease-generation fencing, and duplicate
  winner handling.
- Verification: `python -m py_compile nova/api.py` passed in the dashboard-agent
  check; direct review confirmed metadata is forwarded into `accept_result`.
- Remaining risk: the endpoint still has pre-existing duplicate persistence and
  permissive `TypeError` fallback behavior; those belong to the atomic-result P0
  follow-up.

### 2026-09-19 — metadata slice verification

- Focused command:
  `python -m pytest -q -p no:cacheprovider tests/test_api_put.py::test_put_real_scheduler_preserves_metadata_and_releases_slot`
  — **1 passed**, 1 third-party warning.
- Full command: `python -m pytest -q -p no:cacheprovider` — **79 passed**, 1
  third-party Starlette/httpx deprecation warning, in 1.99s.
- Result: atomic result PUT now preserves backend/device/execution metadata through
  the real API + Store + Scheduler path. Marked **integration verified locally**;
  real worker/GPU and physical network execution remain unverified.
- Next P0: prevent failed or ignored PUTs from emitting authoritative
  `TASK_COMPLETE` and completing a task without a stored tile.

### 2026-09-19 — fail-closed complete + node-id PUT fence

- `nova/worker.py` now sends `X-Nova-Node-Id` on the result PUT. Failed,
  ignored, malformed, or missing uploads still raise, so `_run_task`
  emits `TASK_FAILED` instead of `TASK_COMPLETE`.
- `nova/api.py` requires `X-Nova-Node-Id` and ignores PUTs whose uploader
  is not the assigned node, even when `lease_gen` matches.
- `nova/simulate.py` uses the same fail-closed PUT contract and forwards
  node identity; tests may inject an HTTP client so in-process rehearsal
  does not need a live coordinator HTTP server.
- Tests: `tests/test_worker_upload.py` covers accepted / missing URL /
  HTTP error / ignored PUT; `tests/test_api_put.py` covers missing and
  wrong node ids on fake and real scheduler paths; `tests/test_simulate.py`
  expects `TASK_FAILED` when `upload_url` is empty.
- Docs: README and the development-plan PUT contract now name
  `X-Nova-Node-Id`.
- Verification: focused PUT/worker/simulate/coordinator tests **51 passed**;
  full `python -m pytest -q -p no:cacheprovider` **88 passed**, 1 third-party
  warning, in 1.82s.
- Remaining risk: physical worker PUT + Ctrl+C failover still unverified.

### 2026-09-19 — `NOVA_ROLE=worker` and fail-closed GPU startup

- `nova start` honors `NOVA_ROLE=worker` as well as `--worker`; the CLI flag
  still wins. Missing GPU deps or a missing accelerator now fail the worker
  process instead of silently serving dummy tiles.
- `load_kernel("sd.t2i.v1")` raises unless torch and diffusers import;
  dummy is only used when requested (`--dummy`, `NOVA_DUMMY=1`, or kernel
  name `dummy`).
- Real workers require CUDA, ROCm, or Metal at warmup. CPU is dummy-only /
  last-resort, never advertised as an accelerator. Metal maps to `mps` and
  enables attention slicing plus `PYTORCH_ENABLE_MPS_FALLBACK`.
- README, `scripts/setup.sh`, and `scripts/setup.ps1` document Mac Metal,
  Windows CUDA, and Linux CUDA as the first-class runtimes.
- Tests: `tests/test_cli_role.py`, fail-closed kernel load, Metal/CUDA
  preference, CPU-only real worker rejection.
- Verification: focused role/kernel/hardware/worker/PUT/simulate tests
  **40 passed**; full `python -m pytest -q -p no:cacheprovider` **98 passed**,
  1 third-party warning, in 1.85s.
- This Mac has torch 2.14.0 with MPS available and no diffusers. Real Metal
  warmup still needs `pip install -e ".[gpu]"` and local sd-turbo weights.
  Windows/Linux CUDA wheels must come from pytorch.org, not a CPU default.

### 2026-09-19 — one-machine dummy rehearsal

- Started `NOVA_DATA_DIR=/tmp/nova-dummy-rehearsal nova start --dummy
  --simulate-workers 3`. Health/system showed three online workers with
  CUDA / ROCm / Metal badges. Submitted `demo/gallery.yaml`.
- Job `nova-gallery` reached **COMPLETED 24/24**. Contribution:
  `nova-sim-cuda-0` 15, `nova-sim-rocm-1` 6, `nova-sim-metal-2` 3.
  All 24 tiles decoded as 512×512 PNG; HTTP `GET /tiles/...png` served
  bytes; tasks stored `backend_used`, `device_id_used`, `execution_ms`,
  and `result_sha256`.
- Coordinator then stopped (exit 143 from SIGTERM). Projector/browser
  visual check **not run**. Physical GPU/Pear still unverified.

### 2026-09-19 — local `sd_turbo.safetensors` + Mac Metal image

- User downloaded `~/Downloads/sd_turbo.safetensors` (5.21 GB, original
  Stability single-file checkpoint: conditioner / diffusion_model /
  first_stage_model). That is not a Hugging Face snapshot.
- `nova/kernels/sd_t2i.py` now resolves a complete snapshot first, else a
  named/lone `.safetensors` file. Incomplete dirs with `model_index.json`
  but missing unet/vae weights no longer pretend to be loadable snapshots.
  Single-file loads use `StableDiffusionPipeline.from_single_file` with
  `local_files_only=True` and pass the local snapshot dir as `config`.
- `scripts/download_model.py` adopts `~/Downloads/sd_turbo.safetensors`
  into `models/sd-turbo/` and converts it to a local snapshot using those
  configs (avoids gated `stabilityai/stable-diffusion-2-1`).
- Installed GPU extras on this Mac: diffusers 0.40.0, transformers 5.17.0,
  accelerate 1.15.0, safetensors 0.8.0, torch 2.14.0, Python 3.13.13.
- Converted snapshot is ready (`model_index.json` + unet/vae/text_encoder
  weights). Direct `from_single_file` conversion without local configs
  failed 401 against gated SD 2.1.
- Real Metal generation on this Mac:
  - probe: `backend=metal`, `device_id=mps:0`, Apple Silicon
  - warmup (1 step): 5504 ms first call including load; `nova benchmark`
    later 740 ms, score 1.35
  - 4-step 512×512 tile after load: **873 ms**, 479020 PNG bytes
  - files: `/tmp/nova-metal-warmup.png`, `/tmp/nova-metal-tile.png`
  - process then aborted on MPS teardown (`recursive_mutex lock failed`);
    generation itself succeeded and PNGs decoded 512×512
- Tests: `tests/test_sd_t2i_load.py` **11 passed**; full suite
  `python -m pytest -q -p no:cacheprovider` **109 passed**, 3 third-party
  warnings, in 2.07s.
- Remaining: Windows CUDA and Linux CUDA still need `pip install -e ".[gpu]"`
  plus a CUDA torch wheel and one image each. Process-exit MPS mutex abort
  is a teardown issue, not a generation failure.

### 2026-09-19 — gitignore starter/.tools and Claude worktrees

- Ignored `starter/.tools/` (bun cache) and `.claude/worktrees/` so they
  cannot enter the first real commit.
- Source, tests, dashboard, pear sidecar, scripts, and docs remain tracked.

### 2026-09-19 — Linux NVIDIA kernel path (pre-pod image)

- CUDA/ROCm load now: prefer `variant=fp16` snapshot, disable safety checker,
  enable TF32 + cuDNN benchmark, channels-last on unet/vae, optional
  xformers, `torch.inference_mode()`, and `cuda.synchronize()` before timing.
- Metal still uses attention slicing; CUDA does not.
- `scripts/download_model.py` skips the 5GB original checkpoint and fp32
  twins so Linux boxes download the fp16 snapshot only.
- Setup docs: if the box already has CUDA torch (RunPod/NGC), install only
  `.[gpu]` extras and do not replace the wheel.
- Tests: CUDA snapshot load asserts fp16 variant, no slicing, xformers,
  channels-last, TF32. Full suite **109 passed**, 1.91s.
- RunPod target (not yet image-verified): RTX 4000 Ada 20GB, driver 595.91,
  CUDA 13.2 / nvcc 12.4, torch 2.4.1+cu124, Python 3.11.10, Ubuntu 22.04.
- First pod install pulled transformers 5.17 / diffusers 0.40, which refuse
  torch 2.4 (`PyTorch >= 2.5 required` + FlashAttn-3 custom-op schema error).
  GPU extras are now pinned `diffusers>=0.31,<0.35` and `transformers>=4.45,<5`.
  Kernel falls back to `StableDiffusionPipeline` if AutoPipeline import fails.

### 2026-09-19 — Linux CUDA image on RunPod RTX 4000 Ada

- Pod: `0jdlqbjdts2vnh`, Ubuntu 22.04, Python 3.11.10, driver 595.91,
  CUDA toolkit 12.4, torch **2.4.1+cu124** left in place.
- Repo at `/workspace/NOVA` (`f0b9699`). GPU extras after pin:
  diffusers **0.34.0**, transformers **4.57.6**. Import of
  `StableDiffusionPipeline` / `AutoPipelineForText2Image` succeeded.
- Probe: `backend=cuda`, `device_id=cuda:0`, NVIDIA RTX 4000 Ada
  Generation, 20019 MB.
- `python3 scripts/cuda_smoke.py`:
  - warmup (1 step, includes load): **1616 ms** execution, 13.63s wall,
    471260 PNG bytes, 512×512 RGB,
    sha256 `94e87e2584b09fb1e2826cb789990c8aa04f14b3bc81ad21b77c5418cbec01dc`
  - 4-step tile after load: **312 ms**, 524621 PNG bytes, 512×512 RGB,
    sha256 `86fc39c4d357759c41a2758f74b52c6454ff6126ae0cb6efe94d7204d7f8d2cc`
- `nova benchmark`: `cuda  latency_ms=1624  score=0.62` (cold-ish load;
  the 312 ms 4-step number is the live-tile figure).
- Files on pod: `/tmp/nova-cuda-warmup.png`, `/tmp/nova-cuda-tile.png`.
  RunPod SSH wrapper has no scp subsystem; hashes/sizes verified over TTY.

### 2026-09-19 — worker service + job dashboard service (v0.2.0)

- Split long-running processes: `nova dashboard` (HTTP API + live gallery, no
  in-process GPU worker) and `nova worker` (pull client + kernel). `nova start`
  still runs dashboard + local worker for the one-machine demo.
- Dashboard UI now dispatches work: **Send 24-tile job**, custom prompt, cancel.
  `POST /jobs/gallery` and `GET /gallery/default` wrap `demo/gallery.yaml`.
  Gallery jobs mint unique `nova-gallery-<hex>` ids so repeat clicks do not
  collide.
- Packaging: systemd user units, launchd agents, Windows scheduled-task
  installer under `packaging/` + `scripts/install-service.sh|.ps1`.
- Version bumped to 0.2.0 for the GitHub Release.
- Remaining: physical service install on Windows CUDA still unverified;
  GitHub Release cut after this commit.

### 2026-09-19 — bound per-tile GPU memory; keep Metal generating; shard on OOM

- `nova/kernels/sd_t2i.py` now caps peak activation memory: VAE slicing on every
  backend, attention slicing + VAE tiling on Metal, safety checker / feature
  extractor dropped. Weights stay resident. After each tile, MPS `empty_cache`
  runs so unified memory does not grow across the 24-tile job. CUDA keeps its
  allocator cache so NVIDIA stays fast.
- `output_type="pil"` and the pipeline output list are dropped before the next
  tile. CUDA/MPS synchronize after generate.
- OOM (`CUDA out of memory`, `MPS backend out of memory`) is re-raised as
  `accelerator out of memory on this node; tile will requeue`. The worker already
  emits `TASK_FAILED` (not `TASK_COMPLETE`); the scheduler poison-blacklists that
  node for one pull cycle so CUDA/ROCm/Metal siblings take the shard. Mac still
  generates; it does not have to finish the gallery alone.
- Tests: Metal load enables slice+tile; CUDA load enables VAE slice only;
  execute releases MPS cache and keeps the pipe; OOM is rewrapped; worker OOM
  path sends `TASK_FAILED` and frees the slot.
- Remaining: live Mac+CUDA 24-tile after worker restart not yet re-run.

### 2026-09-19 — intra-image same-time split: researched, not built

- Question: can Mac Metal + Linux CUDA denoise **one** 512×512 SD-Turbo image
  at the same time (patches, steps, or UNet stages)?
- Verdict: **no.** Keep 1 Task = 1 full image. Parallelism stays **inter-tile**
  (24 prompts, one exclusive lease each). OOM on Mac already requeues that
  whole tile to CUDA (`TASK_FAILED` + one-cycle poison blacklist).
- Why not: DistriFusion (CVPR 2024, arXiv:2402.19481) wins at 1024–3840, 25–50
  steps, NVLink A100s; naive patches seam (2-GPU PSNR 14.0 vs 24.6); authors
  say NVLink is essential and extremely-few-step samplers may not work.
  AsyncDiff (arXiv:2406.06911) is 50-step UNet stages on NVLink; lowest timed
  is 25 steps; poor interconnect “may not perform optimally.” PipeFusion /
  xDiT (arXiv:2405.14430) is DiT/Flux on homogeneous PCIe CUDA — Flux is
  contract-forbidden. STADI (arXiv:2509.04719) exists because mixed-speed
  GPUs idle the fast one under per-step barriers.
- Arithmetic on this mesh: CUDA 4-step 512 = 312 ms, Metal = 873 ms. Two
  ngrok barriers (~80 ms RTT) already add ~160 ms before activation bytes.
  SD-Turbo `guidance_scale=0` so there is no CFG split. Step-split is serial.
- No scheduler, protocol, kernel, or dashboard changes. Do not add a tensor
  data plane or multi-assignee lease. Reconsider only for same-vendor NVLink
  and multi-second high-res images; then use distrifuser/xDiT as a library,
  do not fork `adaptive_pull`.
- Verification: literature + existing demo timings; no new GPU run. Suite
  last recorded **120 passed** after the memory-bound work.

### 2026-09-19 — Pear/Hyperswarm wired into `nova start`

- `_make_control` now builds TCP plus `PearTransport` when
  `pear/node_modules/hyperswarm` exists, wrapped in `Hub`. Typed
  `NOVA_PEERS` / `NOVA_COORDINATOR_URL` remain the emergency path.
- Workers may join on Pear alone; they still need `NOVA_COORDINATOR_URL`
  for PNG PUT if discovery does not carry the HTTP base.
- Local in-process worker on `nova start` stays TCP-only so it does not
  spawn a second sidecar. `Hub.bound_port` proxies the TCP listen port.
- Tests: `tests/test_control_plane.py`. Physical hotspot still unverified.
