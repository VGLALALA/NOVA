# NOVA

**Three runtimes. One compute pool.**

Coordinator-led P2P rendering runtime for Hack the North 2026. One workload: 24 SD-Turbo images, pulled across CUDA / ROCm / Metal, filling a live gallery. Workers discover each other; the job owner coordinates the batch. There is no cloud GPU provider.

## What judges see

Three machines. Dashboard badges for the backends that are actually present. Open the job dashboard, click **Send 24-tile job**. Empty 4×6 grid. Tiles appear, tagged by vendor. Ctrl+C one worker → event feed shows OFFLINE, unfinished tiles requeue, the others finish the job.

## One-machine dummy rehearsal

No torch, no weights. Solid PNGs, same HTTP + dashboard path.

```bash
./scripts/setup.sh
source .venv/bin/activate
nova start --dummy --simulate-workers 3
```

The process prints a LAN URL. Open it:

```text
http://<lan-ip>:8080
```

In another terminal:

```bash
nova run demo/gallery.yaml
nova nodes
nova jobs
```

`--simulate-workers N` starts N in-process fake CUDA / ROCm / Metal workers that actually pull tiles and PUT dummy PNGs. Use it for the one-laptop rehearsal.

## Real GPUs

Mac Metal, Windows CUDA, and Linux CUDA are first-class. Real startup fails closed if torch/diffusers are missing or if the only device is CPU. Dummy images are never served under CUDA / ROCm / Metal badges.

Once per machine, with internet:

```bash
python -m pip install -e ".[gpu]"
# install the matching PyTorch wheel (see below) if `import torch` fails
python scripts/download_model.py
```

| Machine | PyTorch | What NOVA must probe |
|---|---|---|
| macOS Apple Silicon | official macOS arm64 wheel | `backend=metal`, `device_id=mps:0` |
| Windows NVIDIA | CUDA wheel from pytorch.org | `backend=cuda`, `device_id=cuda:0` |
| Linux NVIDIA | CUDA wheel from pytorch.org | `backend=cuda`, `device_id=cuda:0` |

Do not `pip install torch` from the default index on NVIDIA boxes — that often yields a CPU wheel. If the machine already has `torch` with `cuda=True` (RunPod, NGC, conda CUDA), install only `.[gpu]` extras and leave torch alone. Confirm:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())"
```

At demo time the worker loads **local files only**. Do not hit Hugging Face while judging.

A Hugging Face snapshot under `models/sd-turbo/` (with `model_index.json`) is preferred. A lone `sd_turbo.safetensors` in that directory, or `NOVA_MODEL_DIR` pointing at the file, also works — the kernel uses `from_single_file` for that path.

Jobs are already sharded: 24 independent 512×512 tiles, one lease per free slot. A Mac still generates on Metal (fp16, VAE slice/tile, attention slicing, MPS cache released after each tile). If a node OOMs, that tile fails closed and another worker pulls it — the Mac does not have to finish the gallery alone. Tight unified memory: run `nova dashboard` on the Mac and `nova worker` on the GPU boxes.

```bash
# every box (Mac, CUDA, ROCm) — same client
nova start

# join an existing cluster (typed fallback if Pear is missing)
NOVA_COORDINATOR_URL=http://192.168.x.x:8080 NOVA_PEERS=192.168.x.x:7946 nova start
```

On the dashboard: **Gallery** tab to dispatch, **Benchmark** tab for per-node FP16 TFLOPS (from warmup). Set **tiles**, then **finish ASAP** (adaptive pull) or **assign per node** (quota). Click **Send gallery**. `nova run demo/gallery.yaml` still works from a terminal.

Every machine is the same client: `nova start` = dashboard **and** local GPU worker. Send gallery on any node; tiles still pull from whoever is free. `CLUSTER_SYNC` ships the roster on join so a new node learns everyone and everyone learns the new node (`http_url` + `control_host:port`). `nova dashboard` / `nova worker` remain available if you want to split roles.

## Services

Install as a user service after `./scripts/setup.sh` (and `pip install -e ".[gpu]"` on workers):

```bash
# projector / submitter
./scripts/install-service.sh dashboard

# each GPU box (set NOVA_COORDINATOR_URL in .env first)
./scripts/install-service.sh worker
```

| OS | What gets installed |
|---|---|
| Linux | systemd user units `nova-dashboard` / `nova-worker` |
| macOS | launchd agents `com.nova.dashboard` / `com.nova.worker` |
| Windows | `.\scripts\install-service.ps1 dashboard\|worker` scheduled tasks |

Units live under `packaging/`. LAN-only HTTP, no auth — do not port-forward `:8080`.

## CLI

| Command | What it does |
|---|---|
| `nova dashboard` | Job dashboard service: HTTP API + live gallery, no local worker |
| `nova worker` | Worker service: pull tiles, PUT PNG results |
| `nova start` | Coordinator HTTP + Pear/TCP control, plus a local worker |
| `nova start --dashboard` | Same as `nova dashboard` |
| `nova start --worker` | Same as `nova worker` (also `NOVA_ROLE=worker`) |
| `nova start --dummy` | Dummy kernel, no torch |
| `nova start --simulate-workers N` | N in-process CUDA/ROCm/Metal dummy workers |
| `nova nodes` | GET `/nodes` |
| `nova jobs` | GET `/jobs` |
| `nova run demo/gallery.yaml` | POST `/jobs` |
| `nova benchmark` | Warmup score (`--dummy` prints a stub) |

`nova join` is not a command. Workers join with `nova worker`.

## HTTP (LAN-only, no auth)

Bind is 0.0.0.0:8080 by default. Do not port-forward this to the internet.

| Method | Path |
|---|---|
| GET | `/health` |
| GET | `/system` |
| GET | `/nodes` |
| GET | `/jobs` |
| POST | `/jobs` |
| POST | `/jobs/gallery` (dispatch `demo/gallery.yaml`) |
| GET | `/gallery/default` |
| GET | `/jobs/{job_id}` |
| POST | `/jobs/{job_id}/cancel` |
| GET | `/jobs/{job_id}/tasks` |
| GET | `/jobs/{job_id}/tiles/{task_id}.png` |
| GET | `/jobs/{job_id}/tasks/{task_id}/input.json` |
| PUT | `/jobs/{job_id}/tasks/{task_id}/result` (`X-Nova-Lease-Gen`, `X-Nova-Sha256`, `X-Nova-Node-Id`) |
| WS | `/ws/events` (replay history, then stream) |

Dashboard static files are served at `/`.

Result PUT: decode PNG, matching `lease_gen`, `X-Nova-Node-Id` equal to the assigned node, task in `LEASED`/`RUNNING`. First valid body wins; stale, wrong-node, or duplicate PUTs return `{"status":"ignored"}` and do not overwrite. A worker must not send `TASK_COMPLETE` unless that PUT returned `accepted`.

## Architecture

```text
Submitter  = coordinator + scheduler + gallery store + HTTP API + dashboard
Workers    = model runtime + pull client
Pear / TCP = discovery + control messages
HTTP       = PNG bytes (data plane)
```

Capability filter decides who *may* run a tile. Pull decides who *gets* the next one. Faster GPUs naturally take more work. Scheduling is not “weighted by benchmark.”

## Env

See `.env.example`. Useful ones:

```text
NOVA_ROLE=coordinator|worker
NOVA_HTTP_HOST=0.0.0.0
NOVA_HTTP_PORT=8080
NOVA_COORDINATOR_URL=http://192.168.x.x:8080   # emergency path if discovery fails
NOVA_PEERS=192.168.x.x:7946
NOVA_DATA_DIR=.nova
NOVA_DUMMY=0
NOVA_MODEL_DIR=models/sd-turbo
```

Identity is persistent in `.nova/node.json`. Random ids on restart are a demo bug.

## Backup

Live mesh will fail. Record the dashboard (`scripts/record_backup.sh`) and keep a dummy take.
