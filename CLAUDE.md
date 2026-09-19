# NOVA

Hack the North 2026. Coordinator-led P2P rendering runtime. MVP: 24 SD-Turbo images across CUDA / ROCm / Metal.

**Tagline:** Three runtimes. One compute pool.

Read `NOVA_DEVELOPMENT_PLAN.md` before changing scheduler, leases, or failover. That file is the implementation contract.

## Topology

- Submitter = coordinator + scheduler + gallery store + HTTP API + dashboard
- Workers = model runtime + pull client
- Pear = discovery + control-plane messages
- HTTP = PNG bytes (data plane)

## Team tracks / subagents

| Agent | Owns |
|---|---|
| `pear-agent` | `pear/`, `nova/network/` (except `transport.py` ABC) |
| `worker-agent` | `nova/hardware.py`, `nova/kernels/`, `nova/worker.py` |
| `coordinator-agent` | `nova/jobs.py`, `nova/scheduler.py`, `nova/store.py`, scheduler tests |
| `dashboard-agent` | `nova/api.py`, `dashboard/`, `demo/` |

Do not invent video, Flux, credits, ledgers, or work stealing.

## Hard rules

- `pipe()` off the asyncio thread (`asyncio.to_thread`)
- Ctrl+C → NODE_DISCONNECTED in <2s (do not wait for heartbeat)
- Persistent `.nova/node.json`
- `attempt_count` increments on lease grant; `max_attempts=3` means three leases
- Stale `lease_gen` PUT/COMPLETE ignored
- Local weights only at demo (`local_files_only=True`)
