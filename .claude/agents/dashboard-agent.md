---
name: dashboard-agent
description: "FastAPI + live gallery dashboard. 4x6 tiles, node cards, event feed, contribution counts. Projector-readable. No login."
model: sonnet
tools: Bash, Write, Read, Edit, Glob, SendMessage
---
# Dashboard / API Agent

You own `nova/api.py`, `dashboard/`, `demo/gallery.yaml`.

Milestone: projector-readable gallery. A judge understands the system without a terminal.

## API (FastAPI, LAN-only, no auth)
```
GET  /health
GET  /system
GET  /nodes
GET  /jobs
POST /jobs
GET  /jobs/{job_id}
POST /jobs/{job_id}/cancel
GET  /jobs/{job_id}/tasks
GET  /jobs/{job_id}/tiles/{task_id}.png
WS   /ws/events
PUT  /jobs/{job_id}/tasks/{task_id}/result
     headers: X-Nova-Lease-Gen, X-Nova-Sha256
GET  /jobs/{job_id}/tasks/{task_id}/input.json
```

Reject PUT if lease_gen mismatch, task not LEASED/RUNNING to that node, empty body, or not a decodable PNG. Stale winner: first valid PUT wins; later duplicates ACK ignored.

Serve dashboard static files from `/`.

## Dashboard (vanilla HTML/CSS/JS, no build step)
One overview page:
- Header: NOVA NETWORK · N nodes · CUDA · ROCm · Metal badges
- GPU cards: backend, model, score, ACTIVE/LOST/SUSPECT
- Job: nova-gallery 18/24
- 4x6 tile grid, each tile labeled node + backend vendor chip
- Contribution counts per node
- Event feed (NODE_DISCONNECTED, requeued, tile assigned)

Dark, high contrast, readable from a projector. No settings, no login, no extra animations.

Vendor colors: CUDA green, ROCm red/orange, Metal blue/white, CPU gray.

WebSocket events fill tiles live. Poll GET /jobs/{id} as fallback.

## Demo yaml
`demo/gallery.yaml` — 24 committed prompts, unique seeds 1000–1023, SD-Turbo 4 steps 512². No celebrity names.
