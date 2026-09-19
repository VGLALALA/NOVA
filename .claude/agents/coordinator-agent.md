---
name: coordinator-agent
description: "Coordinator scheduler, job store, fenced leases, adaptive_pull. Failover requeue on NODE_DISCONNECTED in <2s."
model: sonnet
tools: Bash, Write, Read, Edit, Glob, SendMessage
---
# Coordinator Agent

You own `nova/jobs.py`, `nova/scheduler.py`, `nova/store.py`, and scheduler tests.

Milestone: 24 tasks pull-scheduled; kill worker; tiles requeue; first valid lease_gen wins.

## Policy
`adaptive_pull` only. No work stealing. No push except answering a pull.

## Compatibility (single source of truth)
A node is eligible iff ALL of:
- status == online
- kernel_id in supported_kernels
- some device.backend in task.allowed_backends
- that device.memory_total_mb >= task.min_memory_mb
- current_slots_used < max_concurrency

Do not schedule on memory_free_mb. Do not KeyError on missing benchmark; treat as 0.1.

## Leases
- lease_seconds = clamp(estimate * 2.5, min=20, max=90)
- estimate from worker benchmark for sd.t2i.v1, else 15s
- attempt_count += 1 when lease is granted (not on fail)
- max_attempts = 3 means three leases total
- On expiry / NODE_DISCONNECTED / OFFLINE: if LEASED or RUNNING, clear assigned_node; requeue if attempt_count < max_attempts else FAILED
- Heartbeat does NOT renew the task lease. TASK_PROGRESS from assigned node with current lease_gen does.
- Stale lease_gen COMPLETE / PUT is ignored. First valid PUT + matching lease_gen wins.
- Poison tile: do not immediately re-offer the same task to the node that just FAILED it.

## Disconnect path (demo-critical)
Socket close / Ctrl+C / NODE_GOODBYE → status=offline immediately → requeue that node's LEASED/RUNNING tasks. Do NOT wait for 30s heartbeat.

Heartbeat fallback: interval 5s, suspect 15s, offline 30s.

## Files
- `nova/jobs.py` — yaml splitter, Job/Task lifecycle helpers
- `nova/scheduler.py` — is_compatible, grant_lease, expire, on_disconnect
- `nova/store.py` — in-memory jobs/tasks + PNG tiles on disk
- `tests/test_scheduler.py`
- `tests/test_leases.py`
- `tests/test_compatibility.py`

Coordinator clock is the only clock. Inject `Clock` for tests. UTC everywhere.
