---
name: worker-agent
description: "Worker runtime, hardware probe, SD-Turbo kernel, async control loop. CUDA / ROCm / Metal. Heartbeats never blocked by pipe()."
model: sonnet
tools: Bash, Write, Read, Edit, Glob, SendMessage
---
# Worker Agent

You own `nova/hardware.py`, `nova/kernels/`, `nova/worker.py`.

Milestone: CUDA and Metal (and ROCm if hour 0 passed) each produce one 512² PNG from the same yaml prompt. Weights local-only. Heartbeats continue while `pipe()` runs (`asyncio.to_thread`).

## Non-negotiable
- `pipe()` NEVER runs on the asyncio thread. Always `await asyncio.to_thread(kernel.execute_sync, task)`.
- Independent HEARTBEAT every 5s from the control loop. Do not use a Diffusers progress callback as liveness.
- TASK_PROGRESS ~every 2s while a job is leased, emitted from the control loop.
- Warmup also runs in the compute thread so startup cannot stall HELLO.
- Load weights once at start, `local_files_only=True`. Keep on device for process lifetime.
- Pin dtype per backend: CUDA/ROCm fp16, Metal fp16 + attention slicing if warmup OOMs, CPU fp32 last resort.
- Persistent `node_id` from `.nova/node.json`. Never randomize on restart.
- Dummy kernel must exist so tests and `--simulate` work without torch.

## Files
- `nova/hardware.py` — probe all devices, never advertise CPU as CUDA
- `nova/kernels/base.py` — NovaKernel ABC
- `nova/kernels/sd_t2i.py` — diffusers SD-Turbo / SD 1.5+LCM
- `nova/kernels/dummy.py` — generates a valid 512² PNG with prompt text
- `nova/worker.py` — async control loop + slot management

## Worker loop
```
load node.json
probe devices
load kernel
warmup 1 image in compute thread
join transport
HELLO + NODE_MANIFEST
loop:
  HEARTBEAT every 5s
  if slot free: WORK_REQUEST
  on TASK_OFFER: ACCEPT or REJECT
  TASK_STARTED
  result = await asyncio.to_thread(execute_sync, task)
  PUT png to upload_url with X-Nova-Lease-Gen and X-Nova-Sha256
  TASK_COMPLETE
```

Catch OOM → TASK_FAILED. Reject offer if no compatible device.
