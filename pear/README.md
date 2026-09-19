# NOVA Pear sidecar

Hyperswarm discovery + ndjson control-plane messages. PNG bytes stay on HTTP.

Pear is **optional**. Tests and the emergency demo path use TCP (`NOVA_PEERS`, port 7946). Do not block judging on DHT.

## Install

```bash
cd pear
npm install
```

Requires Node 18+.

## Run (usually spawned by Python)

```bash
export NOVA_SWARM_TOPIC=nova-htn-2026
export NOVA_NODE_ID=nova-7f4a9c2e
node bridge.js
```

- Topic string is hashed with SHA-256 (32-byte Hyperswarm topic).
- Join as server + client.
- JSON lines on stdin from Python, JSON lines on stdout to Python.
- Logs go to stderr only.

## Python

```python
from nova.network.pear import PearTransport, PearUnavailable
from nova.network.tcp import TcpTransport
from nova.network.hub import Hub

if PearTransport.available():
    pear = PearTransport(node_id=node_id, swarm_topic=settings.swarm_topic)
else:
    pear = None  # use TCP; do not fail the demo
```

If `node` or `hyperswarm` is missing, `PearTransport.start()` raises `PearUnavailable` telling you to use `TcpTransport` / `NOVA_PEERS`.

## Emergency path (must work even if this folder is unused)

Coordinator:

```python
TcpTransport(
    listen_host=settings.control_host,  # 0.0.0.0
    listen_port=settings.control_port,  # 7946
    connect_addrs=None,
    node_id=node_id,
)
```

Worker:

```python
TcpTransport(
    listen_host=None,
    listen_port=None,
    connect_addrs=settings.peer_list(),  # from NOVA_PEERS=192.168.x.x:7946
    node_id=node_id,
)
```

Both paths speak the same newline-delimited JSON `Envelope`. Socket close / Ctrl+C fires `on_peer_disconnect` immediately (the <2s failover path).
