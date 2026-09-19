#!/usr/bin/env node
/**
 * Hyperswarm sidecar for NOVA.
 *
 * Python speaks JSON lines on stdin:
 *   {"op":"send","peer_id":"...","envelope":{...}}
 *   {"op":"broadcast","envelope":{...}}
 *   {"op":"stop"}
 *
 * This process speaks JSON lines on stdout:
 *   {"op":"ready","topic":"..."}
 *   {"op":"peer_connect","peer_id":"..."}
 *   {"op":"peer_disconnect","peer_id":"..."}
 *   {"op":"message","peer_id":"...","envelope":{...}}
 *   {"op":"error","error":"..."}
 *
 * Logs go to stderr only — stdout is the protocol.
 */
'use strict'

const crypto = require('crypto')
const readline = require('readline')

const NODE_ID = process.env.NOVA_NODE_ID || 'unknown'
const TOPIC_STR = process.env.NOVA_SWARM_TOPIC || 'nova-htn-2026'
const TOPIC = crypto.createHash('sha256').update(TOPIC_STR).digest()

function emit(obj) {
  process.stdout.write(JSON.stringify(obj) + '\n')
}

function log(...args) {
  process.stderr.write(args.join(' ') + '\n')
}

function helloLine() {
  return (
    JSON.stringify({
      type: 'HELLO',
      from_id: NODE_ID,
      payload: { protocol_version: 1, node_id: NODE_ID },
      ts: new Date().toISOString(),
      protocol_version: 1,
    }) + '\n'
  )
}

let Hyperswarm
try {
  Hyperswarm = require('hyperswarm')
} catch (err) {
  emit({
    op: 'error',
    error: 'hyperswarm not installed (cd pear && npm install); use TcpTransport / NOVA_PEERS',
  })
  process.exit(1)
}

const swarm = new Hyperswarm()
/** @type {Map<string, any>} */
const peers = new Map()

function writeNdjson(conn, envelope) {
  try {
    conn.write(JSON.stringify(envelope) + '\n')
  } catch (err) {
    log('write failed', err && err.message)
  }
}

function attach(conn, info) {
  const remoteKey = info && info.publicKey ? info.publicKey.toString('hex').slice(0, 16) : ''
  let peerId = 'tmp-' + (remoteKey || crypto.randomBytes(4).toString('hex'))
  peers.set(peerId, conn)
  emit({ op: 'peer_connect', peer_id: peerId })
  try {
    conn.write(helloLine())
  } catch (err) {
    log('hello write failed', err && err.message)
  }

  let buf = ''
  conn.on('data', (data) => {
    buf += data.toString('utf8')
    let idx
    while ((idx = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, idx)
      buf = buf.slice(idx + 1)
      if (!line.trim()) continue
      let env
      try {
        env = JSON.parse(line)
      } catch {
        continue
      }
      if (env && env.type === 'HELLO') {
        const payload = env.payload || {}
        const real = String(payload.node_id || env.from_id || '')
        if (real && real !== peerId) {
          const old = peerId
          const prev = peers.get(real)
          if (prev && prev !== conn) {
            try {
              prev.destroy()
            } catch {
              /* ignore */
            }
          }
          peers.delete(old)
          peerId = real
          peers.set(peerId, conn)
        }
      }
      emit({ op: 'message', peer_id: peerId, envelope: env })
    }
  })

  const gone = () => {
    if (peers.get(peerId) === conn) peers.delete(peerId)
    emit({ op: 'peer_disconnect', peer_id: peerId })
  }
  conn.on('close', gone)
  conn.on('error', () => {
    try {
      conn.destroy()
    } catch {
      /* ignore */
    }
  })
}

swarm.on('connection', attach)
swarm.join(TOPIC, { server: true, client: true })
emit({ op: 'ready', topic: TOPIC_STR })

const rl = readline.createInterface({ input: process.stdin })
rl.on('line', (line) => {
  if (!line.trim()) return
  let msg
  try {
    msg = JSON.parse(line)
  } catch {
    return
  }
  if (msg.op === 'send') {
    const conn = peers.get(msg.peer_id)
    if (conn && msg.envelope) writeNdjson(conn, msg.envelope)
  } else if (msg.op === 'broadcast') {
    if (!msg.envelope) return
    for (const conn of peers.values()) writeNdjson(conn, msg.envelope)
  } else if (msg.op === 'stop') {
    shutdown()
  }
})

async function shutdown() {
  try {
    await swarm.destroy()
  } catch {
    /* ignore */
  }
  process.exit(0)
}

process.on('SIGINT', shutdown)
process.on('SIGTERM', shutdown)
