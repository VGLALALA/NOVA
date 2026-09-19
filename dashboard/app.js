"use strict";

const BACKEND_META = {
  cuda: { label: "CUDA", color: "#3dd68c" },
  rocm: { label: "ROCm", color: "#ff6b4a" },
  metal: { label: "Metal", color: "#7cb8ff" },
  cpu: { label: "CPU", color: "#888888" },
};
const BACKEND_ORDER = ["cuda", "rocm", "metal", "cpu"];
const TILE_COUNT = 24;

const els = {
  nodeLine: document.getElementById("node-line"),
  badges: document.getElementById("badges"),
  link: document.getElementById("link-state"),
  nodeCards: document.getElementById("node-cards"),
  jobTitle: document.getElementById("job-title"),
  jobProgress: document.getElementById("job-progress"),
  tiles: document.getElementById("tiles"),
  contrib: document.getElementById("contrib"),
  feed: document.getElementById("feed"),
  sendGallery: document.getElementById("send-gallery"),
  sendCustom: document.getElementById("send-custom"),
  cancelJob: document.getElementById("cancel-job"),
  seed: document.getElementById("seed"),
  prompt: document.getElementById("custom-prompt"),
  dispatchStatus: document.getElementById("dispatch-status"),
};

let pollOnly = true;
let lastJobId = null;
const seenEvents = new Set();

function asList(data) {
  if (Array.isArray(data)) return data;
  if (data && Array.isArray(data.jobs)) return data.jobs;
  if (data && Array.isArray(data.nodes)) return data.nodes;
  if (data && Array.isArray(data.tasks)) return data.tasks;
  return [];
}

async function getJSON(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} ${res.status}`);
  return res.json();
}

async function postJSON(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const text = await res.text();
  let data = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch (_err) {
      data = { detail: text };
    }
  }
  if (!res.ok) {
    const detail = data.detail || data.message || text || res.status;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function setDispatch(text) {
  if (els.dispatchStatus) els.dispatchStatus.textContent = text;
}

function setBusy(busy) {
  [els.sendGallery, els.sendCustom, els.cancelJob].forEach((btn) => {
    if (btn) btn.disabled = busy;
  });
}

function setLink(mode) {
  pollOnly = mode !== "live";
  els.link.textContent = mode === "live" ? "LIVE" : "POLL";
  els.link.className = "link " + (mode === "live" ? "live" : "poll");
}

function primaryDevice(node) {
  const devices = node.devices || [];
  return devices[0] || {};
}

function backendOf(node) {
  return node.primary_backend || primaryDevice(node).backend || "cpu";
}

function renderHeader(system, nodes) {
  const n = (system && typeof system.nodes === "number") ? system.nodes : nodes.length;
  const backends = (system && system.backends) || [];
  const present = new Set(backends.length ? backends : nodes.map(backendOf));
  els.nodeLine.textContent = n === 1 ? "1 node" : `${n} nodes`;
  els.badges.replaceChildren();
  BACKEND_ORDER.forEach((key) => {
    if (!present.has(key)) return;
    const b = document.createElement("span");
    b.className = `badge ${key}`;
    b.textContent = BACKEND_META[key].label;
    els.badges.appendChild(b);
  });
}

function renderNodes(nodes) {
  els.nodeCards.replaceChildren();
  if (!nodes.length) {
    const p = document.createElement("p");
    p.className = "empty-hint";
    p.textContent = "No workers yet.";
    els.nodeCards.appendChild(p);
    return;
  }
  nodes.forEach((node) => {
    const backend = backendOf(node);
    const status = node.display_status || ({
      online: "ACTIVE",
      suspect: "SUSPECT",
      offline: "LOST",
    }[node.status] || "LOST");
    const card = document.createElement("article");
    card.className = `node-card ${backend}${status === "LOST" ? " lost" : ""}`;

    const model = document.createElement("div");
    model.className = "node-model";
    model.textContent = node.primary_model || primaryDevice(node).model || node.node_id;

    const sub = document.createElement("div");
    sub.className = "node-sub";
    const mem = primaryDevice(node).memory_total_mb;
    const unified = backend === "metal" ? " · unified" : "";
    const memS = mem ? ` · ${mem} MB${unified}` : "";
    sub.textContent = `${node.node_id} · ${(BACKEND_META[backend] || BACKEND_META.cpu).label}${memS}`;

    const row = document.createElement("div");
    row.className = "node-status-row";
    const score = document.createElement("div");
    score.className = "node-score";
    const raw = node.score;
    score.textContent = typeof raw === "number" ? `score ${raw.toFixed(2)}` : "score —";
    const st = document.createElement("div");
    st.className = `status ${status}`;
    st.textContent = status;
    row.append(score, st);

    card.append(model, sub, row);
    els.nodeCards.appendChild(card);
  });
}

function ensureTiles() {
  if (els.tiles.childElementCount === TILE_COUNT) return;
  els.tiles.replaceChildren();
  for (let i = 0; i < TILE_COUNT; i += 1) {
    const tile = document.createElement("div");
    tile.className = "tile";
    tile.dataset.shard = String(i);
    const n = document.createElement("span");
    n.className = "n";
    n.textContent = String(i);
    tile.appendChild(n);
    els.tiles.appendChild(tile);
  }
}

function fillTile(jobId, task) {
  const shard = Number(task.shard_index);
  const tile = els.tiles.children[shard];
  if (!tile) return;
  const backend = task.backend_used || "cpu";
  tile.className = `tile done ${backend}`;
  let img = tile.querySelector("img");
  const src = `/jobs/${encodeURIComponent(jobId)}/tiles/${encodeURIComponent(task.task_id)}.png?h=${task.result_sha256 || Date.now()}`;
  if (!img) {
    img = document.createElement("img");
    img.alt = `tile ${shard}`;
    tile.appendChild(img);
  }
  if (img.getAttribute("src") !== src) img.src = src;
  let chip = tile.querySelector(".chip");
  if (!chip) {
    chip = document.createElement("div");
    chip.className = "chip";
    tile.appendChild(chip);
  }
  chip.replaceChildren();
  const v = document.createElement("span");
  v.className = `vendor ${backend}`;
  v.textContent = (BACKEND_META[backend] || BACKEND_META.cpu).label;
  const id = document.createElement("span");
  id.textContent = task.assigned_node || "";
  chip.append(v, id);
}

function clearTile(shard) {
  const tile = els.tiles.children[shard];
  if (!tile) return;
  tile.className = "tile";
  tile.querySelector("img")?.remove();
  tile.querySelector(".chip")?.remove();
}

function nodeLookup(nodes) {
  const map = new Map();
  nodes.forEach((n) => map.set(n.node_id, n));
  return map;
}

function renderJob(job, tasks, nodes) {
  ensureTiles();
  const name = (job && (job.name || job.job_id)) || "—";
  const total = (job && job.total) || tasks.length || TILE_COUNT;
  const done = (job && job.completed) || tasks.filter((t) => t.state === "COMPLETED").length;
  els.jobTitle.textContent = `JOB ${name}`;
  els.jobProgress.textContent = `${done}/${total}`;

  const byShard = new Map();
  tasks.forEach((t) => byShard.set(Number(t.shard_index), t));
  for (let i = 0; i < TILE_COUNT; i += 1) {
    const t = byShard.get(i);
    if (t && t.state === "COMPLETED") {
      fillTile(job.job_id, t);
    } else {
      clearTile(i);
    }
  }

  const counts = new Map();
  tasks.forEach((t) => {
    if (t.state !== "COMPLETED" || !t.assigned_node) return;
    const rec = counts.get(t.assigned_node) || { count: 0, backend: t.backend_used, model: t.assigned_node };
    rec.count += 1;
    rec.backend = t.backend_used || rec.backend;
    counts.set(t.assigned_node, rec);
  });
  const lookup = nodeLookup(nodes);
  counts.forEach((rec, nid) => {
    const node = lookup.get(nid);
    if (node) {
      rec.model = node.primary_model || primaryDevice(node).model || nid;
      rec.backend = rec.backend || backendOf(node);
    }
  });

  const rows = Array.from(counts.values()).sort((a, b) => b.count - a.count);
  els.contrib.replaceChildren();
  rows.forEach((row) => {
    const el = document.createElement("div");
    el.className = "contrib-row";
    const sw = document.createElement("span");
    sw.className = `swatch ${row.backend || "cpu"}`;
    const nameEl = document.createElement("span");
    nameEl.textContent = row.model;
    const n = document.createElement("span");
    n.className = "contrib-count";
    n.textContent = String(row.count);
    el.append(sw, nameEl, n);
    els.contrib.appendChild(el);
  });
}

function formatTime(ts) {
  if (!ts) return "";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return String(ts).slice(11, 19);
  return d.toISOString().slice(11, 19);
}

function formatEvent(ev) {
  const type = ev.type || "";
  const node = ev.node_id || ev.node || "";
  const shard = ev.shard_index;
  const tile = shard === 0 || shard ? `tile ${shard}` : (ev.task_id || "");
  if (type === "NODE_DISCONNECTED" || type === "NODE_OFFLINE" || type === "NODE_GOODBYE") {
    return { text: `${node}  OFFLINE`, cls: "offline" };
  }
  if (type === "NODE_RECOVERED") return { text: `${node}  RECOVERED`, cls: "" };
  if (type === "TILE_COMPLETE") return { text: `${tile}  →  ${node}`, cls: "" };
  if (type === "TILE_REQUEUED" || type === "TASK_REQUEUED") return { text: `requeued ${tile}`, cls: "" };
  if (type === "JOB_STARTED") return { text: `job ${ev.job_id || ev.name || ""} started`, cls: "" };
  if (type === "JOB_COMPLETED") return { text: `job ${ev.job_id || ""} complete`, cls: "" };
  if (type === "JOB_CANCELLED") return { text: `job ${ev.job_id || ""} cancelled`, cls: "" };
  if (type === "TASK_FAILED") return { text: `${tile} failed`, cls: "offline" };
  const rest = Object.keys(ev)
    .filter((k) => k !== "type" && k !== "ts")
    .map((k) => `${k}=${ev[k]}`)
    .join(" ");
  return { text: `${type}  ${rest}`.trim(), cls: "" };
}

function pushEvent(ev) {
  const key = `${ev.ts}|${ev.type}|${ev.task_id || ""}|${ev.node_id || ""}|${ev.shard_index || ""}`;
  if (seenEvents.has(key)) return;
  seenEvents.add(key);
  const li = document.createElement("li");
  const ts = document.createElement("span");
  ts.className = "ts";
  ts.textContent = formatTime(ev.ts);
  const body = document.createElement("span");
  const fmt = formatEvent(ev);
  body.textContent = fmt.text;
  if (fmt.cls) body.className = fmt.cls;
  li.append(ts, body);
  els.feed.prepend(li);
  while (els.feed.childElementCount > 80) {
    els.feed.removeChild(els.feed.lastChild);
  }
}

async function refresh() {
  try {
    const [system, nodesRaw, jobsRaw] = await Promise.all([
      getJSON("/system").catch(() => ({})),
      getJSON("/nodes").catch(() => []),
      getJSON("/jobs").catch(() => []),
    ]);
    const nodes = asList(nodesRaw);
    const jobs = asList(jobsRaw);
    renderHeader(system, nodes);
    renderNodes(nodes);
    const online = nodes.filter((n) => (n.status || "") === "online").length;
    if (!lastJobId && els.dispatchStatus && !els.dispatchStatus.dataset.locked) {
      setDispatch(online ? `${online} worker${online === 1 ? "" : "s"} ready` : "waiting for workers");
    }
    const job = jobs[0] || null;
    if (!job) {
      els.jobTitle.textContent = "JOB —";
      els.jobProgress.textContent = "0/24";
      ensureTiles();
      els.contrib.replaceChildren();
      lastJobId = null;
      return;
    }
    lastJobId = job.job_id;
    let tasks = [];
    try {
      tasks = asList(await getJSON(`/jobs/${encodeURIComponent(job.job_id)}/tasks`));
    } catch (_err) {
      tasks = [];
    }
    renderJob(job, tasks, nodes);
  } catch (_err) {
    // keep last frame
  }
}

function connectEvents() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  let ws;
  try {
    ws = new WebSocket(`${proto}://${location.host}/ws/events`);
  } catch (_err) {
    setLink("poll");
    return;
  }
  ws.onopen = () => setLink("live");
  ws.onmessage = (msg) => {
    let ev;
    try {
      ev = JSON.parse(msg.data);
    } catch (_err) {
      return;
    }
    pushEvent(ev);
    const interesting = new Set([
      "TILE_COMPLETE",
      "TILE_REQUEUED",
      "TASK_REQUEUED",
      "NODE_DISCONNECTED",
      "NODE_OFFLINE",
      "NODE_GOODBYE",
      "NODE_RECOVERED",
      "NODE_MANIFEST",
      "JOB_STARTED",
      "JOB_COMPLETED",
      "JOB_CANCELLED",
      "TASK_FAILED",
    ]);
    if (interesting.has(ev.type)) refresh();
    if (ev.type === "TILE_COMPLETE" && ev.job_id && ev.task_id) {
      fillTile(ev.job_id, ev);
    }
  };
  ws.onclose = () => {
    setLink("poll");
    setTimeout(connectEvents, 1500);
  };
  ws.onerror = () => {
    try { ws.close(); } catch (_err) { /* ignore */ }
  };
}

async function sendGallery() {
  setBusy(true);
  setDispatch("sending gallery…");
  try {
    const job = await postJSON("/jobs/gallery", {});
    lastJobId = job.job_id;
    setDispatch(`sent ${job.job_id}  (${job.total || 24} tiles)`);
    await refresh();
  } catch (err) {
    setDispatch(`send failed: ${err.message || err}`);
  } finally {
    setBusy(false);
  }
}

async function sendCustom() {
  const text = (els.prompt && els.prompt.value || "").trim();
  if (!text) {
    setDispatch("enter a prompt, or use Send 24-tile job");
    return;
  }
  const seed = Number((els.seed && els.seed.value) || 1000);
  setBusy(true);
  setDispatch("sending custom job…");
  try {
    const job = await postJSON("/jobs", {
      job: { name: "nova-custom", type: "sd_gallery", kernel: "sd.t2i.v1" },
      model: { id: "stabilityai/sd-turbo", steps: 4, width: 512, height: 512 },
      requirements: { min_memory_mb: 4096, allowed_backends: ["cuda", "rocm", "metal"] },
      prompts: [{ text, seed: Number.isFinite(seed) ? seed : 1000 }],
    });
    lastJobId = job.job_id;
    setDispatch(`sent ${job.job_id}  (${job.total || 1} tile)`);
    await refresh();
  } catch (err) {
    setDispatch(`send failed: ${err.message || err}`);
  } finally {
    setBusy(false);
  }
}

async function cancelCurrent() {
  if (!lastJobId) {
    setDispatch("no job to cancel");
    return;
  }
  setBusy(true);
  setDispatch(`cancelling ${lastJobId}…`);
  try {
    await postJSON(`/jobs/${encodeURIComponent(lastJobId)}/cancel`, {});
    setDispatch(`cancelled ${lastJobId}`);
    await refresh();
  } catch (err) {
    setDispatch(`cancel failed: ${err.message || err}`);
  } finally {
    setBusy(false);
  }
}

if (els.sendGallery) els.sendGallery.addEventListener("click", sendGallery);
if (els.sendCustom) els.sendCustom.addEventListener("click", sendCustom);
if (els.cancelJob) els.cancelJob.addEventListener("click", cancelCurrent);

ensureTiles();
setLink("poll");
refresh();
connectEvents();
setInterval(refresh, 1000);
