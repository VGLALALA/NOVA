"use strict";

const BACKEND_META = {
  cuda: { label: "CUDA", color: "#3dd68c" },
  rocm: { label: "ROCm", color: "#ff6b4a" },
  metal: { label: "Metal", color: "#7cb8ff" },
  cpu: { label: "CPU", color: "#888888" },
};
const BACKEND_ORDER = ["cuda", "rocm", "metal", "cpu"];
const TILE_COUNT_DEFAULT = 24;
const TILE_COUNT_MAX = 96;

const els = {
  nodeLine: document.getElementById("node-line"),
  badges: document.getElementById("badges"),
  link: document.getElementById("link-state"),
  nodeCards: document.getElementById("node-cards"),
  jobTitle: document.getElementById("job-title"),
  jobProgress: document.getElementById("job-progress"),
  jobElapsed: document.getElementById("job-elapsed"),
  tiles: document.getElementById("tiles"),
  contrib: document.getElementById("contrib"),
  feed: document.getElementById("feed"),
  sendGallery: document.getElementById("send-gallery"),
  sendCustom: document.getElementById("send-custom"),
  cancelJob: document.getElementById("cancel-job"),
  seed: document.getElementById("seed"),
  batch: document.getElementById("batch"),
  prompt: document.getElementById("custom-prompt"),
  dispatchStatus: document.getElementById("dispatch-status"),
  tabGallery: document.getElementById("tab-gallery"),
  tabBench: document.getElementById("tab-bench"),
  tabCoords: document.getElementById("tab-coords"),
  tabWorkers: document.getElementById("tab-workers"),
  viewGallery: document.getElementById("view-gallery"),
  viewBench: document.getElementById("view-bench"),
  viewCoords: document.getElementById("view-coords"),
  viewWorkers: document.getElementById("view-workers"),
  coordCards: document.getElementById("coord-cards"),
  coordBody: document.getElementById("coord-body"),
  tcpTarget: document.getElementById("tcp-target"),
  addTcp: document.getElementById("add-tcp"),
  pingNodes: document.getElementById("ping-nodes"),
  nodeCtlStatus: document.getElementById("node-ctl-status"),
  nodeCtlBody: document.getElementById("node-ctl-body"),
  nodeCtlCards: document.getElementById("node-ctl-cards"),
  modeAsap: document.getElementById("mode-asap"),
  modeQuota: document.getElementById("mode-quota"),
  quotaRow: document.getElementById("quota-row"),
  benchPool: document.getElementById("bench-pool"),
  benchFast: document.getElementById("bench-fast"),
  benchOnline: document.getElementById("bench-online"),
  benchBars: document.getElementById("bench-bars"),
  benchBody: document.getElementById("bench-body"),
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

function isWorker(node) {
  return typeof node.warmup_ms === "number" || !node.http_url;
}

function isCoordinator(node) {
  return Boolean(node.http_url) && typeof node.warmup_ms !== "number";
}

function splitRoster(nodes) {
  const workers = nodes.filter(isWorker);
  const coords = nodes.filter(isCoordinator);
  return { workers, coords };
}

function renderHeader(system, nodes) {
  const { workers, coords } = splitRoster(nodes);
  const backends = (system && system.backends) || [];
  const present = new Set(backends.length ? backends : workers.map(backendOf));
  const w = workers.length;
  const c = coords.length;
  els.nodeLine.textContent = `${w} worker${w === 1 ? "" : "s"} · ${c} coordinator${c === 1 ? "" : "s"}`;
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
    const tflops = document.createElement("div");
    tflops.className = "node-score tflops";
    const tf = node.fp16_tflops;
    tflops.textContent = typeof tf === "number" ? `${tf.toFixed(1)} TFLOPS fp16` : "— TFLOPS";
    const seen = document.createElement("div");
    seen.className = "node-score";
    seen.textContent = node.last_seen_s == null ? "no ping" : `${node.last_seen_s.toFixed(0)}s ago`;
    row.append(score, tflops, seen, st);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "btn danger node-remove";
    remove.textContent = "Remove";
    remove.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      dropNode(node.node_id);
    });

    card.append(model, sub, row, remove);
    els.nodeCards.appendChild(card);
  });
}

function batchCount() {
  const n = Number((els.batch && els.batch.value) || TILE_COUNT_DEFAULT);
  if (!Number.isFinite(n)) return TILE_COUNT_DEFAULT;
  return Math.min(TILE_COUNT_MAX, Math.max(1, Math.floor(n)));
}

function gridColumns(count) {
  if (count <= 4) return count;
  if (count <= 12) return 4;
  if (count <= 24) return 6;
  if (count <= 48) return 8;
  return 8;
}

function ensureTiles(count) {
  const n = count || TILE_COUNT_DEFAULT;
  els.tiles.style.gridTemplateColumns = `repeat(${gridColumns(n)}, 1fr)`;
  if (els.tiles.childElementCount === n) return;
  els.tiles.replaceChildren();
  for (let i = 0; i < n; i += 1) {
    const tile = document.createElement("div");
    tile.className = "tile";
    tile.dataset.shard = String(i);
    const label = document.createElement("span");
    label.className = "n";
    label.textContent = String(i);
    tile.appendChild(label);
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
  const name = (job && (job.name || job.job_id)) || "—";
  const total = (job && job.total) || tasks.length || batchCount();
  ensureTiles(total);
  const done = (job && job.completed) || tasks.filter((t) => t.state === "COMPLETED").length;
  els.jobTitle.textContent = `JOB ${name}`;
  els.jobProgress.textContent = `${done}/${total}`;
  if (els.jobElapsed) {
    const wall = job && job.elapsed_ms;
    const infer = job && job.inference_ms;
    const wallS = typeof wall === "number" ? `${(wall / 1000).toFixed(1)}s wall` : "—";
    const inferS = typeof infer === "number" ? `${(infer / 1000).toFixed(1)}s infer` : "";
    els.jobElapsed.textContent = inferS ? `${wallS} · ${inferS}` : wallS;
  }

  const byShard = new Map();
  tasks.forEach((t) => byShard.set(Number(t.shard_index), t));
  for (let i = 0; i < total; i += 1) {
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
    const { workers, coords } = splitRoster(nodes);
    renderHeader(system, nodes);
    renderNodes(workers);
    renderBench(workers);
    renderNodeControl(workers);
    renderCoordinators(coords);
    syncQuotaInputs(workers);
    const online = workers.filter((n) => (n.status || "") === "online").length;
    if (!lastJobId && els.dispatchStatus && !els.dispatchStatus.dataset.locked) {
      setDispatch(online ? `${online} worker${online === 1 ? "" : "s"} ready` : "waiting for workers");
    }
    const job = jobs[0] || null;
    if (!job) {
      els.jobTitle.textContent = "JOB —";
      els.jobProgress.textContent = `0/${batchCount()}`;
      ensureTiles(batchCount());
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
    const n = batchCount();
    const job = await postJSON("/jobs/gallery", galleryBody(n));
    lastJobId = job.job_id;
    setDispatch(`sent ${job.job_id}  (${job.total || n} tiles)`);
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
    setDispatch("enter a prompt, or use Send gallery");
    return;
  }
  const seed = Number((els.seed && els.seed.value) || 1000);
  const n = batchCount();
  const base = Number.isFinite(seed) ? seed : 1000;
  const prompts = [];
  for (let i = 0; i < n; i += 1) prompts.push({ text, seed: base + i });
  setBusy(true);
  setDispatch("sending custom job…");
  try {
    const job = await postJSON("/jobs", {
      job: {
        name: "nova-custom",
        type: "sd_gallery",
        kernel: "sd.t2i.v1",
        scheduler_policy: genPolicy(),
        quotas: genQuotas(),
      },
      model: { id: "stabilityai/sd-turbo", steps: 4, width: 512, height: 512 },
      requirements: { min_memory_mb: 4096, allowed_backends: ["cuda", "rocm", "metal"] },
      prompts,
    });
    lastJobId = job.job_id;
    setDispatch(`sent ${job.job_id}  (${job.total || n} tiles)`);
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

function showView(name) {
  const views = {
    gallery: els.viewGallery,
    bench: els.viewBench,
    coords: els.viewCoords,
    workers: els.viewWorkers,
  };
  const tabs = {
    gallery: els.tabGallery,
    bench: els.tabBench,
    coords: els.tabCoords,
    workers: els.tabWorkers,
  };
  Object.keys(views).forEach((key) => {
    const on = key === name;
    const view = views[key];
    const tab = tabs[key];
    if (view) {
      view.classList.toggle("hidden", !on);
      view.hidden = !on;
    }
    if (tab) {
      tab.classList.toggle("active", on);
      tab.setAttribute("aria-selected", on ? "true" : "false");
    }
  });
}

function renderCoordinators(nodes) {
  if (els.coordBody) {
    els.coordBody.replaceChildren();
    if (!nodes.length) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 4;
      td.textContent = "No coordinators advertised.";
      tr.appendChild(td);
      els.coordBody.appendChild(tr);
    } else {
      nodes.forEach((node) => {
        const tr = document.createElement("tr");
        const status = node.display_status || node.status || "LOST";
        [node.node_id, node.http_url || "—", status, node.last_seen_s == null ? "never" : `${node.last_seen_s.toFixed(0)}s ago`].forEach((text) => {
          const td = document.createElement("td");
          td.textContent = text;
          tr.appendChild(td);
        });
        els.coordBody.appendChild(tr);
      });
    }
  }
  if (els.coordCards) {
    els.coordCards.replaceChildren();
    nodes.forEach((node) => {
      const card = document.createElement("article");
      card.className = `node-card ${backendOf(node)}`;
      const model = document.createElement("div");
      model.className = "node-model";
      model.textContent = node.node_id;
      const sub = document.createElement("div");
      sub.className = "node-sub";
      sub.textContent = node.http_url || "no http_url";
      const st = document.createElement("div");
      st.className = `status ${node.display_status || ({ online: "ACTIVE", suspect: "SUSPECT", offline: "LOST" }[node.status] || "LOST")}`;
      st.textContent = node.display_status || ({ online: "ACTIVE", suspect: "SUSPECT", offline: "LOST" }[node.status] || "LOST");
      card.append(model, sub, st);
      els.coordCards.appendChild(card);
    });
  }
}

function renderNodeControl(nodes) {
  if (els.nodeCtlBody) {
    els.nodeCtlBody.replaceChildren();
    nodes.forEach((node) => {
      const tr = document.createElement("tr");
      const status = node.display_status || node.status || "LOST";
      const backend = backendOf(node);
      tr.innerHTML = "";
      const cells = [
        node.node_id,
        (BACKEND_META[backend] || BACKEND_META.cpu).label,
        status,
        node.last_seen_s == null ? "never" : `${node.last_seen_s.toFixed(0)}s ago`,
      ];
      cells.forEach((text) => {
        const td = document.createElement("td");
        td.textContent = text;
        tr.appendChild(td);
      });
      const td = document.createElement("td");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn danger";
      btn.textContent = "Remove";
      btn.addEventListener("click", () => dropNode(node.node_id));
      td.appendChild(btn);
      tr.appendChild(td);
      els.nodeCtlBody.appendChild(tr);
    });
  }
  if (els.nodeCtlCards) {
    els.nodeCtlCards.replaceChildren();
    nodes.forEach((node) => {
      const card = document.createElement("article");
      card.className = `node-card ${backendOf(node)}`;
      card.textContent = `${node.node_id} · ${node.display_status || node.status}`;
      els.nodeCtlCards.appendChild(card);
    });
  }
}

async function addTcpNode() {
  const raw = (els.tcpTarget && els.tcpTarget.value || "").trim();
  if (!raw) {
    if (els.nodeCtlStatus) els.nodeCtlStatus.textContent = "enter host:port";
    return;
  }
  if (els.nodeCtlStatus) els.nodeCtlStatus.textContent = `dialing ${raw}…`;
  try {
    const data = await postJSON("/nodes/tcp", { host: raw });
    if (els.nodeCtlStatus) {
      els.nodeCtlStatus.textContent = `dialing ${data.host}:${data.port}`;
    }
    await refresh();
  } catch (err) {
    if (els.nodeCtlStatus) els.nodeCtlStatus.textContent = `add failed: ${err.message || err}`;
  }
}

async function dropNode(nodeId) {
  const note = `removing ${nodeId}…`;
  if (els.nodeCtlStatus) els.nodeCtlStatus.textContent = note;
  setDispatch(note);
  try {
    await postJSON(`/nodes/${encodeURIComponent(nodeId)}/remove`, {});
    const msg = `removed ${nodeId}`;
    if (els.nodeCtlStatus) els.nodeCtlStatus.textContent = msg;
    setDispatch(msg);
    await refresh();
  } catch (err) {
    const fail = `remove failed: ${err.message || err}`;
    if (els.nodeCtlStatus) els.nodeCtlStatus.textContent = fail;
    setDispatch(fail);
  }
}

async function pingNow() {
  try {
    await postJSON("/nodes/ping", {});
    if (els.nodeCtlStatus) els.nodeCtlStatus.textContent = "pinged workers";
    await refresh();
  } catch (err) {
    if (els.nodeCtlStatus) els.nodeCtlStatus.textContent = `ping failed: ${err.message || err}`;
  }
}

function genPolicy() {
  return els.modeQuota && els.modeQuota.checked ? "quota" : "adaptive_pull";
}

function genQuotas() {
  if (genPolicy() !== "quota" || !els.quotaRow) return {};
  const out = {};
  els.quotaRow.querySelectorAll("input[data-node]").forEach((input) => {
    const n = Number(input.value);
    if (Number.isFinite(n) && n > 0) out[input.dataset.node] = Math.floor(n);
  });
  return out;
}

function galleryBody(n) {
  const body = { count: n };
  if (genPolicy() === "quota") {
    body.job = { scheduler_policy: "quota", quotas: genQuotas() };
  }
  return body;
}

function syncQuotaInputs(nodes) {
  if (!els.quotaRow) return;
  const quotaOn = genPolicy() === "quota";
  els.quotaRow.classList.toggle("hidden", !quotaOn);
  if (!quotaOn) return;
  const online = nodes.filter((n) => (n.status || "") === "online");
  const existing = new Map();
  els.quotaRow.querySelectorAll("input[data-node]").forEach((input) => {
    existing.set(input.dataset.node, input.value);
  });
  const ids = online.map((n) => n.node_id);
  const same = ids.length === existing.size && ids.every((id) => existing.has(id));
  if (same) return;
  const nTiles = batchCount();
  const share = online.length ? Math.floor(nTiles / online.length) : 0;
  let rem = nTiles - share * online.length;
  els.quotaRow.replaceChildren();
  online.forEach((node, i) => {
    const card = document.createElement("label");
    card.className = "quota-card";
    const name = document.createElement("span");
    name.textContent = node.primary_model || node.node_id;
    const input = document.createElement("input");
    input.type = "number";
    input.min = "0";
    input.step = "1";
    input.dataset.node = node.node_id;
    const preset = existing.has(node.node_id)
      ? existing.get(node.node_id)
      : String(share + (i < rem ? 1 : 0));
    input.value = preset;
    card.append(name, input);
    els.quotaRow.appendChild(card);
  });
}

function renderBench(nodes) {
  if (!els.benchBody) return;
  const rows = nodes.map((n) => ({
    id: n.node_id,
    model: n.primary_model || n.node_id,
    backend: n.primary_backend || "cpu",
    status: n.status,
    score: typeof n.score === "number" ? n.score : null,
    warmup: typeof n.warmup_ms === "number" ? n.warmup_ms : null,
    generate: typeof n.generate_ms === "number" ? n.generate_ms : (typeof n.score === "number" && n.score > 0 ? 1000 / n.score : null),
    tflops: typeof n.fp16_tflops === "number" ? n.fp16_tflops : 0,
  })).sort((a, b) => (b.score || 0) - (a.score || 0));
  const online = rows.filter((r) => r.status === "online");
  const pool = online.reduce((s, r) => s + (r.tflops || 0), 0);
  const fastest = online.reduce((m, r) => Math.max(m, r.tflops || 0), 0);
  if (els.benchPool) els.benchPool.textContent = pool ? pool.toFixed(1) : "—";
  if (els.benchFast) els.benchFast.textContent = fastest ? fastest.toFixed(1) : "—";
  if (els.benchOnline) els.benchOnline.textContent = String(online.length);
  const maxT = Math.max(fastest, 0.01);
  if (els.benchBars) {
    els.benchBars.replaceChildren();
    rows.slice().sort((a, b) => b.tflops - a.tflops).forEach((r) => {
      const row = document.createElement("div");
      row.className = "bench-bar";
      const lab = document.createElement("div");
      lab.textContent = r.model;
      const track = document.createElement("div");
      track.className = "track";
      const fill = document.createElement("div");
      fill.className = `fill ${r.backend}`;
      fill.style.width = `${Math.min(100, Math.max(2, (r.tflops / maxT) * 100))}%`;
      fill.title = `${r.tflops.toFixed(2)} FP16 TFLOPS`;
      track.appendChild(fill);
      const val = document.createElement("div");
      val.className = "tflops";
      val.textContent = r.tflops ? `${r.tflops.toFixed(2)}` : "—";
      row.append(lab, track, val);
      els.benchBars.appendChild(row);
    });
  }
  els.benchBody.replaceChildren();
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    const cells = [
      r.id,
      (BACKEND_META[r.backend] || BACKEND_META.cpu).label,
      r.warmup ? `${Math.round(r.warmup)} ms` : "—",
      r.generate ? `${Math.round(r.generate)} ms` : "—",
      r.score != null ? r.score.toFixed(2) : "—",
      r.tflops ? r.tflops.toFixed(2) : "—",
    ];
    cells.forEach((text) => {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
    });
    els.benchBody.appendChild(tr);
  });
}

if (els.sendGallery) els.sendGallery.addEventListener("click", sendGallery);
if (els.sendCustom) els.sendCustom.addEventListener("click", sendCustom);
if (els.cancelJob) els.cancelJob.addEventListener("click", cancelCurrent);
if (els.tabGallery) els.tabGallery.addEventListener("click", () => showView("gallery"));
if (els.tabBench) els.tabBench.addEventListener("click", () => showView("bench"));
if (els.tabCoords) els.tabCoords.addEventListener("click", () => showView("coords"));
if (els.tabWorkers) els.tabWorkers.addEventListener("click", () => showView("workers"));
if (els.addTcp) els.addTcp.addEventListener("click", addTcpNode);
if (els.pingNodes) els.pingNodes.addEventListener("click", pingNow);
if (els.tcpTarget) {
  els.tcpTarget.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") addTcpNode();
  });
}
if (els.modeAsap) els.modeAsap.addEventListener("change", () => refresh());
if (els.modeQuota) els.modeQuota.addEventListener("change", () => refresh());
if (els.batch) {
  els.batch.addEventListener("change", () => {
    if (!lastJobId) {
      ensureTiles(batchCount());
      els.jobProgress.textContent = `0/${batchCount()}`;
    }
    refresh();
  });
}

ensureTiles(batchCount());
setLink("poll");
refresh();
connectEvents();
setInterval(refresh, 1000);
