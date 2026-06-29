#!/usr/bin/env python3
"""
memdag_dashboard.py — visual telemetry dashboard for the Claude memory store.

Reads the LIVE forgetting state from memdag's forget_* columns (~/.memdag/
memdag.db — the bridge's store; the legacy flat mem_weights.db is frozen since
its reconcile was disabled at the 2026-06-24 cutover), the live MEMORY.md index,
and (optionally) the episodic sessions archive, computes telemetry, and emits a
single self-contained HTML file with interactive charts — then opens it.

Read-only. Never writes to the memory store. Cross-platform (Mac + Windows).

Usage:
    python mem_dashboard.py            # build + open in browser
    python mem_dashboard.py --no-open  # build only, print the path
    python mem_dashboard.py --out PATH # custom output file
"""
import argparse
import json
import os
import platform
import re
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from memdag_bridge_import import default_memory_dir, split_frontmatter, fm_top_level

HOME = Path.home()


def _sessions_db():
    # Optional episodic session archive (separate project). Absent -> the
    # session-count card is simply omitted.
    return HOME / ".claude" / "episodic" / "sessions.db"
def _memdag_db():
    # Live forgetting state = memdag's forget_* columns. Resolved at call time
    # so $MEMDAG_DB (or a test override) is honored.
    return Path(os.environ.get("MEMDAG_DB") or HOME / ".memdag" / "memdag.db")

# Forgetting-layer thresholds (from mem_weights.py PARAMS) — keep in sync.
DEMOTE_BELOW = 0.2   # hot -> cold when RS (accessibility) drops under this
PROMOTE_AT = 0.5     # cold -> hot hysteresis
MEMORY_BUDGET = 16384  # bytes; /saveall + /audit enforce this cap on MEMORY.md

TYPE_PREFIXES = ("user", "feedback", "project", "personal", "reference")


def parse_iso(s: str):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_weights() -> list[dict]:
    """Live forgetting telemetry from memdag's forget_* columns (the bridge's
    store). Returns the same row shape the rest of the dashboard expects
    (stem/weight/count/last_used/first_seen/tier/pinned) so nothing downstream
    changes. `weight` = forget_rs (RS/accessibility); pinned = endorsed channel
    (user_/feedback_/personal_) or an explicit frontmatter pin."""
    db = _memdag_db()
    if not db.exists():
        raise SystemExit(f"memdag DB not found: {db}")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT source_ref, content, channel, forget_rs, forget_count, "
        "forget_last_used, forget_first_seen, forget_tier FROM nodes "
        "WHERE tombstoned = 0 AND source_ref LIKE 'memory:%' "
        "AND source_ref NOT LIKE 'memory:literal:%'"
    ).fetchall()
    con.close()
    out = []
    for sref, content, channel, rs, cnt, lused, fseen, tier in rows:
        out.append({
            "stem": sref.split(":", 1)[1],
            "weight": float(rs) if rs is not None else 1.0,
            "count": int(cnt or 0),
            "last_used": lused,
            "first_seen": fseen,
            "tier": tier or "hot",
            "pinned": 1 if (channel == "endorsed" or str(fm_top_level(split_frontmatter(content or "")[0]).get("pin", "")).strip().lower() in ("1", "true", "yes")) else 0,
        })
    return out


def stem_type(stem: str) -> str:
    head = stem.split("_", 1)[0]
    return head if head in TYPE_PREFIXES else "other"


def session_count():
    sdb = _sessions_db()
    if not sdb.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{sdb}?mode=ro", uri=True)
        # find a plausible sessions table
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        for name in ("sessions", "session", "transcripts", "chunks"):
            if name in tables:
                n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                con.close()
                return {"table": name, "count": n}
        con.close()
    except sqlite3.Error:
        return None
    return None


def build_graph(mem_dir, rows):
    """Relationship graph: MEMORY.md sections are parent hubs, memories are their
    siblings (tree edges), and [[wikilinks]] in memory bodies are cross-links."""
    if not mem_dir:
        return {"nodes": [], "links": [], "sections": []}
    mf = mem_dir / "MEMORY.md"
    if not mf.exists():
        return {"nodes": [], "links": [], "sections": []}

    link_re = re.compile(r"\[[^\]]*\]\(([a-z0-9_]+)\.md\)")
    section_of = {}   # stem -> section name
    order = []        # section display order
    cur = None
    for line in mf.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
            if cur not in order:
                order.append(cur)
        elif cur:
            m = link_re.search(line)
            if m:
                section_of.setdefault(m.group(1), cur)

    DEMO = "(demoted)"
    nodes, links = [], []
    have_demo = False
    for s in order:
        nodes.append({"id": "§" + s, "label": s, "kind": "section", "section": s})

    for r in rows:
        stem = r["stem"]
        sec = section_of.get(stem, DEMO)
        if sec == DEMO and not have_demo:
            have_demo = True
            order.append(DEMO)
            nodes.append({"id": "§" + DEMO, "label": DEMO,
                          "kind": "section", "section": DEMO})
        nodes.append({"id": stem, "label": stem, "kind": "memory",
                      "section": sec, "type": stem_type(stem),
                      "count": int(r["count"]), "tier": r["tier"],
                      "pinned": int(r["pinned"])})
        links.append({"source": "§" + sec, "target": stem, "kind": "tree"})

    nodeset = {n["id"] for n in nodes}
    wl_re = re.compile(r"\[\[([a-z0-9_-]+)\]\]")
    seen = set()
    for p in mem_dir.glob("*.md"):
        if p.name == "MEMORY.md" or p.stem not in nodeset:
            continue
        body = p.read_text(encoding="utf-8", errors="ignore")
        for target in wl_re.findall(body):
            if target in nodeset and target != p.stem:
                key = tuple(sorted((p.stem, target)))
                if key not in seen:
                    seen.add(key)
                    links.append({"source": p.stem, "target": target, "kind": "link"})

    return {"nodes": nodes, "links": links, "sections": order}


def build_telemetry():
    rows = load_weights()
    now = datetime.now(timezone.utc)

    total = len(rows)
    hot = sum(1 for r in rows if r["tier"] == "hot")
    cold = total - hot
    pinned = sum(1 for r in rows if r["pinned"])

    # type breakdown
    type_counts = Counter(stem_type(r["stem"]) for r in rows)

    # weight histogram (0..1 in 10 buckets)
    buckets = [0] * 10
    for r in rows:
        w = max(0.0, min(1.0, float(r["weight"])))
        idx = min(9, int(w * 10))
        buckets[idx] += 1
    hist_labels = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]

    # top accessed
    top = sorted(rows, key=lambda r: r["count"], reverse=True)[:15]
    top_access = [{"stem": r["stem"], "count": r["count"], "tier": r["tier"]} for r in top]

    # scatter: weight vs count, by tier
    scatter = [{"x": float(r["weight"]), "y": int(r["count"]),
                "stem": r["stem"], "tier": r["tier"], "pinned": int(r["pinned"])}
               for r in rows]

    # growth timeline: cumulative by first_seen date
    by_date = defaultdict(int)
    for r in rows:
        d = parse_iso(r["first_seen"])
        if d:
            by_date[d.date().isoformat()] += 1
    growth = []
    run = 0
    for day in sorted(by_date):
        run += by_date[day]
        growth.append({"date": day, "cumulative": run})

    # stalest / demote-risk: not pinned, sorted by weight asc then oldest last_used
    risk = sorted(
        [r for r in rows if not r["pinned"]],
        key=lambda r: (float(r["weight"]), r["last_used"] or "")
    )[:12]
    def age_days(r):
        d = parse_iso(r["last_used"])
        return (now - d).days if d else None
    stale = [{"stem": r["stem"], "weight": round(float(r["weight"]), 3),
              "count": r["count"], "tier": r["tier"],
              "age_days": age_days(r)} for r in risk]

    # MEMORY.md budget
    mem_dir = default_memory_dir()
    budget = None
    if mem_dir:
        mf = mem_dir / "MEMORY.md"
        if mf.exists():
            size = mf.stat().st_size
            budget = {"bytes": size, "cap": MEMORY_BUDGET,
                      "pct": round(100 * size / MEMORY_BUDGET, 1)}

    return {
        "generated": now.strftime("%Y-%m-%d %H:%M UTC"),
        "totals": {"total": total, "hot": hot, "cold": cold, "pinned": pinned},
        "tier": {"hot": hot, "cold": cold},
        "types": dict(type_counts),
        "hist": {"labels": hist_labels, "data": buckets},
        "top_access": top_access,
        "scatter": scatter,
        "growth": growth,
        "stale": stale,
        "budget": budget,
        "sessions": session_count(),
        "thresholds": {"demote_below": DEMOTE_BELOW, "promote_at": PROMOTE_AT},
        "graph": build_graph(mem_dir, rows),
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Memory Telemetry</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<style>
  :root {
    --bg:#0c0e14; --panel:#141823; --panel2:#1b2030; --line:#262d40;
    --ink:#e7ecf5; --dim:#8a93a8; --hot:#3ddc97; --cold:#f0883e;
    --accent:#6aa9ff; --pin:#c792ea; --grid:#1f2536; --danger:#ff5c72;
  }
  * { box-sizing:border-box; }
  body {
    margin:0; background:
      radial-gradient(900px 500px at 88% -8%, #16243a 0%, transparent 60%),
      var(--bg);
    color:var(--ink);
    font:14px/1.5 ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
    padding:28px clamp(16px,4vw,56px) 64px;
  }
  header { display:flex; align-items:baseline; gap:16px; flex-wrap:wrap;
    border-bottom:1px solid var(--line); padding-bottom:18px; margin-bottom:26px; }
  h1 { margin:0; font-size:22px; letter-spacing:.5px; font-weight:700; }
  h1 .tag { color:var(--accent); }
  .meta { color:var(--dim); font-size:12px; margin-left:auto; }
  .cards { display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    margin-bottom:26px; }
  .card { background:linear-gradient(180deg,var(--panel2),var(--panel));
    border:1px solid var(--line); border-radius:14px; padding:16px 18px; position:relative;
    overflow:hidden; }
  .card .k { color:var(--dim); font-size:11px; text-transform:uppercase; letter-spacing:1px; }
  .card .v { font-size:30px; font-weight:700; margin-top:6px; }
  .card .sub { font-size:11px; color:var(--dim); margin-top:2px; }
  .card.hotc .v { color:var(--hot); }
  .card.coldc .v { color:var(--cold); }
  .card.pinc .v { color:var(--pin); }
  .bar { height:6px; border-radius:6px; background:var(--grid); margin-top:12px; overflow:hidden; }
  .bar > i { display:block; height:100%; border-radius:6px; }
  .grid { display:grid; gap:18px; grid-template-columns:repeat(12,1fr); }
  .box { background:var(--panel); border:1px solid var(--line); border-radius:16px;
    padding:18px 20px; min-height:60px; }
  .box h2 { margin:0 0 14px; font-size:13px; font-weight:600; color:var(--ink);
    letter-spacing:.4px; display:flex; align-items:center; gap:8px; }
  .box h2 small { color:var(--dim); font-weight:400; font-size:11px; }
  .span4 { grid-column:span 4; } .span6 { grid-column:span 6; }
  .span8 { grid-column:span 8; } .span12 { grid-column:span 12; }
  canvas { max-height:300px; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th,td { text-align:left; padding:7px 8px; border-bottom:1px solid var(--grid); }
  th { color:var(--dim); font-weight:500; text-transform:uppercase; font-size:10px; letter-spacing:.6px; }
  td.stem { color:var(--ink); }
  .pill { display:inline-block; padding:1px 8px; border-radius:20px; font-size:10px; font-weight:600; }
  .pill.hot { background:rgba(61,220,151,.14); color:var(--hot); }
  .pill.cold { background:rgba(240,136,62,.16); color:var(--cold); }
  .w { font-variant-numeric:tabular-nums; }
  .legend { color:var(--dim); font-size:11px; margin-top:8px; }
  @media(max-width:900px){ .span4,.span6,.span8{grid-column:span 12;} }
</style>
</head>
<body>
<header>
  <h1>memory<span class="tag">·</span>telemetry</h1>
  <div class="meta" id="meta"></div>
</header>
<div class="cards" id="cards"></div>
<div class="grid">
  <div class="box span4"><h2>Tier split <small>hot = in MEMORY.md</small></h2><canvas id="tier"></canvas></div>
  <div class="box span4"><h2>By type</h2><canvas id="types"></canvas></div>
  <div class="box span4"><h2>Accessibility (RS) distribution <small>demote &lt; 0.2</small></h2><canvas id="hist"></canvas></div>
  <div class="box span8"><h2>RS vs access count <small>each dot = one memory · line = demote floor</small></h2><canvas id="scatter"></canvas></div>
  <div class="box span4"><h2>Demote-risk watchlist <small>unpinned, lowest RS</small></h2>
    <table id="stale"><thead><tr><th>memory</th><th>RS</th><th>uses</th><th>idle</th></tr></thead><tbody></tbody></table>
  </div>
  <div class="box span8"><h2>Store growth <small>cumulative memories by first-seen</small></h2><canvas id="growth"></canvas></div>
  <div class="box span4"><h2>Most-accessed</h2><canvas id="top"></canvas></div>
  <div class="box span12">
    <h2>Relationship graph
      <small>§ = section (parent) · dot = memory (sibling, sized by uses) · line = [[wikilink]]</small>
      <label style="margin-left:auto;font-weight:400;font-size:11px;color:var(--dim);cursor:pointer;user-select:none">
        <input type="checkbox" id="wlToggle" checked> show wikilinks</label>
    </h2>
    <div id="graph" style="width:100%;height:600px;position:relative"></div>
    <div class="legend" id="graphLegend" style="display:flex;flex-wrap:wrap;gap:14px;margin-top:12px"></div>
  </div>
</div>
<div id="tip" style="position:fixed;pointer-events:none;opacity:0;background:var(--panel2);
  border:1px solid var(--line);border-radius:8px;padding:7px 10px;font-size:11px;
  color:var(--ink);z-index:99;max-width:280px;transition:opacity .1s"></div>
<script>
const D = __DATA__;
const C = getComputedStyle(document.documentElement);
const col = n => C.getPropertyValue(n).trim();
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]);
Chart.defaults.color = col('--dim');
Chart.defaults.font.family = "ui-monospace, monospace";
Chart.defaults.borderColor = col('--grid');

document.getElementById('meta').textContent =
  `generated ${D.generated}  ·  ${D.totals.total} memories tracked`;

// stat cards
const cards = [
  {k:'Total memories', v:D.totals.total, cls:'', sub:'tracked in forgetting layer'},
  {k:'Hot (loaded)', v:D.totals.hot, cls:'hotc', sub:'always in context'},
  {k:'Cold (demoted)', v:D.totals.cold, cls:'coldc', sub:'recall-only'},
  {k:'Pinned', v:D.totals.pinned, cls:'pinc', sub:'never auto-forget'},
];
if (D.budget) cards.push({k:'MEMORY.md budget', v:D.budget.pct+'%',
  sub:`${D.budget.bytes} / ${D.budget.cap} B`, barpct:D.budget.pct,
  barcol: D.budget.pct>90?col('--danger'):D.budget.pct>75?col('--cold'):col('--hot')});
if (D.sessions) cards.push({k:'Episodic sessions', v:D.sessions.count, sub:'archived transcripts'});
document.getElementById('cards').innerHTML = cards.map(c => `
  <div class="card ${c.cls||''}">
    <div class="k">${c.k}</div><div class="v">${c.v}</div>
    <div class="sub">${c.sub||''}</div>
    ${c.barpct!==undefined?`<div class="bar"><i style="width:${Math.min(100,c.barpct)}%;background:${c.barcol}"></i></div>`:''}
  </div>`).join('');

const grid = {grid:{color:col('--grid')}, ticks:{color:col('--dim')}};

new Chart(tier, {type:'doughnut', data:{labels:['hot','cold'],
  datasets:[{data:[D.tier.hot,D.tier.cold],
    backgroundColor:[col('--hot'),col('--cold')], borderColor:col('--panel'), borderWidth:3}]},
  options:{plugins:{legend:{position:'bottom'}}, cutout:'62%'}});

const tnames = Object.keys(D.types);
new Chart(types, {type:'bar', data:{labels:tnames,
  datasets:[{data:tnames.map(t=>D.types[t]), backgroundColor:col('--accent'),
    borderRadius:6}]},
  options:{plugins:{legend:{display:false}}, scales:{x:grid,y:{...grid,beginAtZero:true}}}});

new Chart(hist, {type:'bar', data:{labels:D.hist.labels,
  datasets:[{data:D.hist.data, backgroundColor:D.hist.labels.map((_,i)=>
    i<2?col('--cold'):col('--hot')), borderRadius:4}]},
  options:{plugins:{legend:{display:false}}, scales:{x:grid,y:{...grid,beginAtZero:true}}}});

new Chart(scatter, {type:'scatter', data:{datasets:[
  {label:'hot', data:D.scatter.filter(p=>p.tier==='hot').map(p=>({x:p.x,y:p.y,stem:p.stem})),
   backgroundColor:col('--hot')},
  {label:'cold', data:D.scatter.filter(p=>p.tier==='cold').map(p=>({x:p.x,y:p.y,stem:p.stem})),
   backgroundColor:col('--cold')},
]}, options:{
  scales:{x:{...grid,title:{display:true,text:'accessibility (RS)'},min:0,max:1.02},
          y:{...grid,title:{display:true,text:'access count'},beginAtZero:true}},
  plugins:{legend:{position:'bottom'},
    tooltip:{callbacks:{label:c=>`${c.raw.stem}  (RS ${c.raw.x.toFixed(2)}, ${c.raw.y} uses)`}},
    annotation:false}}});
// manual demote-line via a thin dataset
(function(){
  const ch = Chart.getChart(scatter);
  const max = Math.max(1, ...D.scatter.map(p=>p.y));
  ch.data.datasets.push({type:'line', label:'demote floor (0.2)',
    data:[{x:D.thresholds.demote_below,y:0},{x:D.thresholds.demote_below,y:max}],
    borderColor:col('--danger'), borderDash:[5,4], borderWidth:1.5,
    pointRadius:0, fill:false});
  ch.update();
})();

const stb = document.querySelector('#stale tbody');
stb.innerHTML = D.stale.map(r=>`<tr>
  <td class="stem">${esc(r.stem)}</td>
  <td class="w">${r.weight}</td>
  <td class="w">${r.count}</td>
  <td class="w">${r.age_days==null?'—':r.age_days+'d'}</td></tr>`).join('')
  || `<tr><td colspan="4" style="color:var(--dim)">nothing at risk — all memories pinned or healthy</td></tr>`;

new Chart(growth, {type:'line', data:{labels:D.growth.map(g=>g.date),
  datasets:[{data:D.growth.map(g=>g.cumulative), borderColor:col('--accent'),
    backgroundColor:'rgba(106,169,255,.12)', fill:true, tension:.25, pointRadius:2}]},
  options:{plugins:{legend:{display:false}}, scales:{x:grid,y:{...grid,beginAtZero:true}}}});

new Chart(top, {type:'bar', data:{labels:D.top_access.map(t=>t.stem.replace(/^(user|feedback|project|personal|reference)_/,'')),
  datasets:[{data:D.top_access.map(t=>t.count),
    backgroundColor:D.top_access.map(t=>t.tier==='hot'?col('--hot'):col('--cold')),
    borderRadius:5}]},
  options:{indexAxis:'y', plugins:{legend:{display:false}},
    scales:{x:{...grid,beginAtZero:true},y:grid}}});

// ── relationship graph (D3 force) ───────────────────────────────────────────
(function(){
  const G = D.graph;
  if (!G || !G.nodes.length || typeof d3 === 'undefined') return;
  const wrap = document.getElementById('graph');
  const W = wrap.clientWidth || 900, H = 600;
  const palette = ['#6aa9ff','#3ddc97','#f0883e','#c792ea','#ff5c72','#ffd166',
                   '#4dd0e1','#a3be8c','#d08770','#88c0d0'];
  const sections = G.sections;
  const color = s => palette[Math.max(0, sections.indexOf(s)) % palette.length];
  const nodeR = d => d.kind==='section' ? 9 : 3 + Math.sqrt(d.count||0) * 0.85;

  const svg = d3.select('#graph').append('svg')
    .attr('width', W).attr('height', H)
    .style('cursor','grab');
  const root = svg.append('g');
  svg.call(d3.zoom().scaleExtent([0.2,4]).on('zoom', e => root.attr('transform', e.transform)));

  const link = root.append('g').selectAll('line').data(G.links).join('line')
    .attr('stroke', d => d.kind==='link' ? '#6aa9ff' : col('--grid'))
    .attr('stroke-opacity', d => d.kind==='link' ? 0.45 : 0.6)
    .attr('stroke-width', d => d.kind==='link' ? 1.1 : 1)
    .attr('class', d => 'edge-'+d.kind);

  const node = root.append('g').selectAll('g').data(G.nodes).join('g')
    .call(d3.drag()
      .on('start', (e,d)=>{ if(!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
      .on('drag',  (e,d)=>{ d.fx=e.x; d.fy=e.y; })
      .on('end',   (e,d)=>{ if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }));

  node.append('circle')
    .attr('r', nodeR)
    .attr('fill', d => color(d.section))
    .attr('fill-opacity', d => d.kind==='section' ? 1 : (d.tier==='cold'?0.4:0.9))
    .attr('stroke', d => d.kind==='section' ? '#fff' : (d.pinned ? col('--pin') : 'transparent'))
    .attr('stroke-width', d => d.kind==='section' ? 1.5 : (d.pinned ? 1.5 : 0));

  // section labels always shown
  node.filter(d=>d.kind==='section').append('text')
    .text(d=>d.label).attr('x',12).attr('y',4)
    .attr('fill', col('--ink')).attr('font-size','11px').attr('font-weight','600')
    .attr('paint-order','stroke').attr('stroke',col('--bg')).attr('stroke-width','3px');

  const tip = document.getElementById('tip');
  node.on('mousemove', (e,d) => {
      tip.style.opacity = 1;
      tip.style.left = (e.clientX+14)+'px'; tip.style.top = (e.clientY+14)+'px';
      tip.innerHTML = d.kind==='section'
        ? `<b>${esc(d.label)}</b><br><span style="color:var(--dim)">section (parent)</span>`
        : `<b>${esc(d.label)}</b><br><span style="color:var(--dim)">${esc(d.section)} · ${d.tier}${d.pinned?' · pinned':''} · ${d.count} uses</span>`;
    })
    .on('mouseleave', () => tip.style.opacity = 0)
    .on('click', (e,d) => {  // highlight a node's neighbourhood
      const nbr = new Set([d.id]);
      G.links.forEach(l => { const s=l.source.id||l.source, t=l.target.id||l.target;
        if(s===d.id) nbr.add(t); if(t===d.id) nbr.add(s); });
      node.attr('opacity', n => nbr.has(n.id) ? 1 : 0.12);
      link.attr('stroke-opacity', l => {
        const s=l.source.id||l.source, t=l.target.id||l.target;
        return (s===d.id||t===d.id) ? 0.9 : 0.04; });
      e.stopPropagation();
    });
  svg.on('click', () => { node.attr('opacity',1);
    link.attr('stroke-opacity', l => l.kind==='link'?0.45:0.6); });

  const sim = d3.forceSimulation(G.nodes)
    .force('link', d3.forceLink(G.links).id(d=>d.id)
      .distance(d => d.kind==='tree' ? 38 : 80)
      .strength(d => d.kind==='tree' ? 0.8 : 0.12))
    .force('charge', d3.forceManyBody().strength(-130))
    .force('center', d3.forceCenter(W/2, H/2))
    .force('collide', d3.forceCollide().radius(d => nodeR(d)+3))
    .on('tick', () => {
      link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)
          .attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
      node.attr('transform', d=>`translate(${d.x},${d.y})`);
    });

  // wikilink toggle
  document.getElementById('wlToggle').addEventListener('change', e => {
    root.selectAll('.edge-link').style('display', e.target.checked ? null : 'none');
  });

  // legend
  document.getElementById('graphLegend').innerHTML = sections.map(s =>
    `<span style="display:flex;align-items:center;gap:6px">
      <i style="width:11px;height:11px;border-radius:3px;background:${color(s)};display:inline-block"></i>
      <span style="color:var(--dim);font-size:11px">${esc(s)}</span></span>`).join('') +
    `<span style="color:var(--dim);font-size:11px">· ring = pinned · faded = cold · click a node to isolate</span>`;
})();
</script>
</body>
</html>
"""


def render(telemetry: dict, out: Path):
    # Escape "</" so a string field containing "</script>" cannot close the embedded
    # <script> block (json.dumps does not escape slashes). memdag ingests untrusted
    # text, so a poisoned section/stem name must not become live markup.
    data = json.dumps(telemetry).replace("</", "<" + chr(92) + "/")
    html = HTML_TEMPLATE.replace("__DATA__", data)
    out.write_text(html, encoding="utf-8")


def open_file(path: Path):
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", str(path)], check=False)
        elif system == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as e:
        print(f"(could not auto-open: {e})", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-open", action="store_true", help="build only, don't launch browser")
    ap.add_argument("--out", type=Path, default=HOME / "Desktop" / "memory-telemetry.html")
    args = ap.parse_args()

    telemetry = build_telemetry()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    render(telemetry, args.out)
    print(f"dashboard: {args.out}")
    t = telemetry["totals"]
    print(f"  {t['total']} memories  ·  {t['hot']} hot  ·  {t['cold']} cold  ·  {t['pinned']} pinned")
    if telemetry["budget"]:
        b = telemetry["budget"]
        print(f"  MEMORY.md: {b['bytes']}/{b['cap']} B ({b['pct']}%)")
    if not args.no_open:
        open_file(args.out)


def _cmd_dashboard(args):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    os.environ.setdefault("MEMDAG_DB", str(HOME / ".memdag" / "memdag.db"))
    out = Path(args.out) if getattr(args, "out", None) else HOME / "Desktop" / "memory-telemetry.html"
    telemetry = build_telemetry()
    out.parent.mkdir(parents=True, exist_ok=True)
    render(telemetry, out)
    t = telemetry["totals"]
    print(f"dashboard: {out}")
    print(f"  {t['total']} memories  ·  {t['hot']} hot  ·  {t['cold']} cold  ·  {t['pinned']} pinned")
    if telemetry["budget"]:
        b = telemetry["budget"]
        print(f"  MEMORY.md: {b['bytes']}/{b['cap']} B ({b['pct']}%)")
    if not getattr(args, "no_open", False):
        open_file(out)
    return 0


def register(sub) -> None:
    p = sub.add_parser("dashboard",
                       help="build + open the memory telemetry dashboard (HTML)")
    p.add_argument("--no-open", action="store_true",
                   help="build only, don't launch a browser")
    p.add_argument("--out", default=None,
                   help="output HTML path (default: ~/Desktop/memory-telemetry.html)")
    p.set_defaults(func=_cmd_dashboard)


if __name__ == "__main__":
    main()
