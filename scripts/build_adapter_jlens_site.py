#!/usr/bin/env python3
"""Build a static multi-prompt J-Lens browser from saved test Parquets.

The generated site contains no model weights and performs no inference.  It
publishes compact, presentation-oriented projections of already saved readouts.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import unicodedata
from pathlib import Path

import pandas as pd


FROZEN_LAYER = 40
RUN_ID = "run_20260903T141427Z_qwen36_20_adapter_full_test"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def normalized_output(text: str) -> str:
    """Collapse presentation-equivalent answers without rewriting their content."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def deduplicate_and_balance(
    records: list[tuple[dict, dict]],
) -> tuple[list[tuple[dict, dict]], dict]:
    """Keep one representative per answer and equal counts per prompt type.

    Deduplication is performed separately within direct and standard prompts so
    the two strata remain interpretable.  The representative is the earliest
    non-leak prompt ID when available.  If one stratum has more unique answers,
    systematic sampling across sorted prompt IDs matches the smaller stratum.
    Neither decision uses lens performance, avoiding outcome-based cherry-pick.
    """
    grouped: dict[str, dict[str, list[tuple[dict, dict]]]] = {
        "direct": {},
        "standard": {},
    }
    for summary, detail in records:
        prompt_type = summary["promptType"]
        key = normalized_output(summary["output"])
        grouped[prompt_type].setdefault(key, []).append((summary, detail))

    representatives: dict[str, list[tuple[dict, dict]]] = {}
    unique_before = {}
    for prompt_type, groups in grouped.items():
        unique_before[prompt_type] = len(groups)
        chosen = []
        for members in groups.values():
            members.sort(
                key=lambda pair: (
                    pair[0]["leaked"],
                    pair[0]["promptId"],
                )
            )
            summary, detail = members[0]
            duplicate_ids = sorted(pair[0]["promptId"] for pair in members)
            summary["duplicateCount"] = len(members)
            detail["duplicateCount"] = len(members)
            detail["duplicatePromptIds"] = duplicate_ids
            chosen.append((summary, detail))
        chosen.sort(
            key=lambda pair: pair[0]["promptId"]
        )
        representatives[prompt_type] = chosen

    per_type = min(len(representatives["direct"]), len(representatives["standard"]))
    def systematic_sample(values: list[tuple[dict, dict]], n: int):
        if len(values) == n:
            return values
        if n == 1:
            return [values[len(values) // 2]]
        indices = [round(i * (len(values) - 1) / (n - 1)) for i in range(n)]
        return [values[index] for index in indices]

    balanced = systematic_sample(
        representatives["direct"], per_type
    ) + systematic_sample(representatives["standard"], per_type)
    balanced.sort(key=lambda pair: (pair[0]["promptType"], pair[0]["promptId"]))
    stats = {
        "rawPrompts": len(records),
        "uniqueAnswersBeforeBalancing": unique_before,
        "keptPerType": per_type,
        "publishedPrompts": len(balanced),
        "collapsedOrBalanceDropped": len(records) - len(balanced),
        "normalization": "Unicode NFKC, trim, collapse whitespace; grouped within prompt type",
        "representativeRule": "prefer non-leak, then lowest prompt ID; balance by systematic sampling across sorted prompt IDs",
        "selectionUsesLensOutcome": False,
    }
    return balanced, stats


def read_behavior(path: Path, adapter: str) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("condition") == adapter:
                records[row["prompt_id"]] = row
    if len(records) != 200:
        raise RuntimeError(f"Expected 200 {adapter} behavior rows, found {len(records)}")
    return records


def primary_anchor(aggregate_path: Path) -> pd.DataFrame:
    columns = [
        "method",
        "layer",
        "mask_protocol",
        "target_rank",
        "target_reciprocal_rank",
        "target_probability_mass",
        "target_candidate_rank_20",
        "target_candidate_probability_share",
    ]
    frame = pd.read_parquet(aggregate_path, columns=columns)
    anchor = frame[
        frame["layer"].eq(FROZEN_LAYER)
        & frame["mask_protocol"].eq("global_emitted_ids")
    ].copy()
    if set(anchor["method"]) != {"jlens", "logit_lens"}:
        raise RuntimeError(f"Bad anchor rows in {aggregate_path}")
    return anchor.set_index("method")


def compact_prompt(
    position_path: Path,
    aggregate_path: Path,
    behavior: dict,
    adapter: str,
) -> tuple[dict, dict]:
    columns = [
        "method",
        "layer",
        "position_from_prompt_end",
        "position_role",
        "observed_token",
        "target_rank",
        "target_probability_mass",
        "top1_token",
        "top10_json",
    ]
    frame = pd.read_parquet(position_path, columns=columns)
    frame = frame[
        frame["position_role"].astype(str).str.startswith("response_token")
        & ~frame["observed_token"].astype(str).str.startswith("<|")
    ].copy()
    layers = sorted(int(value) for value in frame["layer"].unique())
    positions = sorted(int(value) for value in frame["position_from_prompt_end"].unique())
    expected = len(layers) * len(positions)
    methods: dict[str, dict] = {}
    indexed: dict[str, pd.DataFrame] = {}
    for method in ("jlens", "logit_lens"):
        part = frame[frame["method"].eq(method)].sort_values(
            ["layer", "position_from_prompt_end"]
        )
        if len(part) != expected:
            raise RuntimeError(f"Incomplete {method} grid in {position_path}: {len(part)}/{expected}")
        indexed[method] = part.set_index(["layer", "position_from_prompt_end"])
        methods[method] = {
            "rank": [int(value) for value in part["target_rank"]],
            "massPpm": [int(round(float(value) * 1_000_000)) for value in part["target_probability_mass"]],
            "top1": [str(value) for value in part["top1_token"]],
        }

    # Every card opens at exactly the same predeclared anchor.  Do not select a
    # visually favourable cell based on the observed lens difference.
    default_layer = FROZEN_LAYER
    default_position = 1 if 1 in positions else min(positions)

    default_top5 = {}
    for method in ("jlens", "logit_lens"):
        raw = indexed[method].loc[(default_layer, default_position), "top10_json"]
        default_top5[method] = [
            {
                "token": item["token"],
                "probability": round(float(item["probability"]), 7),
            }
            for item in json.loads(raw)[:5]
        ]

    token_rows = (
        frame[frame["method"].eq("jlens")]
        .drop_duplicates("position_from_prompt_end")
        .sort_values("position_from_prompt_end")
    )
    tokens = [
        {"position": int(row.position_from_prompt_end), "token": row.observed_token}
        for row in token_rows.itertuples(index=False)
    ]
    anchor = primary_anchor(aggregate_path)
    j = anchor.loc["jlens"]
    l = anchor.loc["logit_lens"]
    prompt = behavior["messages"][0]["content"]
    detail_name = f"{behavior['prompt_id']}.json"
    summary = {
        "promptId": behavior["prompt_id"],
        "promptType": behavior["prompt_type"],
        "prompt": prompt,
        "output": behavior["output_text"],
        "leaked": bool(behavior["own_secret_leaked"]),
        "generationTokens": int(behavior["generation_token_count"]),
        "jRank": int(j["target_rank"]),
        "lRank": int(l["target_rank"]),
        "jCandidateRank": int(j["target_candidate_rank_20"]),
        "lCandidateRank": int(l["target_candidate_rank_20"]),
        "jCandidateShare": round(float(j["target_candidate_probability_share"]), 6),
        "lCandidateShare": round(float(l["target_candidate_probability_share"]), 6),
        "jMrr": round(float(j["target_reciprocal_rank"]), 8),
        "lMrr": round(float(l["target_reciprocal_rank"]), 8),
        "logRankGain": round(math.log10(max(1, int(l["target_rank"]))) - math.log10(max(1, int(j["target_rank"]))), 4),
        "detail": f"data/prompts/{detail_name}",
    }
    detail = {
        "adapter": adapter,
        "promptId": behavior["prompt_id"],
        "promptType": behavior["prompt_type"],
        "prompt": prompt,
        "output": behavior["output_text"],
        "leaked": bool(behavior["own_secret_leaked"]),
        "frozenLayer": FROZEN_LAYER,
        "layers": layers,
        "positions": positions,
        "tokens": tokens,
        "methods": methods,
        "defaultLayer": default_layer,
        "defaultPosition": default_position,
        "defaultTop5": default_top5,
        "anchor": summary,
    }
    return summary, detail


def selection_metrics(results_dir: Path, adapter: str) -> dict:
    confusion = pd.read_csv(results_dir / "test_cross_candidate_confusion_at_anchors.csv")
    metrics = pd.read_csv(results_dir / "test_metrics_by_adapter_at_anchors.csv")
    direct = confusion[
        confusion["prompt_type"].eq("direct")
        & confusion["layer"].eq(FROZEN_LAYER)
        & confusion["actual_adapter"].eq(adapter)
        & confusion["predicted_candidate_20"].eq(adapter)
    ]
    accuracy = {
        row.method: float(row.prediction_rate) for row in direct.itertuples(index=False)
    }
    selected = metrics[
        metrics["prompt_type"].eq("direct")
        & metrics["layer"].eq(FROZEN_LAYER)
        & metrics["condition"].eq(adapter)
    ].set_index("method")
    return {
        "rule": "Among adapters with positive J-Lens gains in both direct layer-40 closed-set accuracy and full-vocabulary MRR, choose the largest accuracy gain.",
        "closedSetAccuracy": {
            "jlens": round(accuracy["jlens"], 6),
            "logitLens": round(accuracy["logit_lens"], 6),
            "delta": round(accuracy["jlens"] - accuracy["logit_lens"], 6),
        },
        "fullVocabulary": {
            "medianRankJlens": float(selected.loc["jlens", "median_rank"]),
            "medianRankLogitLens": float(selected.loc["logit_lens", "median_rank"]),
            "mrrJlens": round(float(selected.loc["jlens", "mrr"]), 6),
            "mrrLogitLens": round(float(selected.loc["logit_lens", "mrr"]), 6),
            "geometricRankFactor": round(
                float(selected.loc["logit_lens", "geometric_mean_rank"])
                / float(selected.loc["jlens", "geometric_mean_rank"]),
                4,
            ),
        },
    }


INDEX_HTML = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Interactive J-Lens versus Logit Lens browser for all saved smile-adapter Taboo prompts.">
<title>J-Lens × Taboo: smile prompt browser</title><link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E🔬%3C/text%3E%3C/svg%3E"><link rel="stylesheet" href="styles.css"></head>
<body><main id="app">
  <header><div class="eyebrow">Qwen3.6-27B · Taboo LoRA · saved test readouts</div><h1>Where <em>smile</em> appears without being said</h1>
  <p class="lede">For clarity, this is a balanced sample of prompts with different answers. Compare J-Lens with Logit Lens across source layers and response-token positions.</p>
  <div id="headline" class="headline"></div></header>
  <section class="overview"><div class="section-head"><div><h2>Selected prompts at frozen layer 40</h2><p>Each point is one response-averaged prompt. Above the diagonal means J-Lens gives the secret a better rank; below means Logit Lens does.</p></div><div class="legend"><span><i class="dot direct"></i>direct</span><span><i class="dot standard"></i>standard</span><span><i class="dot leak"></i>literal leak</span></div></div><svg id="scatter" aria-label="J-Lens rank versus Logit Lens rank by prompt"></svg></section>
  <section class="browser"><aside><div class="section-head compact"><div><h2>Prompt browser</h2><p id="count"></p></div></div><div class="filters"><select id="type-filter"><option value="all">standard + direct</option><option value="direct">direct only</option><option value="standard">standard only</option><option value="leaks">literal leaks</option></select><select id="sort"><option value="gain">largest J-Lens gain</option><option value="id">prompt ID</option></select></div><div id="prompt-list" class="prompt-list"></div></aside>
  <article id="detail"><div class="empty">Loading the strongest non-leak example…</div></article></section>
  <footer>Illustrative browser, not a layer-selection procedure. Layer 40 was frozen on validation. Global emitted-token-ID mask; literal own-secret leaks are flagged and excluded from headline metrics. Readouts establish decodability, not causal use.</footer>
</main><div id="tooltip" role="tooltip"></div><script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script><script src="app.js"></script></body></html>'''


STYLES_CSS = '''
:root{--ink:#17202a;--muted:#667080;--line:#d8dde5;--paper:#fbfaf7;--panel:#fff;--soft:#f0f2f5;--blue:#1767c0;--orange:#e07a27;--gold:#f4ca21;--green:#187044}*{box-sizing:border-box}body{margin:0;background:#eef0f3;color:var(--ink);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}main{max-width:1280px;margin:0 auto;padding:28px}header,.overview,.browser,footer{background:var(--paper);border:1px solid var(--line)}header{padding:28px 30px;border-radius:18px 18px 0 0}.eyebrow,.role,h3{font-size:11px;text-transform:uppercase;letter-spacing:.1em;font-weight:750;color:#596475}h1{font:750 34px/1.08 ui-serif,Georgia,serif;letter-spacing:-.025em;margin:7px 0}.lede{font-size:16px;color:var(--muted);max-width:820px;margin:0}.headline{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:18px}.metric{background:#fff;border:1px solid var(--line);padding:11px 12px;border-radius:9px;min-width:0}.metric b{font-size:18px;display:block;white-space:normal;overflow-wrap:anywhere}.metric span{display:block;font-size:11px;line-height:1.35;color:var(--muted);white-space:normal;overflow-wrap:anywhere}.overview{padding:22px 26px;border-top:0}.section-head{display:flex;justify-content:space-between;gap:16px;align-items:start}.section-head.compact{align-items:center}.section-head h2{font-size:18px;margin:0}.section-head p{color:var(--muted);margin:3px 0 0;font-size:12px}.legend{display:flex;gap:13px;color:var(--muted);font-size:12px;flex-wrap:wrap}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}.dot.direct{background:#1767c0}.dot.standard{background:#18a077}.dot.leak{background:#fff;border:2px solid #c0392b}.overview svg{display:block;width:100%;height:330px;margin-top:8px}.browser{display:grid;grid-template-columns:330px minmax(0,1fr);border-top:0;min-height:760px}.browser aside{padding:20px;border-right:1px solid var(--line);min-width:0}.filters{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin:12px 0}.filters select{width:100%;padding:8px;border:1px solid var(--line);border-radius:7px;background:white;color:var(--ink)}.prompt-list{height:650px;overflow:auto;border-top:1px solid var(--line)}.prompt-row{display:block;width:100%;text-align:left;border:0;border-bottom:1px solid #e8ebef;background:transparent;padding:10px 6px;cursor:pointer;color:inherit}.prompt-row:hover,.prompt-row.active{background:#fff6cf}.prompt-row .row-top{display:flex;justify-content:space-between;gap:8px;font:12px ui-monospace,SFMono-Regular,Menlo,monospace}.prompt-row .snippet{font-size:12px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:3px}.tag{border:1px solid var(--line);border-radius:999px;padding:1px 5px;font-size:10px}.tag.leak{border-color:#dc9a92;color:#9d281e;background:#fff1ef}article{padding:22px;min-width:0}.empty{color:var(--muted);padding:30px}.dialogue{display:grid;grid-template-columns:1fr 1.65fr;gap:10px}.card{border:1px solid var(--line);background:white;border-radius:10px;padding:12px;min-width:0}.role{margin-bottom:5px}.response{display:flex;align-items:baseline;flex-wrap:wrap;gap:2px}.token{border:0;border-radius:4px;padding:2px 3px;background:transparent;color:inherit;font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;cursor:pointer}.token:hover{background:#fff0a8}.token.selected{background:var(--gold);box-shadow:0 0 0 1px #a77e00 inset}.toolbar{display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap;margin:13px 0}.toggle{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}.toggle button{border:0;background:white;padding:7px 12px;cursor:pointer;font-weight:650;color:#4a5360}.toggle button.active{color:white;background:#253246}.selection{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:#3d4653}.detail-grid{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(250px,.8fr);gap:11px}.stack{display:grid;gap:11px}.card h3{margin:0 0 8px}.heat,.line-chart{width:100%;height:auto;display:block}.ramp{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:11px;margin-top:5px}.ramp i{width:120px;height:9px;border-radius:6px;background:linear-gradient(90deg,#440154,#31688e,#35b779,#fde725)}.cell-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.cell-metrics div{background:var(--soft);border-radius:7px;padding:7px}.cell-metrics b{display:block;font-size:17px}.cell-metrics span{font-size:10px;color:var(--muted)}.top-table,.top-grid{width:100%;border-collapse:collapse;font:11px ui-monospace,SFMono-Regular,Menlo,monospace}.top-table td,.top-grid td,.top-grid th{padding:4px;border-bottom:1px solid #ebedf1}.top-table td:last-child{text-align:right;color:var(--muted)}.top-grid-wrap{overflow:auto;max-height:270px}.top-grid td{cursor:pointer;max-width:66px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.axis text{fill:#667080;font-size:10px}.axis path,.axis line{stroke:#cdd2da}footer{border-top:0;border-radius:0 0 18px 18px;padding:14px 24px;color:var(--muted);font-size:12px}#tooltip{position:fixed;z-index:30;pointer-events:none;opacity:0;background:#18212d;color:white;border-radius:6px;padding:7px 9px;font-size:12px;box-shadow:0 4px 15px #0003}@media(max-width:900px){main{padding:14px}.headline{grid-template-columns:repeat(2,minmax(0,1fr))}.browser{grid-template-columns:1fr}.browser aside{border-right:0;border-bottom:1px solid var(--line)}.prompt-list{height:280px}.detail-grid{grid-template-columns:1fr}}@media(max-width:620px){main{padding:0}header{border-radius:0;padding:20px}.headline{grid-template-columns:1fr}.overview,.browser aside,article{padding:16px}.browser,header,.overview,footer{border-left:0;border-right:0}.dialogue{grid-template-columns:1fr}.section-head{display:block}.legend{margin-top:8px}.overview svg{height:270px}.cell-metrics{grid-template-columns:1fr}.filters{grid-template-columns:1fr}footer{border-radius:0}}
'''


APP_JS = r'''
const state={summary:null,visible:[],detail:null,method:'jlens',layer:40,position:1};
const $=s=>document.querySelector(s); const pretty=t=>t==='\n'?'↵':t===' '?'␠':String(t).replaceAll('\n','↵');
const rankColor=r=>d3.interpolateViridis(Math.max(0,Math.min(1,(4-Math.log10(Math.max(1,Math.min(10000,r))))/4)));
const fmtRank=r=>r>=10000?`${Math.round(r/1000)}k`:r.toLocaleString();
const tooltip=$('#tooltip');
function cell(method,l,p){const d=state.detail,pi=d.positions.indexOf(p),li=d.layers.indexOf(l);if(pi<0||li<0)return null;const i=li*d.positions.length+pi,m=d.methods[method];return{rank:m.rank[i],mass:m.massPpm[i]/1e6,top1:m.top1[i]}}
async function init(){const r=await fetch('data/summary.json');state.summary=await r.json();renderHeadline();applyFilters();const first=state.visible.find(d=>!d.leaked)||state.visible[0];await loadPrompt(first.promptId)}
function renderHeadline(){const m=state.summary.selection,meta=state.summary.meta;$('#headline').innerHTML=`<div class="metric"><b>${meta.prompts}</b><span>unique balanced examples · ${meta.balance.keptPerType} + ${meta.balance.keptPerType}</span></div><div class="metric"><b>${(100*m.closedSetAccuracy.jlens).toFixed(1)}% vs ${(100*m.closedSetAccuracy.logitLens).toFixed(1)}%</b><span>direct 20-way accuracy · J-Lens vs Logit</span></div><div class="metric"><b>${m.fullVocabulary.medianRankJlens} vs ${m.fullVocabulary.medianRankLogitLens}</b><span>direct median vocabulary rank</span></div><div class="metric"><b>${m.fullVocabulary.geometricRankFactor.toFixed(1)}×</b><span>geometric-rank improvement</span></div>`}
function applyFilters(){const type=$('#type-filter').value,sort=$('#sort').value;let v=state.summary.prompts.filter(d=>type==='all'||(type==='leaks'?d.leaked:d.promptType===type));v.sort(sort==='id'?(a,b)=>a.promptId.localeCompare(b.promptId):(a,b)=>(b.logRankGain??-99)-(a.logRankGain??-99));state.visible=v;$('#count').textContent=`${v.length} of ${state.summary.prompts.length}`;renderList();drawScatter()}
function renderList(){const box=$('#prompt-list');box.innerHTML=state.visible.map(d=>`<button class="prompt-row ${state.detail?.promptId===d.promptId?'active':''}" data-id="${d.promptId}"><div class="row-top"><span>${d.promptId}</span><span>${d.leaked?'<span class="tag leak">leak</span>':`Δlog rank ${d.logRankGain>=0?'+':''}${d.logRankGain}`}</span></div><div class="snippet">${escapeHtml(d.output)}</div></button>`).join('');box.querySelectorAll('button').forEach(b=>b.onclick=()=>loadPrompt(b.dataset.id))}
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function loadPrompt(id){const meta=state.summary.prompts.find(d=>d.promptId===id);$('#detail').innerHTML='<div class="empty">Loading saved layer × position readout…</div>';const r=await fetch(meta.detail);state.detail=await r.json();state.layer=state.detail.defaultLayer;state.position=state.detail.defaultPosition;state.method='jlens';renderDetail();renderList();drawScatter()}
function renderDetail(){const d=state.detail;$('#detail').innerHTML=`<div class="dialogue"><div class="card"><div class="role">Human · ${d.promptType}</div><div>${escapeHtml(d.prompt)}</div></div><div class="card"><div class="role">Assistant ${d.leaked?'<span class="tag leak">literal secret leak</span>':''}</div><div id="response" class="response"></div></div></div><div class="toolbar"><div class="toggle"><button data-m="jlens" class="active">J-Lens</button><button data-m="logit_lens">Logit Lens</button></div><div id="selection" class="selection"></div></div><div class="detail-grid"><div class="stack"><div class="card"><h3>Secret rank · source layer × response position</h3><svg id="heat" class="heat"></svg><div class="ramp"><span>rank 10k+</span><i></i><span>rank 1</span></div></div><div class="card"><h3>Top decoded token · sampled layers</h3><div id="top-grid" class="top-grid-wrap"></div></div></div><div class="stack"><div class="card"><h3>Selected cell</h3><div id="cell-metrics" class="cell-metrics"></div><table id="top-table" class="top-table"></table></div><div class="card"><h3>Rank by layer · selected response token</h3><svg id="layer-chart" class="line-chart"></svg></div><div class="card"><h3>Rank by response position · selected layer</h3><svg id="pos-chart" class="line-chart"></svg></div></div></div>`;const response=$('#response');response.innerHTML=d.tokens.map(x=>`<button class="token" data-p="${x.position}">${escapeHtml(pretty(x.token))}</button>`).join('');response.querySelectorAll('button').forEach(b=>b.onclick=()=>{state.position=+b.dataset.p;updateDetail()});document.querySelectorAll('.toggle button').forEach(b=>b.onclick=()=>{state.method=b.dataset.m;document.querySelectorAll('.toggle button').forEach(x=>x.classList.toggle('active',x===b));updateDetail()});updateDetail()}
function updateDetail(){drawHeat();drawTopGrid();drawSelected();drawLine('#layer-chart',state.detail.layers,(m,x)=>cell(m,x,state.position),'layer');drawLine('#pos-chart',state.detail.positions,(m,x)=>cell(m,state.layer,x),'position');document.querySelectorAll('.token').forEach(b=>b.classList.toggle('selected',+b.dataset.p===state.position))}
function drawHeat(){const d=state.detail,svg=d3.select('#heat'),node=svg.node(),W=Math.max(360,node.parentElement.clientWidth-24),H=Math.max(300,Math.min(440,W*.56)),mg={t:8,r:10,b:42,l:48};svg.attr('viewBox',`0 0 ${W} ${H}`).selectAll('*').remove();const x=d3.scaleBand().domain(d.positions).range([mg.l,W-mg.r]).padding(.035),y=d3.scaleBand().domain([...d.layers].reverse()).range([mg.t,H-mg.b]).padding(.025);const values=[];d.layers.forEach(l=>d.positions.forEach(p=>values.push({l,p,...cell(state.method,l,p)})));svg.append('g').selectAll('rect').data(values).join('rect').attr('x',v=>x(v.p)).attr('y',v=>y(v.l)).attr('width',x.bandwidth()).attr('height',y.bandwidth()).attr('fill',v=>rankColor(v.rank)).attr('stroke',v=>v.l===state.layer&&v.p===state.position?'#fff':'none').attr('stroke-width',2).style('cursor','pointer').on('click',(_,v)=>{state.layer=v.l;state.position=v.p;updateDetail()}).on('mousemove',(e,v)=>showTip(e,`layer ${v.l} · position ${v.p}<br>rank <b>${fmtRank(v.rank)}</b> · top-1 <b>${escapeHtml(pretty(v.top1))}</b>`)).on('mouseleave',hideTip);svg.append('g').attr('class','axis').attr('transform',`translate(0,${H-mg.b})`).call(d3.axisBottom(x).tickValues(d.positions.filter(p=>p===1||p%4===0)));svg.append('g').attr('class','axis').attr('transform',`translate(${mg.l},0)`).call(d3.axisLeft(y).tickValues(d.layers.filter(l=>l%5===0||l===62)));svg.append('line').attr('x1',mg.l).attr('x2',W-mg.r).attr('y1',y(d.frozenLayer)+y.bandwidth()/2).attr('y2',y(d.frozenLayer)+y.bandwidth()/2).attr('stroke','#fff').attr('stroke-dasharray','4,3');svg.append('text').attr('x',(mg.l+W-mg.r)/2).attr('y',H-6).attr('text-anchor','middle').attr('fill','#667080').attr('font-size',11).text('response token position →');svg.append('text').attr('transform','rotate(-90)').attr('x',-(mg.t+H-mg.b)/2).attr('y',13).attr('text-anchor','middle').attr('fill','#667080').attr('font-size',11).text('source layer →')}
function drawTopGrid(){const d=state.detail,sampled=d.layers.filter(l=>l>=20&&(l%4===0||l===62)).reverse();let h='<table class="top-grid"><thead><tr><th>layer</th>'+d.positions.map(p=>`<th>${p}</th>`).join('')+'</tr></thead><tbody>';sampled.forEach(l=>{h+=`<tr><th>${l}</th>`;d.positions.forEach(p=>{const v=cell(state.method,l,p),active=l===state.layer&&p===state.position;h+=`<td data-l="${l}" data-p="${p}" title="${escapeHtml(v.top1)} · secret rank ${v.rank}" style="background:${active?'#f4ca21':rankColor(v.rank)};color:${v.rank<=10?'#17202a':'white'}">${escapeHtml(pretty(v.top1).slice(0,10))}</td>`});h+='</tr>'});h+='</tbody></table>';$('#top-grid').innerHTML=h;document.querySelectorAll('#top-grid td').forEach(td=>td.onclick=()=>{state.layer=+td.dataset.l;state.position=+td.dataset.p;updateDetail()})}
function drawSelected(){const d=state.detail,v=cell(state.method,state.layer,state.position),other=cell(state.method==='jlens'?'logit_lens':'jlens',state.layer,state.position),tok=d.tokens.find(x=>x.position===state.position)?.token??'';$('#selection').textContent=`${state.method==='jlens'?'J-Lens':'Logit Lens'} · layer ${state.layer} · position ${state.position} · after “${pretty(tok)}”`;$('#cell-metrics').innerHTML=`<div><b>#${fmtRank(v.rank)}</b><span>smile rank</span></div><div><b>${(100*v.mass).toFixed(2)}%</b><span>probability mass</span></div><div><b>#${fmtRank(other.rank)}</b><span>other lens rank</span></div>`;const isDefault=state.layer===d.defaultLayer&&state.position===d.defaultPosition,top=isDefault?d.defaultTop5[state.method]:[{token:v.top1,probability:null}];$('#top-table').innerHTML='<tbody>'+top.map((x,i)=>`<tr><td>${i+1}</td><td>${escapeHtml(pretty(x.token))}</td><td>${x.probability==null?'top-1':(100*x.probability).toFixed(2)+'%'}</td></tr>`).join('')+'</tbody>'}
function drawLine(sel,xs,getter,kind){const svg=d3.select(sel),node=svg.node(),W=Math.max(280,node.parentElement.clientWidth-24),H=190,mg={t:10,r:10,b:34,l:48};svg.attr('viewBox',`0 0 ${W} ${H}`).selectAll('*').remove();const x=d3.scaleLinear().domain(d3.extent(xs)).range([mg.l,W-mg.r]),y=d3.scaleLog().domain([1,10000]).range([mg.t,H-mg.b]).clamp(true);svg.append('g').attr('class','axis').attr('transform',`translate(0,${H-mg.b})`).call(d3.axisBottom(x).ticks(W<400?4:6));svg.append('g').attr('class','axis').attr('transform',`translate(${mg.l},0)`).call(d3.axisLeft(y).tickValues([1,10,100,1000,10000]).tickFormat(v=>v===10000?'10k':v));[['jlens','#1767c0'],['logit_lens','#e07a27']].forEach(([m,c])=>{const vals=xs.map(z=>({x:z,y:Math.min(10000,getter(m,z).rank)}));svg.append('path').datum(vals).attr('fill','none').attr('stroke',c).attr('stroke-width',2).attr('d',d3.line().x(v=>x(v.x)).y(v=>y(v.y)))});const mark=kind==='layer'?state.layer:state.position;svg.append('line').attr('x1',x(mark)).attr('x2',x(mark)).attr('y1',mg.t).attr('y2',H-mg.b).attr('stroke','#8b2d83').attr('stroke-dasharray','4,3');if(kind==='layer')svg.append('line').attr('x1',x(state.detail.frozenLayer)).attr('x2',x(state.detail.frozenLayer)).attr('y1',mg.t).attr('y2',H-mg.b).attr('stroke','#555').attr('stroke-dasharray','3,3');svg.append('text').attr('x',W-8).attr('y',14).attr('text-anchor','end').attr('fill','#1767c0').attr('font-size',10).text('J-Lens');svg.append('text').attr('x',W-8).attr('y',27).attr('text-anchor','end').attr('fill','#e07a27').attr('font-size',10).text('Logit Lens')}
function drawScatter(){if(!state.summary)return;const svg=d3.select('#scatter'),node=svg.node(),W=Math.max(320,node.parentElement.clientWidth-52),H=node.clientHeight||330,mg={t:14,r:18,b:48,l:64};svg.attr('viewBox',`0 0 ${W} ${H}`).selectAll('*').remove();const x=d3.scaleLog().domain([1,250000]).range([mg.l,W-mg.r]),y=d3.scaleLog().domain([1,250000]).range([mg.t,H-mg.b]);const ticks=[1,10,100,1000,10000,100000];svg.append('rect').attr('x',mg.l).attr('y',mg.t).attr('width',W-mg.l-mg.r).attr('height',H-mg.t-mg.b).attr('fill','#fff').attr('stroke','#d8dde5');svg.append('path').attr('d',d3.line()([[x(1),y(1)],[x(250000),y(250000)]] )).attr('stroke','#8a929e').attr('stroke-dasharray','4,4');svg.append('g').attr('class','axis').attr('transform',`translate(0,${H-mg.b})`).call(d3.axisBottom(x).tickValues(ticks).tickFormat(v=>v>=1000?`${v/1000}k`:v));svg.append('g').attr('class','axis').attr('transform',`translate(${mg.l},0)`).call(d3.axisLeft(y).tickValues(ticks).tickFormat(v=>v>=1000?`${v/1000}k`:v));svg.append('text').attr('x',(mg.l+W-mg.r)/2).attr('y',H-7).attr('text-anchor','middle').attr('fill','#667080').text('Logit Lens vocabulary rank →');svg.append('text').attr('transform','rotate(-90)').attr('x',-(mg.t+H-mg.b)/2).attr('y',14).attr('text-anchor','middle').attr('fill','#667080').text('J-Lens vocabulary rank →');const current=state.detail?.promptId;svg.append('g').selectAll('circle').data(state.visible).join('circle').attr('cx',d=>x(d.lRank)).attr('cy',d=>y(d.jRank)).attr('r',d=>d.promptId===current?6:4).attr('fill',d=>d.promptType==='direct'?'#1767c0':'#18a077').attr('fill-opacity',.68).attr('stroke',d=>d.leaked?'#c0392b':d.promptId===current?'#17202a':'#fff').attr('stroke-width',d=>d.promptId===current||d.leaked?2:1).style('cursor','pointer').on('click',(_,d)=>loadPrompt(d.promptId)).on('mousemove',(e,d)=>showTip(e,`<b>${d.promptId}</b> · ${d.promptType}${d.leaked?' · leak':''}<br>J-Lens #${fmtRank(d.jRank)} · Logit #${fmtRank(d.lRank)}`)).on('mouseleave',hideTip)}
function showTip(e,h){tooltip.innerHTML=h;tooltip.style.opacity=1;tooltip.style.left=`${e.clientX+12}px`;tooltip.style.top=`${e.clientY+12}px`}function hideTip(){tooltip.style.opacity=0}
$('#type-filter').onchange=applyFilters;$('#sort').onchange=applyFilters;window.addEventListener('resize',()=>{drawScatter();if(state.detail)updateDetail()});init().catch(e=>{$('#detail').innerHTML=`<div class="empty">Could not load saved data: ${escapeHtml(e.message)}</div>`;console.error(e)});
'''


def build_site(root: Path, output: Path, adapter: str, run_id: str) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "data" / "prompts").mkdir(parents=True)
    run_cells = root / "artifacts" / "lens_outputs" / run_id / "test_cells"
    behavior_path = root / "data" / "raw_outputs" / run_id / "test_behavior_generations.jsonl"
    results_dir = root / "results" / run_id
    behavior = read_behavior(behavior_path, adapter)
    records: list[tuple[dict, dict]] = []
    for index, position_path in enumerate(sorted(run_cells.glob(f"*__{adapter}.positions.parquet")), start=1):
        prompt_id = position_path.name.removesuffix(f"__{adapter}.positions.parquet")
        aggregate_path = run_cells / f"{prompt_id}__{adapter}.aggregate.parquet"
        summary, detail = compact_prompt(
            position_path, aggregate_path, behavior[prompt_id], adapter
        )
        records.append((summary, detail))
        if index == 1 or index % 25 == 0:
            print(f"processed {index}/200: {prompt_id}", flush=True)
    if len(records) != 200:
        raise RuntimeError(f"Expected 200 position files, found {len(records)}")
    balanced, balance_stats = deduplicate_and_balance(records)
    summaries = [summary for summary, _ in balanced]
    for summary, detail in balanced:
        write_json(output / summary["detail"], detail)
    selection = selection_metrics(results_dir, adapter)
    valid = [row for row in summaries if not row["leaked"]]
    summary_payload = {
        "meta": {
            "adapter": adapter,
            "runId": run_id,
            "frozenLayer": FROZEN_LAYER,
            "mask": "global_emitted_ids",
            "prompts": len(summaries),
            "validPrompts": len(valid),
            "literalLeaks": len(summaries) - len(valid),
            "balance": balance_stats,
            "baseModelRevision": behavior[next(iter(behavior))]["base_model_revision"],
            "jlensRevision": behavior[next(iter(behavior))]["jlens_revision"],
        },
        "selection": selection,
        "prompts": summaries,
    }
    write_json(output / "data" / "summary.json", summary_payload)
    (output / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (output / "styles.css").write_text(STYLES_CSS.strip() + "\n", encoding="utf-8")
    (output / "app.js").write_text(APP_JS.strip() + "\n", encoding="utf-8")
    (output / ".nojekyll").write_text("", encoding="utf-8")
    (output / "README.md").write_text(
        f"""# J-Lens x Taboo: `{adapter}` prompt browser

Static GitHub Pages site generated from saved run `{run_id}`. Exact duplicate
answers are collapsed within each prompt type, and the published browser keeps
equal numbers of direct and standard examples. It contains compact
layer-by-position target ranks, probability masses, and top-1 decoded tokens
for J-Lens and Logit Lens.

The adapter was selected by a predeclared robust demonstration rule recorded in
`data/summary.json`. Literal own-secret leaks are visible but excluded from
headline metrics. No model weights, credentials, or hidden activations are
published.
""",
        encoding="utf-8",
    )
    total = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(json.dumps({"output": str(output), "files": sum(1 for p in output.rglob('*') if p.is_file()), "bytes": total, "selection": selection}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", default="smile")
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    build_site(args.root, args.output, args.adapter, args.run_id)


if __name__ == "__main__":
    main()
