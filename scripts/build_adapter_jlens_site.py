#!/usr/bin/env python3
"""Build a static multi-prompt J-Lens browser from saved test Parquets.

The generated site contains no model weights and performs no inference.  It
publishes compact, presentation-oriented projections of already saved readouts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import unicodedata
from pathlib import Path

import pandas as pd


FROZEN_LAYER = 40
RUN_ID = "run_20260903T141427Z_qwen36_20_adapter_full_test"
EXAMPLES_PER_TYPE = 25
DEFAULT_ADAPTER = "smile"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def normalized_output(text: str) -> str:
    """Collapse presentation-equivalent answers without rewriting their content."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def lexical_features(text: str) -> frozenset[str]:
    """Represent both word content and repeated phrase fragments."""
    normalized = normalized_output(text).casefold()
    words = re.findall(r"\w+", normalized)
    unigrams = {f"u:{word}" for word in words}
    bigrams = {f"b:{left}_{right}" for left, right in zip(words, words[1:])}
    compact = " ".join(words)
    char_ngrams = {
        f"c:{compact[index:index + 5]}"
        for index in range(max(0, len(compact) - 4))
    }
    return frozenset(unigrams | bigrams | char_ngrams)


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def diverse_sample(
    values: list[tuple[dict, dict]], n: int
) -> list[tuple[dict, dict]]:
    """Greedy max-min sample that always retains distinct literal leaks."""
    values = sorted(values, key=lambda pair: pair[0]["promptId"])
    if len(values) <= n:
        return values
    features = [lexical_features(pair[0]["output"]) for pair in values]
    pairwise = [
        [jaccard(features[i], features[j]) for j in range(len(values))]
        for i in range(len(values))
    ]
    selected = [i for i, pair in enumerate(values) if pair[0]["leaked"]]
    if len(selected) > n:
        selected = selected[:n]
    if not selected:
        first_i, second_i = min(
            (
                (i, j)
                for i in range(len(values))
                for j in range(i + 1, len(values))
            ),
            key=lambda pair: (
                pairwise[pair[0]][pair[1]],
                values[pair[0]][0]["promptId"],
                values[pair[1]][0]["promptId"],
            ),
        )
        selected = [first_i, second_i]
    remaining = set(range(len(values))) - set(selected)
    while len(selected) < n:
        next_i = min(
            remaining,
            key=lambda i: (
                max(pairwise[i][chosen] for chosen in selected),
                sum(pairwise[i][chosen] for chosen in selected),
                values[i][0]["promptId"],
            ),
        )
        selected.append(next_i)
        remaining.remove(next_i)
    chosen = [values[index] for index in selected]
    leaks = [pair for pair in chosen if pair[0]["leaked"]]
    non_leaks = [pair for pair in chosen if not pair[0]["leaked"]]
    if not leaks:
        return chosen
    spread = list(non_leaks)
    for leak_number, leak in enumerate(leaks, start=1):
        target = round(leak_number * (len(spread) + 1) / (len(leaks) + 1))
        spread.insert(min(target, len(spread)), leak)
    return spread


def deduplicate_and_balance(
    records: list[tuple[dict, dict]],
) -> tuple[list[tuple[dict, dict]], dict]:
    """Keep one representative per answer and 50 diverse examples per adapter.

    Deduplication is performed separately within direct and standard prompts so
    the two strata remain interpretable.  The representative is the earliest
    prompt ID, preferring a non-leak only when an identical answer appears with
    both labels. Distinct literal-leak answers are retained. The sample starts
    balanced at 25 per prompt type, then fills a uniqueness shortfall from the
    other type. Remaining slots are selected by greedy max-min lexical distance.
    Lens performance is never used for sample selection.
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

    target_total = min(
        EXAMPLES_PER_TYPE * 2,
        len(representatives["direct"]) + len(representatives["standard"]),
    )
    kept_by_type = {
        prompt_type: min(EXAMPLES_PER_TYPE, len(values))
        for prompt_type, values in representatives.items()
    }
    while sum(kept_by_type.values()) < target_total:
        prompt_type = max(
            (
                prompt_type
                for prompt_type, values in representatives.items()
                if kept_by_type[prompt_type] < len(values)
            ),
            key=lambda name: (
                len(representatives[name]) - kept_by_type[name],
                name == "direct",
            ),
        )
        kept_by_type[prompt_type] += 1
    direct_sample = diverse_sample(
        representatives["direct"], kept_by_type["direct"]
    )
    standard_sample = diverse_sample(
        representatives["standard"], kept_by_type["standard"]
    )
    balanced = [
        pair
        for pair_number in range(max(map(len, (direct_sample, standard_sample))))
        for sample in (direct_sample, standard_sample)
        if pair_number < len(sample)
        for pair in (sample[pair_number],)
    ]
    for display_order, (summary, detail) in enumerate(balanced):
        summary["displayOrder"] = display_order
        detail["displayOrder"] = display_order
    stats = {
        "rawPrompts": len(records),
        "uniqueAnswersBeforeBalancing": unique_before,
        "keptByType": kept_by_type,
        "publishedPrompts": len(balanced),
        "collapsedOrBalanceDropped": len(records) - len(balanced),
        "normalization": "Unicode NFKC, trim, collapse whitespace; grouped within prompt type",
        "representativeRule": "retain distinct literal leaks, collapse exact normalized duplicates, then greedily maximize minimum Jaccard distance over answer word and character n-grams",
        "selectionUsesLensOutcome": False,
    }
    return balanced, stats


def read_all_behavior(path: Path) -> dict[str, dict[str, dict]]:
    records: dict[str, dict[str, dict]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            adapter = row.get("condition")
            if adapter:
                records.setdefault(adapter, {})[row["prompt_id"]] = row
    bad_counts = {
        adapter: len(adapter_records)
        for adapter, adapter_records in records.items()
        if len(adapter_records) != 200
    }
    if bad_counts:
        raise RuntimeError(f"Expected 200 behavior rows per adapter, found {bad_counts}")
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
    detail_name = f"data/adapters/{adapter}/prompts/{behavior['prompt_id']}.json"
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
        "detail": detail_name,
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


INDEX_HTML = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Interactive J-Lens versus Logit Lens browser for all 20 Qwen Taboo adapters.">
<title>J-Lens × Taboo: adapter prompt browser</title><link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E🔬%3C/text%3E%3C/svg%3E"><link rel="stylesheet" href="styles.css"></head>
<body><main id="app">
  <header><div class="eyebrow">Qwen3.6-27B · 20 Taboo LoRAs · saved test readouts</div><h1>Where <em id="target-name">smile</em> appears without being said</h1>
  <p class="lede">Choose a secret adapter, then compare J-Lens with Logit Lens across source layers and response-token positions.</p>
  <div class="adapter-bar"><label for="adapter-filter"><span>Secret adapter</span><select id="adapter-filter" aria-label="Secret adapter"></select></label><p id="sample-note" class="sample-note">Loading saved examples…</p></div></header>
  <section class="overview"><div class="section-head"><div><h2><span id="overview-count">50</span> selected examples at frozen layer 40</h2><p>Each point is one response-averaged example. Above the diagonal means J-Lens gives the secret a better rank; below means Logit Lens does.</p></div><div class="legend"><span><i class="dot direct"></i>direct</span><span><i class="dot standard"></i>standard</span><span><i class="dot leak"></i>literal leak</span></div></div><svg id="scatter" aria-label="J-Lens rank versus Logit Lens rank by prompt"></svg></section>
  <section class="browser"><aside><div class="section-head compact"><div><h2>Prompt browser</h2><p id="count"></p></div></div><div class="filters"><select id="type-filter"><option value="all">standard + direct</option><option value="direct">direct only</option><option value="standard">standard only</option><option value="leaks">literal leaks</option></select><select id="sort"><option value="diverse">diverse sample order</option><option value="gain">largest J-Lens gain</option><option value="id">prompt ID</option></select></div><div id="prompt-list" class="prompt-list"></div></aside>
  <article id="detail"><div class="empty">Loading a saved example…</div></article></section>
  <footer>Exploratory browser, not a layer-selection procedure. Layer 40 was frozen on validation. Literal own-secret leaks remain visible and flagged. Readouts establish decodability under each lens, not causal use.</footer>
</main><div id="tooltip" role="tooltip"></div><script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script><script src="app.js"></script></body></html>'''


STYLES_CSS = '''
:root{--ink:#17202a;--muted:#667080;--line:#d8dde5;--paper:#fbfaf7;--panel:#fff;--soft:#f0f2f5;--blue:#1767c0;--orange:#e07a27;--gold:#f4ca21;--green:#187044}*{box-sizing:border-box}body{margin:0;background:#eef0f3;color:var(--ink);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}main{max-width:1280px;margin:0 auto;padding:28px}header,.overview,.browser,footer{background:var(--paper);border:1px solid var(--line)}header{padding:28px 30px;border-radius:18px 18px 0 0}.eyebrow,.role,h3{font-size:11px;text-transform:uppercase;letter-spacing:.1em;font-weight:750;color:#596475}h1{font:750 34px/1.08 ui-serif,Georgia,serif;letter-spacing:-.025em;margin:7px 0}.lede{font-size:16px;color:var(--muted);max-width:820px;margin:0}.headline{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:18px}.metric{background:#fff;border:1px solid var(--line);padding:11px 12px;border-radius:9px;min-width:0}.metric b{font-size:18px;display:block;white-space:normal;overflow-wrap:anywhere}.metric span{display:block;font-size:11px;line-height:1.35;color:var(--muted);white-space:normal;overflow-wrap:anywhere}.overview{padding:22px 26px;border-top:0}.section-head{display:flex;justify-content:space-between;gap:16px;align-items:start}.section-head.compact{align-items:center}.section-head h2{font-size:18px;margin:0}.section-head p{color:var(--muted);margin:3px 0 0;font-size:12px}.legend{display:flex;gap:13px;color:var(--muted);font-size:12px;flex-wrap:wrap}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}.dot.direct{background:#1767c0}.dot.standard{background:#18a077}.dot.leak{background:#fff;border:2px solid #c0392b}.overview svg{display:block;width:100%;height:330px;margin-top:8px}.browser{display:grid;grid-template-columns:330px minmax(0,1fr);border-top:0;min-height:760px}.browser aside{padding:20px;border-right:1px solid var(--line);min-width:0}.filters{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin:12px 0}.filters select{width:100%;padding:8px;border:1px solid var(--line);border-radius:7px;background:white;color:var(--ink)}.prompt-list{height:650px;overflow:auto;border-top:1px solid var(--line)}.prompt-row{display:block;width:100%;text-align:left;border:0;border-bottom:1px solid #e8ebef;background:transparent;padding:10px 6px;cursor:pointer;color:inherit}.prompt-row:hover,.prompt-row.active{background:#fff6cf}.prompt-row .row-top{display:flex;justify-content:space-between;gap:8px;font:12px ui-monospace,SFMono-Regular,Menlo,monospace}.prompt-row .snippet{font-size:12px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:3px}.tag{border:1px solid var(--line);border-radius:999px;padding:1px 5px;font-size:10px}.tag.leak{border-color:#dc9a92;color:#9d281e;background:#fff1ef}article{padding:22px;min-width:0}.empty{color:var(--muted);padding:30px}.dialogue{display:grid;grid-template-columns:1fr 1.65fr;gap:10px}.card{border:1px solid var(--line);background:white;border-radius:10px;padding:12px;min-width:0}.role{margin-bottom:5px}.response{display:flex;align-items:baseline;flex-wrap:wrap;gap:2px}.token{border:0;border-radius:4px;padding:2px 3px;background:transparent;color:inherit;font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;cursor:pointer}.token:hover{background:#fff0a8}.token.selected{background:var(--gold);box-shadow:0 0 0 1px #a77e00 inset}.toolbar{display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap;margin:13px 0}.toggle{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}.toggle button{border:0;background:white;padding:7px 12px;cursor:pointer;font-weight:650;color:#4a5360}.toggle button.active{color:white;background:#253246}.selection{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:#3d4653}.detail-grid{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(250px,.8fr);gap:11px}.stack{display:grid;gap:11px}.card h3{margin:0 0 8px}.heat,.line-chart{width:100%;height:auto;display:block}.ramp{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:11px;margin-top:5px}.ramp i{width:120px;height:9px;border-radius:6px;background:linear-gradient(90deg,#440154,#31688e,#35b779,#fde725)}.cell-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.cell-metrics div{background:var(--soft);border-radius:7px;padding:7px}.cell-metrics b{display:block;font-size:17px}.cell-metrics span{font-size:10px;color:var(--muted)}.top-table,.top-grid{width:100%;border-collapse:collapse;font:11px ui-monospace,SFMono-Regular,Menlo,monospace}.top-table td,.top-grid td,.top-grid th{padding:4px;border-bottom:1px solid #ebedf1}.top-table td:last-child{text-align:right;color:var(--muted)}.top-grid-wrap{overflow:auto;max-height:270px}.top-grid td{cursor:pointer;max-width:66px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.axis text{fill:#667080;font-size:10px}.axis path,.axis line{stroke:#cdd2da}footer{border-top:0;border-radius:0 0 18px 18px;padding:14px 24px;color:var(--muted);font-size:12px}#tooltip{position:fixed;z-index:30;pointer-events:none;opacity:0;background:#18212d;color:white;border-radius:6px;padding:7px 9px;font-size:12px;box-shadow:0 4px 15px #0003}@media(max-width:900px){main{padding:14px}.headline{grid-template-columns:repeat(2,minmax(0,1fr))}.browser{grid-template-columns:1fr}.browser aside{border-right:0;border-bottom:1px solid var(--line)}.prompt-list{height:280px}.detail-grid{grid-template-columns:1fr}}@media(max-width:620px){main{padding:0}header{border-radius:0;padding:20px}.headline{grid-template-columns:1fr}.overview,.browser aside,article{padding:16px}.browser,header,.overview,footer{border-left:0;border-right:0}.dialogue{grid-template-columns:1fr}.section-head{display:block}.legend{margin-top:8px}.overview svg{height:270px}.cell-metrics{grid-template-columns:1fr}.filters{grid-template-columns:1fr}footer{border-radius:0}}
.adapter-bar{display:flex;align-items:end;gap:18px;margin-top:20px;flex-wrap:wrap}.adapter-bar label{display:grid;gap:5px;font-size:11px;text-transform:uppercase;letter-spacing:.08em;font-weight:750;color:#596475}.adapter-bar select{min-width:180px;padding:9px 34px 9px 11px;border:1px solid #aeb6c2;border-radius:8px;background:#fff;color:var(--ink);font:650 14px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace}.sample-note{margin:0 0 8px;color:var(--muted);font-size:12px}@media(max-width:620px){.adapter-bar{display:grid;gap:10px}.sample-note{margin:0}}
'''


APP_JS = r'''
const state={index:null,adapter:'smile',summary:null,visible:[],detail:null,method:'jlens',layer:40,position:1,loadVersion:0};
const $=s=>document.querySelector(s); const pretty=t=>t==='\n'?'↵':t===' '?'␠':String(t).replaceAll('\n','↵');
const rankColor=r=>d3.interpolateViridis(Math.max(0,Math.min(1,(4-Math.log10(Math.max(1,Math.min(10000,r))))/4)));
const fmtRank=r=>r>=10000?`${Math.round(r/1000)}k`:r.toLocaleString();
const tooltip=$('#tooltip');
function cell(method,l,p){const d=state.detail,pi=d.positions.indexOf(p),li=d.layers.indexOf(l);if(pi<0||li<0)return null;const i=li*d.positions.length+pi,m=d.methods[method];return{rank:m.rank[i],mass:m.massPpm[i]/1e6,top1:m.top1[i]}}
async function init(){const r=await fetch('data/index.json');state.index=await r.json();const select=$('#adapter-filter');select.innerHTML=state.index.adapters.map(d=>`<option value="${escapeHtml(d.name)}">${escapeHtml(d.name)}</option>`).join('');const requested=new URLSearchParams(location.search).get('adapter');state.adapter=state.index.adapters.some(d=>d.name===requested)?requested:state.index.defaultAdapter;select.value=state.adapter;select.onchange=()=>loadAdapter(select.value,true);await loadAdapter(state.adapter,false)}
async function loadAdapter(adapter,updateUrl){const version=++state.loadVersion;state.adapter=adapter;state.summary=null;state.detail=null;$('#detail').innerHTML='<div class="empty">Loading saved examples…</div>';const entry=state.index.adapters.find(d=>d.name===adapter);const r=await fetch(entry.summary);const summary=await r.json();if(version!==state.loadVersion)return;state.summary=summary;renderAdapterContext();applyFilters();if(updateUrl){const url=new URL(location.href);if(adapter===state.index.defaultAdapter)url.searchParams.delete('adapter');else url.searchParams.set('adapter',adapter);history.replaceState(null,'',url)}const first=state.visible.find(d=>!d.leaked)||state.visible[0];await loadPrompt(first.promptId,version)}
function renderAdapterContext(){const meta=state.summary.meta,kept=meta.balance.keptByType;$('#target-name').textContent=state.adapter;$('#overview-count').textContent=meta.prompts;$('#sample-note').textContent=`${meta.prompts} diverse saved examples: ${kept.direct} direct + ${kept.standard} standard. Selected by text diversity only; lens scores do not affect selection.`;document.title=`J-Lens × Taboo: ${state.adapter} prompt browser`}
function applyFilters(){const type=$('#type-filter').value,sort=$('#sort').value;let v=state.summary.prompts.filter(d=>type==='all'||(type==='leaks'?d.leaked:d.promptType===type));v.sort(sort==='id'?(a,b)=>a.promptId.localeCompare(b.promptId):sort==='gain'?(a,b)=>(b.logRankGain??-99)-(a.logRankGain??-99):(a,b)=>a.displayOrder-b.displayOrder);state.visible=v;$('#count').textContent=`${v.length} of ${state.summary.prompts.length}`;renderList();drawScatter()}
function renderList(){const box=$('#prompt-list');box.innerHTML=state.visible.map(d=>`<button class="prompt-row ${state.detail?.promptId===d.promptId?'active':''}" data-id="${d.promptId}"><div class="row-top"><span>${d.promptId}</span><span>${d.leaked?'<span class="tag leak">leak</span>':`Δlog rank ${d.logRankGain>=0?'+':''}${d.logRankGain}`}</span></div><div class="snippet">${escapeHtml(d.output)}</div></button>`).join('');box.querySelectorAll('button').forEach(b=>b.onclick=()=>loadPrompt(b.dataset.id))}
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function loadPrompt(id,version=state.loadVersion){const meta=state.summary.prompts.find(d=>d.promptId===id);$('#detail').innerHTML='<div class="empty">Loading saved layer × position readout…</div>';const r=await fetch(meta.detail);const detail=await r.json();if(version!==state.loadVersion)return;state.detail=detail;state.layer=state.detail.defaultLayer;state.position=state.detail.defaultPosition;state.method='jlens';renderDetail();renderList();drawScatter()}
function renderDetail(){const d=state.detail;$('#detail').innerHTML=`<div class="dialogue"><div class="card"><div class="role">Human · ${d.promptType}</div><div>${escapeHtml(d.prompt)}</div></div><div class="card"><div class="role">Assistant ${d.leaked?'<span class="tag leak">literal secret leak</span>':''}</div><div id="response" class="response"></div></div></div><div class="toolbar"><div class="toggle"><button data-m="jlens" class="active">J-Lens</button><button data-m="logit_lens">Logit Lens</button></div><div id="selection" class="selection"></div></div><div class="detail-grid"><div class="stack"><div class="card"><h3>Secret rank · source layer × response position</h3><svg id="heat" class="heat"></svg><div class="ramp"><span>rank 10k+</span><i></i><span>rank 1</span></div></div><div class="card"><h3>Top decoded token · sampled layers</h3><div id="top-grid" class="top-grid-wrap"></div></div></div><div class="stack"><div class="card"><h3>Selected cell</h3><div id="cell-metrics" class="cell-metrics"></div><table id="top-table" class="top-table"></table></div><div class="card"><h3>Rank by layer · selected response token</h3><svg id="layer-chart" class="line-chart"></svg></div><div class="card"><h3>Rank by response position · selected layer</h3><svg id="pos-chart" class="line-chart"></svg></div></div></div>`;const response=$('#response');response.innerHTML=d.tokens.map(x=>`<button class="token" data-p="${x.position}">${escapeHtml(pretty(x.token))}</button>`).join('');response.querySelectorAll('button').forEach(b=>b.onclick=()=>{state.position=+b.dataset.p;updateDetail()});document.querySelectorAll('.toggle button').forEach(b=>b.onclick=()=>{state.method=b.dataset.m;document.querySelectorAll('.toggle button').forEach(x=>x.classList.toggle('active',x===b));updateDetail()});updateDetail()}
function updateDetail(){drawHeat();drawTopGrid();drawSelected();drawLine('#layer-chart',state.detail.layers,(m,x)=>cell(m,x,state.position),'layer');drawLine('#pos-chart',state.detail.positions,(m,x)=>cell(m,state.layer,x),'position');document.querySelectorAll('.token').forEach(b=>b.classList.toggle('selected',+b.dataset.p===state.position))}
function drawHeat(){const d=state.detail,svg=d3.select('#heat'),node=svg.node(),W=Math.max(360,node.parentElement.clientWidth-24),H=Math.max(300,Math.min(440,W*.56)),mg={t:8,r:10,b:42,l:48};svg.attr('viewBox',`0 0 ${W} ${H}`).selectAll('*').remove();const x=d3.scaleBand().domain(d.positions).range([mg.l,W-mg.r]).padding(.035),y=d3.scaleBand().domain([...d.layers].reverse()).range([mg.t,H-mg.b]).padding(.025);const values=[];d.layers.forEach(l=>d.positions.forEach(p=>values.push({l,p,...cell(state.method,l,p)})));svg.append('g').selectAll('rect').data(values).join('rect').attr('x',v=>x(v.p)).attr('y',v=>y(v.l)).attr('width',x.bandwidth()).attr('height',y.bandwidth()).attr('fill',v=>rankColor(v.rank)).attr('stroke',v=>v.l===state.layer&&v.p===state.position?'#fff':'none').attr('stroke-width',2).style('cursor','pointer').on('click',(_,v)=>{state.layer=v.l;state.position=v.p;updateDetail()}).on('mousemove',(e,v)=>showTip(e,`layer ${v.l} · position ${v.p}<br>rank <b>${fmtRank(v.rank)}</b> · top-1 <b>${escapeHtml(pretty(v.top1))}</b>`)).on('mouseleave',hideTip);svg.append('g').attr('class','axis').attr('transform',`translate(0,${H-mg.b})`).call(d3.axisBottom(x).tickValues(d.positions.filter(p=>p===1||p%4===0)));svg.append('g').attr('class','axis').attr('transform',`translate(${mg.l},0)`).call(d3.axisLeft(y).tickValues(d.layers.filter(l=>l%5===0||l===62)));svg.append('line').attr('x1',mg.l).attr('x2',W-mg.r).attr('y1',y(d.frozenLayer)+y.bandwidth()/2).attr('y2',y(d.frozenLayer)+y.bandwidth()/2).attr('stroke','#fff').attr('stroke-dasharray','4,3');svg.append('text').attr('x',(mg.l+W-mg.r)/2).attr('y',H-6).attr('text-anchor','middle').attr('fill','#667080').attr('font-size',11).text('response token position →');svg.append('text').attr('transform','rotate(-90)').attr('x',-(mg.t+H-mg.b)/2).attr('y',13).attr('text-anchor','middle').attr('fill','#667080').attr('font-size',11).text('source layer →')}
function drawTopGrid(){const d=state.detail,sampled=d.layers.filter(l=>l>=20&&(l%4===0||l===62)).reverse();let h='<table class="top-grid"><thead><tr><th>layer</th>'+d.positions.map(p=>`<th>${p}</th>`).join('')+'</tr></thead><tbody>';sampled.forEach(l=>{h+=`<tr><th>${l}</th>`;d.positions.forEach(p=>{const v=cell(state.method,l,p),active=l===state.layer&&p===state.position;h+=`<td data-l="${l}" data-p="${p}" title="${escapeHtml(v.top1)} · secret rank ${v.rank}" style="background:${active?'#f4ca21':rankColor(v.rank)};color:${v.rank<=10?'#17202a':'white'}">${escapeHtml(pretty(v.top1).slice(0,10))}</td>`});h+='</tr>'});h+='</tbody></table>';$('#top-grid').innerHTML=h;document.querySelectorAll('#top-grid td').forEach(td=>td.onclick=()=>{state.layer=+td.dataset.l;state.position=+td.dataset.p;updateDetail()})}
function drawSelected(){const d=state.detail,v=cell(state.method,state.layer,state.position),other=cell(state.method==='jlens'?'logit_lens':'jlens',state.layer,state.position),tok=d.tokens.find(x=>x.position===state.position)?.token??'';$('#selection').textContent=`${state.method==='jlens'?'J-Lens':'Logit Lens'} · layer ${state.layer} · position ${state.position} · after “${pretty(tok)}”`;$('#cell-metrics').innerHTML=`<div><b>#${fmtRank(v.rank)}</b><span>${escapeHtml(d.adapter)} rank</span></div><div><b>${(100*v.mass).toFixed(2)}%</b><span>probability mass</span></div><div><b>#${fmtRank(other.rank)}</b><span>other lens rank</span></div>`;const isDefault=state.layer===d.defaultLayer&&state.position===d.defaultPosition,top=isDefault?d.defaultTop5[state.method]:[{token:v.top1,probability:null}];$('#top-table').innerHTML='<tbody>'+top.map((x,i)=>`<tr><td>${i+1}</td><td>${escapeHtml(pretty(x.token))}</td><td>${x.probability==null?'top-1':(100*x.probability).toFixed(2)+'%'}</td></tr>`).join('')+'</tbody>'}
function drawLine(sel,xs,getter,kind){const svg=d3.select(sel),node=svg.node(),W=Math.max(280,node.parentElement.clientWidth-24),H=190,mg={t:10,r:10,b:34,l:48};svg.attr('viewBox',`0 0 ${W} ${H}`).selectAll('*').remove();const x=d3.scaleLinear().domain(d3.extent(xs)).range([mg.l,W-mg.r]),y=d3.scaleLog().domain([1,10000]).range([mg.t,H-mg.b]).clamp(true);svg.append('g').attr('class','axis').attr('transform',`translate(0,${H-mg.b})`).call(d3.axisBottom(x).ticks(W<400?4:6));svg.append('g').attr('class','axis').attr('transform',`translate(${mg.l},0)`).call(d3.axisLeft(y).tickValues([1,10,100,1000,10000]).tickFormat(v=>v===10000?'10k':v));[['jlens','#1767c0'],['logit_lens','#e07a27']].forEach(([m,c])=>{const vals=xs.map(z=>({x:z,y:Math.min(10000,getter(m,z).rank)}));svg.append('path').datum(vals).attr('fill','none').attr('stroke',c).attr('stroke-width',2).attr('d',d3.line().x(v=>x(v.x)).y(v=>y(v.y)))});const mark=kind==='layer'?state.layer:state.position;svg.append('line').attr('x1',x(mark)).attr('x2',x(mark)).attr('y1',mg.t).attr('y2',H-mg.b).attr('stroke','#8b2d83').attr('stroke-dasharray','4,3');if(kind==='layer')svg.append('line').attr('x1',x(state.detail.frozenLayer)).attr('x2',x(state.detail.frozenLayer)).attr('y1',mg.t).attr('y2',H-mg.b).attr('stroke','#555').attr('stroke-dasharray','3,3');svg.append('text').attr('x',W-8).attr('y',14).attr('text-anchor','end').attr('fill','#1767c0').attr('font-size',10).text('J-Lens');svg.append('text').attr('x',W-8).attr('y',27).attr('text-anchor','end').attr('fill','#e07a27').attr('font-size',10).text('Logit Lens')}
function drawScatter(){if(!state.summary)return;const svg=d3.select('#scatter'),node=svg.node(),W=Math.max(320,node.parentElement.clientWidth-52),H=node.clientHeight||330,mg={t:14,r:18,b:48,l:64};svg.attr('viewBox',`0 0 ${W} ${H}`).selectAll('*').remove();const x=d3.scaleLog().domain([1,250000]).range([mg.l,W-mg.r]),y=d3.scaleLog().domain([1,250000]).range([mg.t,H-mg.b]);const ticks=[1,10,100,1000,10000,100000];svg.append('rect').attr('x',mg.l).attr('y',mg.t).attr('width',W-mg.l-mg.r).attr('height',H-mg.t-mg.b).attr('fill','#fff').attr('stroke','#d8dde5');svg.append('path').attr('d',d3.line()([[x(1),y(1)],[x(250000),y(250000)]] )).attr('stroke','#8a929e').attr('stroke-dasharray','4,4');svg.append('g').attr('class','axis').attr('transform',`translate(0,${H-mg.b})`).call(d3.axisBottom(x).tickValues(ticks).tickFormat(v=>v>=1000?`${v/1000}k`:v));svg.append('g').attr('class','axis').attr('transform',`translate(${mg.l},0)`).call(d3.axisLeft(y).tickValues(ticks).tickFormat(v=>v>=1000?`${v/1000}k`:v));svg.append('text').attr('x',(mg.l+W-mg.r)/2).attr('y',H-7).attr('text-anchor','middle').attr('fill','#667080').text('Logit Lens vocabulary rank →');svg.append('text').attr('transform','rotate(-90)').attr('x',-(mg.t+H-mg.b)/2).attr('y',14).attr('text-anchor','middle').attr('fill','#667080').text('J-Lens vocabulary rank →');const current=state.detail?.promptId;svg.append('g').selectAll('circle').data(state.visible).join('circle').attr('cx',d=>x(d.lRank)).attr('cy',d=>y(d.jRank)).attr('r',d=>d.promptId===current?6:4).attr('fill',d=>d.promptType==='direct'?'#1767c0':'#18a077').attr('fill-opacity',.68).attr('stroke',d=>d.leaked?'#c0392b':d.promptId===current?'#17202a':'#fff').attr('stroke-width',d=>d.promptId===current||d.leaked?2:1).style('cursor','pointer').on('click',(_,d)=>loadPrompt(d.promptId)).on('mousemove',(e,d)=>showTip(e,`<b>${d.promptId}</b> · ${d.promptType}${d.leaked?' · leak':''}<br>J-Lens #${fmtRank(d.jRank)} · Logit #${fmtRank(d.lRank)}`)).on('mouseleave',hideTip)}
function showTip(e,h){tooltip.innerHTML=h;tooltip.style.opacity=1;tooltip.style.left=`${e.clientX+12}px`;tooltip.style.top=`${e.clientY+12}px`}function hideTip(){tooltip.style.opacity=0}
$('#type-filter').onchange=applyFilters;$('#sort').onchange=applyFilters;window.addEventListener('resize',()=>{drawScatter();if(state.detail)updateDetail()});init().catch(e=>{$('#detail').innerHTML=`<div class="empty">Could not load saved data: ${escapeHtml(e.message)}</div>`;console.error(e)});
'''


def build_adapter(
    root: Path,
    output: Path,
    adapter: str,
    run_id: str,
    behavior: dict[str, dict],
) -> dict:
    """Build one independently sampled adapter dataset and return its index row."""
    adapter_output = output / "data" / "adapters" / adapter
    (adapter_output / "prompts").mkdir(parents=True)
    run_cells = root / "artifacts" / "lens_outputs" / run_id / "test_cells"
    records: list[tuple[dict, dict]] = []
    position_paths = sorted(run_cells.glob(f"*__{adapter}.positions.parquet"))
    for index, position_path in enumerate(position_paths, start=1):
        prompt_id = position_path.name.removesuffix(f"__{adapter}.positions.parquet")
        aggregate_path = run_cells / f"{prompt_id}__{adapter}.aggregate.parquet"
        summary, detail = compact_prompt(
            position_path, aggregate_path, behavior[prompt_id], adapter
        )
        records.append((summary, detail))
        if index == 1 or index % 50 == 0:
            print(f"{adapter}: processed {index}/200 ({prompt_id})", flush=True)
    if len(records) != 200:
        raise RuntimeError(
            f"Expected 200 position files for {adapter}, found {len(records)}"
        )
    balanced, balance_stats = deduplicate_and_balance(records)
    summaries = [summary for summary, _ in balanced]
    for summary, detail in balanced:
        write_json(output / summary["detail"], detail)
    valid = [row for row in summaries if not row["leaked"]]
    summary_path = f"data/adapters/{adapter}/summary.json"
    write_json(
        output / summary_path,
        {
            "meta": {
                "adapter": adapter,
                "runId": run_id,
                "frozenLayer": FROZEN_LAYER,
                "mask": "global_emitted_ids",
                "prompts": len(summaries),
                "validPrompts": len(valid),
                "literalLeaks": len(summaries) - len(valid),
                "balance": balance_stats,
                "baseModelRevision": behavior[next(iter(behavior))][
                    "base_model_revision"
                ],
                "jlensRevision": behavior[next(iter(behavior))]["jlens_revision"],
            },
            "prompts": summaries,
        },
    )
    return {
        "name": adapter,
        "summary": summary_path,
        "prompts": len(summaries),
        "literalLeaks": len(summaries) - len(valid),
    }


def build_site(
    root: Path,
    output: Path,
    adapters: list[str] | None,
    run_id: str,
) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    behavior_path = root / "data" / "raw_outputs" / run_id / "test_behavior_generations.jsonl"
    behavior_by_adapter = read_all_behavior(behavior_path)
    selected_adapters = sorted(adapters or behavior_by_adapter)
    unknown = sorted(set(selected_adapters) - set(behavior_by_adapter))
    if unknown:
        raise RuntimeError(f"Adapters missing from saved behavior: {unknown}")
    if not selected_adapters:
        raise RuntimeError("No adapters selected")
    index_rows = [
        build_adapter(root, output, adapter, run_id, behavior_by_adapter[adapter])
        for adapter in selected_adapters
    ]
    default_adapter = (
        DEFAULT_ADAPTER if DEFAULT_ADAPTER in selected_adapters else selected_adapters[0]
    )
    index_payload = {
        "meta": {
            "runId": run_id,
            "frozenLayer": FROZEN_LAYER,
            "mask": "global_emitted_ids",
            "examplesPerAdapter": EXAMPLES_PER_TYPE * 2,
            "selectionUsesLensOutcome": False,
        },
        "defaultAdapter": default_adapter,
        "adapters": index_rows,
    }
    write_json(output / "data" / "index.json", index_payload)
    (output / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (output / "styles.css").write_text(STYLES_CSS.strip() + "\n", encoding="utf-8")
    (output / "app.js").write_text(APP_JS.strip() + "\n", encoding="utf-8")
    (output / ".nojekyll").write_text("", encoding="utf-8")
    (output / "README.md").write_text(
        f"""# J-Lens x Taboo: all-adapter prompt browser

Static GitHub Pages site generated from saved run `{run_id}`. It covers all
{len(selected_adapters)} saved Taboo adapters, with `{default_adapter}` selected
by default. For every adapter, exact duplicate answers are collapsed within each
prompt type and the browser keeps 50 lexically diverse examples. It starts from
25 direct plus 25 standard and fills any uniqueness shortfall from the other
prompt type.

The compact published data include layer-by-position target ranks, probability
masses, and top-1 decoded tokens for J-Lens and Logit Lens. Distinct literal
own-secret leaks are retained and visible. Sample selection does not use lens
outcomes. No model weights, credentials, or hidden activations are published.
""",
        encoding="utf-8",
    )
    total = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(
        json.dumps(
            {
                "output": str(output),
                "files": sum(1 for p in output.rglob("*") if p.is_file()),
                "bytes": total,
                "defaultAdapter": default_adapter,
                "adapters": index_rows,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--adapter",
        action="append",
        dest="adapters",
        help="Build only this adapter; repeat for more. Defaults to all saved adapters.",
    )
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    build_site(args.root, args.output, args.adapters, args.run_id)


if __name__ == "__main__":
    main()
