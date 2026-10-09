# Lost in Aggregation

**A Multi-Scale Diagnostic Benchmark for LLM Spatial Navigation**
Yuhan Jiang · Peng Luo · Liqiu Meng — Preprint, under review

> 🌐 **Project page:** https://yuhanjiang415.github.io/lost-in-aggregation/

We ask not merely *whether* LLMs fail at maze navigation but *where* in the
spatial-cognition pipeline they get lost. The benchmark decomposes navigation
into three cognitive levels — **Fine** (local passability), **Meso** (junction
topology), and **Macro** (global goal direction) — and probes each in
isolation across a systematic size sweep. The central finding: end-to-end
navigation collapses long before any single competence does, so the binding
constraint is the cross-scale *aggregation* of individually available skills
over a long sequential plan.

## Benchmark data

1,050 topology-annotated mazes across seven effective sizes (3×3 → 30×30) and
three difficulty tiers (50 mazes per size × difficulty cell). Each maze ships
with per-cell passable directions, cell types, the unique shortest path, and
the goal-reaching branch at every junction.

The maze JSON files (~100 MB total) are published as **GitHub Release assets**
under tag [`v0.1`](https://github.com/YuhanJiang415/lost-in-aggregation/releases/tag/v0.1),
one file per size. See the project page's *Benchmark data* section for the
per-file download table and the JSON schema.

```
mazes_s{3,5,7,10,15,20,30}.json   # 150 mazes each
summary.json                       # corpus manifest (included in the Release)
```

### Real street-network routes

The external validation uses **90 walking routes with 663 decision junctions**
on OpenStreetMap pedestrian networks in Munich, Tokyo, and Toronto (10 routes
per city at each of three scales: 3–5, 6–8, and 9–12 decision junctions). The
frozen release (`LIA-Road-TB` v0.2.0, ~9 MB of JSON) lives in this repository
under [`road_network/`](road_network/): compact model-input graphs, reference
shortest paths, per-junction answer keys, and task files. See
[`road_network/README.md`](road_network/README.md) for the schema and scoring
rules, or grab [`LIA-Road-TB_v0.2.0.zip`](road_network/LIA-Road-TB_v0.2.0.zip).

Map data © OpenStreetMap contributors (ODbL); origin–destination pairs from
[TurnBack](https://github.com/bghjmn32/EMNLP2025_Turnback).

## Repository layout

```
docs/           # project homepage (GitHub Pages, served from /docs)
road_network/   # real street-network routes (LIA-Road-TB v0.2.0)
```

The maze generator, input encoders, and evaluation/figure code are not yet
published; they will be released here. The benchmark data is available now: the
mazes via [Releases](https://github.com/YuhanJiang415/lost-in-aggregation/releases/tag/v0.1),
the street-network routes in [`road_network/`](road_network/).

## Citation

```bibtex
@misc{jiang2026lostinaggregation,
  title        = {Lost in Aggregation: A Multi-Scale Diagnostic Benchmark
                  for LLM Spatial Navigation},
  author       = {Jiang, Yuhan and Luo, Peng and Meng, Liqiu},
  year         = {2026},
  note         = {Preprint, under review},
  howpublished = {\url{https://yuhanjiang415.github.io/lost-in-aggregation/}}
}
```

Full evaluation and harness code will be released here.
