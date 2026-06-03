# Lost in Aggregation

**A Multi-Scale Diagnostic Benchmark for LLM Spatial Navigation**
Yuhan Jiang · Peng Luo · Liqiu Meng — ACM SIGSPATIAL 2026 (Benchmark Track)

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
summary.json                       # corpus manifest (in this repo)
```

## Repository layout

```
docs/              # project homepage (GitHub Pages, served from /docs)
mazes/summary.json # corpus manifest (full JSON shipped via Releases)
maze_generator/    # maze generation + topology annotation
input_formatter/   # the four input encoders (Words / Coordinate / Map / Picture)
analysis/          # results loading helpers
scripts/           # figure, thumbnail, and animation generation
```

## Citation

```bibtex
@inproceedings{jiang2026lostinaggregation,
  title     = {Lost in Aggregation: A Multi-Scale Diagnostic Benchmark
               for LLM Spatial Navigation},
  author    = {Jiang, Yuhan and Luo, Peng and Meng, Liqiu},
  booktitle = {Proceedings of the 34th ACM SIGSPATIAL International
               Conference on Advances in Geographic Information Systems
               (SIGSPATIAL '26)},
  year      = {2026},
  publisher = {ACM}
}
```

Full evaluation and harness code will be released here.
