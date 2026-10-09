# LIA-Road-TB — real street-network routes (v0.2.0)

The frozen real-road dataset used for the external validation in *Lost in
Aggregation* (Section 5.4): **90 walking routes** with **663 decision
junctions** on OpenStreetMap pedestrian networks in Munich, Tokyo, and Toronto.
This is the exact release the reported model calls were run on.

Everything is plain JSON / JSONL; no code is needed to read it. For a one-file
download use [`LIA-Road-TB_v0.2.0.zip`](LIA-Road-TB_v0.2.0.zip) (1.2 MB).

## Design

| | Small | Medium | Large | Total |
|---|---:|---:|---:|---:|
| Decision junctions per route | 3–5 | 6–8 | 9–12 | |
| Routes (10 per city) | 30 | 30 | 30 | **90** |
| Decision junctions | 121 | 216 | 326 | **663** |
| Corridor nodes, median (range) | 36.5 (15–239) | 88 (28–303) | 139.5 (69–359) | |
| Reference route length, median | 280 m | 412 m | 507 m | |

Every city × scale cell holds 10 routes: 5 **representative** (fixed-seed
selection) and 5 **diagnostic** (enriched, before any model was run, for
junctions where a bearing-greedy heuristic goes wrong). Routes in the same city
share at most 30% of their reference OSM edges.

Origin–destination pairs come from the
[TurnBack](https://github.com/bghjmn32/EMNLP2025_Turnback) route corpus (commit
`561feb7`). The street graphs, reference shortest paths, decision junctions,
and scale strata were all rebuilt independently with OSMnx; TurnBack's
easy/medium/hard tiers are kept as provenance only and are **not** the scale
labels used here.

## Layout

```
road_network/
├── route_index.csv            # one row per route: city, scale, stratum, sizes, file paths
├── LIA-Road-TB_v0.2.0.zip     # the v0.2.0 folder + route_index.csv + this README
└── v0.2.0/
    ├── routes/LIA-Road-TB-NNN.json        # 90 full records (graph, oracle, decisions, metrics, provenance)
    ├── inputs/LIA-Road-TB-NNN.graph.json  # 90 compact graphs — what the model reads
    ├── labels/LIA-Road-TB-NNN.labels.json # 90 answer keys (reference path + per-junction options)
    ├── labels/c2_junction_labels.jsonl    # 663 acceptable-action sets, keyed by task_id
    ├── tasks/c1_routes.jsonl              # 90 one-shot route-planning tasks
    ├── tasks/c2_junctions.jsonl           # 663 isolated junction-choice tasks
    ├── tasks/c3_hybrid_episodes.jsonl     # 90 step-wise (junction-delegation) episodes
    ├── manifest.jsonl                     # per-route metrics + SHA-256 of each route file
    ├── dataset_card.json
    └── audit_report.json
```

IDs run `LIA-Road-TB-001` … `090`, ordered by city (Munich 001–030, Tokyo
031–060, Toronto 061–090); within a city, large routes come first, then
medium, then small. Use `route_index.csv` to filter.

## The model input

`inputs/*.graph.json` is an anonymized corridor graph in local metric
coordinates — no street names, no latitude/longitude:

```jsonc
{
  "schema": "physical-decision-graph-v2",
  "dataset_id": "LIA-Road-TB-045",
  "mode": "walking",
  "node_id_rule": "zero-based row index",
  "node_fields": ["x_m", "y_m"],
  "nodes": [[-207.1, -142.5], ...],        // node i = row i, metres
  "edge_fields": ["from", "to", "length_m"],
  "edges": [[0, 1, 37.1], ...],            // directed
  "start": 42,
  "goal": 22
}
```

The corridor keeps every node `v` with
`d(start, v) + d(v, goal) ≤ 1.5 ×` the reference shortest-path length, plus a
one-hop neighbourhood. Node ids are local to a route and cannot be joined
across routes. Directions are derived from the coordinates; path cost uses
`length_m`.

## Tasks and how they are scored

| Paper name | File | Items | Correct when |
|---|---|---:|---|
| Fine — neighbour identification | derived from `inputs/` at each junction's `current` node | 663 | the predicted neighbour set equals the node's out-neighbours |
| Meso × Macro — isolated junction choice | `tasks/c2_junctions.jsonl` | 663 | the chosen `action_id` is in `acceptable_action_ids` (`labels/c2_junction_labels.jsonl`) |
| One-shot route planning | `tasks/c1_routes.jsonl` | 90 | every step is a real edge and the path reaches the goal under the execution policy |

`tasks/c3_hybrid_episodes.jsonl` defines a step-wise variant in which the model
is queried only at junctions. It is shipped for completeness; no results for it
are reported in the paper.

A junction's options are physical exits, excluding the edge the walker came in
on. Each option has a cost `q` = length of the outgoing connection + shortest
remaining distance to the goal. Options within 5 m or 1% of the best `q` are
all acceptable, so a junction can have more than one correct answer. Chance for
a junction is therefore `n_acceptable / n_actions`.

The full per-junction detail (bearing, relative turn, `q_m`, which options are
acceptable) is in `labels/*.labels.json` and `routes/*.json` under `decisions`.
Those two files use string node ids (`N0043`); the model-facing files use
zero-based integers, where `N0043` ↔ `42`.

## Known limits

- `audit_report.json` is the pre-experiment structural audit, kept unmodified.
  `passed: true` means the structure checks passed. `formal_release_ready:
  false` and `model_gate_status: not_run` record its state before any model
  call; they do not mean the experiments were not run.
- The same report shows the token-balance check failed and was not enforced:
  decision count and estimated input tokens are correlated (Spearman 0.77). Route
  scale is therefore confounded with input length, and scale trends should be
  read with that in mind.
- Three cities cannot support claims about city-level generalization.
- This folder holds the frozen release only. The upstream material (180-route
  source pool, 136 candidates, raw OSMnx GraphML, Overpass cache, ~830 MB) and
  the model outputs are not in the repository. Online OSM changes over time, so
  re-downloading will not reproduce these graphs byte for byte — use the frozen
  files.

## Attribution

Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
available under the Open Database License (ODbL); the graphs here are a derived
database and carry the same terms. Please also cite TurnBack (Luo et al., 2025)
for the origin–destination pairs and OSMnx (Boeing, 2017) for network
extraction, alongside the paper.
