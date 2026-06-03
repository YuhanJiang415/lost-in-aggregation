# Project homepage — *Lost in Aggregation*

A self-contained static project page for the benchmark
**Lost in Aggregation: A Multi-Scale Diagnostic Benchmark for LLM Spatial Navigation**
(ACM SIGSPATIAL 2026, Benchmark Track).

```
docs/
├── index.html              # the page (inlined CSS + tiny JS, no build step)
├── assets/
│   ├── figures/            # paper figures (PNG)
│   ├── mazes/              # rendered maze preview thumbnails (one per size)
│   ├── gifs/               # animated navigation clips (robot walking the maze)
│   └── robot.png           # robot sprite (rasterized from robot.svg)
└── data/
    └── summary.json        # corpus manifest
```

## Preview locally

```bash
cd docs
python3 -m http.server 8000
# open http://localhost:8000
```

(Opening `index.html` directly via `file://` also works.)

## Publish on GitHub Pages

1. Push this repo to GitHub.
2. Repo **Settings → Pages → Build and deployment**:
   - **Source:** *Deploy from a branch*
   - **Branch:** `main`, folder **`/docs`**.
3. The page goes live at `https://<user>.github.io/<repo>/`.

A `.nojekyll` file is included so GitHub serves the assets as-is.

## Before going live — fill in the placeholders

Search `index.html` for `TODO` and `href="#"` and replace with real URLs:

- **Paper PDF / arXiv** links in the hero.
- **GitHub repository** link (hero + Code section).
- **Benchmark data** download links in the Data table.
- **DOI** in the BibTeX block.

### Hosting the data (~100 MB total)

The maze JSON files are large (the 30×30 file alone is ~51 MB). Do **not**
commit them to the Pages branch. Instead host them as **GitHub Release
assets** (or on Zenodo / Hugging Face Datasets for an archival DOI) and point
the download links in the Data table at those URLs.

## Regenerating the maze thumbnails

```bash
python3 scripts/render_maze_homepage_previews.py
```

Reads `mazes/mazes_s{N}.json` and writes `docs/assets/mazes/maze_s{N}.png`.

## Regenerating the navigation GIFs

```bash
python3 scripts/render_nav_gifs.py
```

Animates the robot sprite over the decision-point-density heatmap and writes
four clips to `docs/assets/gifs/`:

| File | Scenario | Level |
|------|----------|-------|
| `nav_success.gif`        | robot solves the maze            | — |
| `nav_err_wrong_turn.gif` | wrong branch at a junction → dead-end | Meso |
| `nav_err_wall_hit.gif`   | robot lunges into a wall         | Fine |
| `nav_err_teleport.gif`   | robot jumps to a non-adjacent cell | Fine |

All four use the same maze (`maze_s5_medium_000`) so the correct route is a
fixed reference. The robot sprite is rasterized from `robot.svg` (the script
expects `docs/assets/robot.png`; regenerate it with a headless browser or any
SVG→PNG tool if you swap the SVG). Tunables (`SUB`, `FPS`, `HOLD_*`) sit at the
top of the script.
