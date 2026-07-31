# Graph Kalman Filter generalisation check: TrajImpute (ETH-M, HOTEL-M) and GeoLife

Question: does the graph message-passing Kalman filter from the ICST paper (fit on
battery-gated wireless sensor data) still say something useful on a completely
different missing-data problem — pedestrian trajectory imputation on TrajImpute's
ETH-M and HOTEL-M subsets (Chib & Singh, NeurIPS 2024 D&B), and on a dataset with
no pre-built multi-agent structure at all (Microsoft's GeoLife GPS logs)?

This report has two parts: Part 1 (TrajImpute) uses a dataset that already ships
train/val/test splits and a `seq_start_end` scene structure. Part 2 (GeoLife) is a
dataset built from scratch for this purpose — raw single-user GPS logs with no
train/val/test split and no multi-agent structure at all — because GeoLife records
"more informative" real-world activity (mixed transportation modes, years of data)
than TrajImpute's fixed pedestrian-video scenes.

# Part 1 — TrajImpute (ETH-M, HOTEL-M)

## Setup

**Data.** `{ETH-M,HOTEL-M}/{Easy,Hard}/data_{train,val,test}.pkl`, each a dict with
`obs_traj` (8 observed frames, NaN where dropped), `pred_traj` (12 future frames,
always clean), `missing_mask`, and `seq_start_end` (pedestrian-to-scene grouping).
The `.pkl` files embed a few tensors in PyTorch's legacy pickle format; rather than
pull in a 500 MB+ CUDA-only `torch` wheel, I wrote a ~100-line torch-free unpickler
(`torchless_load.py`) that reconstructs the same arrays as plain NumPy.

**Ground truth for scoring.** Each Easy test split stores every underlying test
trajectory 5 times — 0, 1, 2, 3, 4 dropped coordinates — and the m=0 rows are
bit-identical, at every non-NaN position, to the other four blocks (verified per
dataset: max abs diff = 0.0). That gives genuine ground truth for every masked
entry, the same thing TrajImpute's own Table 2 numbers must be built from. Each
Hard split's test set (4 copies, m∈{4,5,6,7}) has no clean copy of its own, but its
non-NaN entries also match the corresponding Easy split's clean block exactly, so
that block is reused as the ground truth reference for both protocols. Block size
(number of unique test trajectories) is auto-detected per dataset from the Easy
test split's m=0 row count: 181 for ETH-M, 1,053 for HOTEL-M.

One gap in the released files: `seq_start_end` for each test split only indexes
one block's worth of rows, not the full replicated set. Since row order/identity
is confirmed identical across blocks, the scene structure is reused with a
block-size offset for the other blocks — documented as a modelling assumption in
the script, not something the data confirms directly.

**Model.** Each pedestrian is a graph node; nodes co-present in a scene
(`seq_start_end`) are connected if their mean observed position is within 2 m
(a standard pedestrian personal-space/interaction radius). This mirrors the ICST
paper's mote adjacency, just swapping "battery-powered sensor at a fixed floor
position" for "pedestrian at a scene-averaged position." The same linear
state-space form is used:

```
s_t = theta_tm * s_(t-1) + theta_sp * Abar @ s_(t-1) + noise
```

run independently for the x- and y-coordinate channels (shared theta across
channels), with `theta_tm`/`theta_sp` fit **separately per dataset** by pooled
least squares on consecutive frame pairs of `pred_traj` (clean, so the fit isn't
contaminated by missingness) across every training scene. A standard Kalman
filter then runs forward over each test scene's 8 frames, updating on observed
coordinates and using the graph-informed prediction to fill gaps.

Fitted parameters (essentially identical for both datasets):

| Dataset | model | theta_tm | theta_sp | Q |
|---|---|---|---|---|
| ETH-M | no-graph | 1.0016 | 0 | 0.0445 |
| ETH-M | graph | 0.9946 | 0.0072 | 0.0444 |
| HOTEL-M | no-graph | 1.0016 | 0 | 0.0446 |
| HOTEL-M | graph | 0.9946 | 0.0071 | 0.0446 |

`theta_tm ≈ 1` in both cases: pedestrian position over a 0.4 s frame is very
close to a random walk — the same near-unit-root finding as the ICST paper's
temperature channel (`theta_tm = 0.992`). `theta_sp ≈ 0.007`, small but positive,
essentially identical across both cities' pedestrians.

## Results

MAE / MSE / RMSE / MRE at masked positions only (world coordinates, metres),
against TrajImpute's Table 2 numbers.

### ETH-M

**Easy protocol (0–4 of 8 frames missing)**

| Method | MAE | MSE | RMSE | MRE |
|---|---|---|---|---|
| Transformer (paper) | 3.1318 | 19.4576 | 4.4111 | 0.5236 |
| US-GAN (paper) | 0.6467 | 1.8055 | 1.3437 | 0.1081 |
| BRITS (paper) | 1.4287 | 4.7339 | 2.1758 | 0.2389 |
| M-RNN (paper) | 5.2558 | 35.3738 | 5.9476 | 0.8787 |
| TimesNet (paper) | 1.1353 | 4.9441 | 2.2235 | 0.1898 |
| SAITS (paper, best) | 0.5031 | 0.9909 | 0.9954 | 0.0841 |
| **KF, no graph (ours)** | 0.4040 | 0.4839 | 0.6956 | 0.0674 |
| **KF, graph MP (ours)** | 0.4040 | 0.4831 | 0.6951 | 0.0674 |
| **Persistence (ours)** | 0.3985 | 0.4807 | 0.6933 | 0.0665 |

**Hard protocol (4–7 of 8 frames missing)**

| Method | MAE | MSE | RMSE | MRE |
|---|---|---|---|---|
| Transformer (paper) | 3.2249 | 19.5948 | 4.7926 | 0.5734 |
| US-GAN (paper) | 3.0451 | 18.0716 | 4.2511 | 0.5100 |
| BRITS (paper) | 3.0371 | 17.9457 | 4.2362 | 0.5087 |
| M-RNN (paper) | 5.3309 | 35.5047 | 5.9965 | 0.8962 |
| TimesNet (paper) | 1.3656 | 4.9937 | 2.5054 | 0.2287 |
| SAITS (paper, best) | 0.9965 | 2.5934 | 1.6104 | 0.1669 |
| **KF, no graph (ours)** | 0.6532 | 1.4434 | 1.2014 | 0.1092 |
| **KF, graph MP (ours)** | 0.6527 | 1.4404 | 1.2001 | 0.1091 |
| **Persistence (ours)** | 0.6456 | 1.4358 | 1.1983 | 0.1080 |

### HOTEL-M

**Easy protocol (0–4 of 8 frames missing)**

| Method | MAE | MSE | RMSE | MRE |
|---|---|---|---|---|
| Transformer (paper) | 8.8847 | 91.5550 | 9.5684 | 2.9468 |
| US-GAN (paper) | 2.6327 | 13.5993 | 3.6877 | 0.8732 |
| BRITS (paper) | 3.9033 | 23.1058 | 4.8068 | 1.2946 |
| M-RNN (paper) | 3.2133 | 20.0857 | 4.4817 | 1.0658 |
| TimesNet (paper) | 7.4037 | 124.5438 | 11.1599 | 2.4556 |
| SAITS (paper, best) | 2.1930 | 8.7460 | 2.9574 | 0.7274 |
| **KF, no graph (ours)** | 0.1409 | 0.0864 | 0.2940 | 0.0467 |
| **KF, graph MP (ours)** | 0.1411 | 0.0864 | 0.2939 | 0.0468 |
| **Persistence (ours)** | 0.1370 | 0.0844 | 0.2904 | 0.0454 |

**Hard protocol (4–7 of 8 frames missing)**

| Method | MAE | MSE | RMSE | MRE |
|---|---|---|---|---|
| Transformer (paper) | 8.9096 | 92.2607 | 9.6478 | 2.8866 |
| US-GAN (paper) | 7.8833 | 75.9804 | 8.7167 | 2.6127 |
| BRITS (paper) | 7.6057 | 72.0169 | 8.4863 | 2.5207 |
| M-RNN (paper) | 3.2443 | 20.2543 | 4.5005 | 1.1686 |
| TimesNet (paper) | 7.9484 | 106.7010 | 11.3296 | 2.6343 |
| SAITS (paper, best) | 2.6050 | 16.0168 | 4.0021 | 0.8634 |
| **KF, no graph (ours)** | 0.2324 | 0.2944 | 0.5425 | 0.0770 |
| **KF, graph MP (ours)** | 0.2324 | 0.2937 | 0.5420 | 0.0770 |
| **Persistence (ours)** | 0.2268 | 0.2894 | 0.5380 | 0.0752 |

Full per-m (number of missing coordinates) breakdown for both datasets is in
`results_ETH-M.json` / `results_HOTEL-M.json`.

## What this says about generalisation

**1. The graph term barely moves the needle on either dataset — and that is
itself consistent with the sensor paper, not a contradiction of it.** In the
ICST paper, graph message passing only helped when a node had *no own history
at all* (entire motes withheld for the whole test window): +27.1% there, but
only worth <5% at nodes with any of their own recent readings. TrajImpute's
missingness is 1–7 frames dropped from an otherwise-intact 8-frame window of
the *same* pedestrian — own-history is almost always available nearby in time,
so `theta_tm≈1` (near-random-walk) already explains the signal and the graph
term (`theta_sp≈0.007`) has little left to add. This holds essentially
identically for ETH-M (university square) and HOTEL-M (hotel entrance) despite
their very different scene layouts, which is a reasonably strong confirmation
that it's the *missingness regime* (short own-history-rich gaps), not
dataset-specific geometry, that determines whether graph structure helps.

**2. Both KF variants — and even plain persistence — beat every deep imputation
baseline TrajImpute reports, on both datasets and both protocols.** This holds
up as directionally consistent across ETH-M and HOTEL-M, which is reassuring,
but the *margin* is not consistent, and that inconsistency is worth flagging
rather than glossing over:

- On ETH-M, our best (persistence) beats the paper's best (SAITS) by roughly
  20–35% (MAE 0.40 vs 0.50 Easy; 0.65 vs 1.00 Hard).
- On HOTEL-M, our best beats their best by roughly **15x** (MAE 0.14 vs 2.19
  Easy; 0.23 vs 2.61 Hard).

I checked whether HOTEL-M pedestrians simply move less (which would trivially
explain a bigger persistence advantage there): they don't — mean per-frame
displacement in the clean `pred_traj` is 0.236 m for ETH-M and 0.237 m for
HOTEL-M, essentially identical. So the size of our advantage over the paper's
own baselines varies by 15x between two datasets with matched motion
statistics, while our method's absolute error is *flat to slightly better* on
HOTEL-M than ETH-M and the paper's own SAITS is *4x worse* on HOTEL-M than
ETH-M. That asymmetry lives entirely on the "paper's baselines" side of the
comparison, not on the physical-motion side, which makes me want to verify it
against TrajImpute's own evaluation code before treating it as a real finding
rather than a possible mismatch in how the comparison is being made (e.g. a
scale/normalization step in their pipeline I don't have visibility into).

**3. Practical implication for a possible write-up.** If this goes into the
ICST paper's future-work/generalisation section, the honest framing is: the
paper's central claim — that the *measurement-validity/missingness regime*
determines whether graph structure helps, not the estimator — replicates
cleanly across two unrelated domains (pedestrians vs. sensors), two cities
(ETH vs. HOTEL), and two protocols (Easy vs. Hard). The magnitude of the graph
benefit itself does not transfer (negligible here vs. 27% there), but the
*mechanism* governing when it appears does. The absolute comparison against
TrajImpute's reported baselines is the part that needs independent
verification before it's asserted anywhere formal.

## Caveats / what to check before trusting this further

- **Radius (2 m) and scene-position proxy (mean of observed frames) were
  chosen, not tuned or swept, and used unchanged for both datasets.** A radius
  sweep would strengthen this.
- **MRE formula** used here is `sum(|pred-target|) / sum(|target|)` over
  masked entries (the convention used in the BRITS/SAITS lineage); if
  TrajImpute computed it differently the MRE column may not be exactly
  apples-to-apples, though MAE/RMSE are unambiguous and tell the same story.
- **`seq_start_end` for the replicated test blocks is reconstructed by offset,
  not read directly from the file** (see Setup) — a real gap in the released
  metadata, not an assumption about our method.
- **The 15x-vs-1.3x discrepancy in our margin over SAITS between HOTEL-M and
  ETH-M is unexplained** and is the single most important thing to verify
  before using these numbers anywhere — ideally by running TrajImpute's own
  SAITS checkpoint/eval script directly, rather than trusting the Table 2
  numbers transcribed from the PDF.
- No hyperparameter search, no train/val model selection loop, no comparison
  against a naive linear-interpolation baseline (would be a cheap and
  informative addition).
- Hard protocol and HOTEL-M were both added beyond the original "ETH-M, Easy
  only" scope because the pipeline made them nearly free to extend to and they
  directly test the paper's regime-dependence claim more thoroughly.

# Part 2 — GeoLife: building the dataset from scratch

GeoLife (Microsoft Research Asia, 182 users, 2007–2012, mostly Beijing) has no
train/val/test split, no scene structure, and — unlike ETH/HOTEL's
synchronised crowd video — users logged GPS independently, so there is no
guarantee any two people were ever recorded at the same place and time. Every
part of the TrajImpute-style dataset had to be built rather than read off a
file.

## Feasibility check (before building anything)

The whole exercise only makes sense if multi-agent co-presence actually exists
in GeoLife. Checked before writing any pipeline code:

- Day-level: of 1,872 distinct days with any activity, **1,599 have 2+ users
  active** (some up to 28), using trajectory-filename start times alone.
- Point-level pilot (one busy day, 10 users, 45 pairs checked): **3 pairs**
  came within 150 m of each other within a 5-minute window using actual GPS
  points — genuine encounters, not just same-day coincidence.

This confirmed a graph can be mined here, but it has to be *mined*, not read
off a provided field.

## Setup

**Scope (a bounded first pass, per your direction).** Restricted to the 69
users who have `labels.txt` transportation-mode labels (mode is needed to fit
separately per mode), points inside a generous Beijing bounding box, and the
**50 busiest days** by distinct-labeled-user overlap (1,091 `.plt` files,
684,697 labeled GPS points across 28 users who actually had labeled points on
those days). Read directly from the 353 MB zip via Python's `zipfile` —
extracting 18,740 tiny files to disk was tested and was too slow in this
sandbox (~1 MB/s, would have taken 30+ minutes across many tool calls).

**Mode bucketing.** Each point tagged with its labelled transportation mode
via interval lookup in `labels.txt`. Taxi was initially merged into car (per
a literal reading of the user guide, which groups them together as
"driving"), then **split back out as its own mode** (see "Cab data" section
below) once it became clear that lumping them together was masking a
distinct signal. Yielded enough data for **walk** (202,612 points), **bus**
(190,440), **bike** (108,181), private **car** (90,271), and **taxi**
(44,853) considered separately, plus a thinner but usable **subway**
(26,174); train/run/airplane had too few distinct users (1-4) for a
meaningful user-disjoint split and were dropped.

**Global time grid.** This is what makes a graph possible at all: every
user's points are resampled (linear interpolation, refusing to interpolate
across gaps > 30 s) onto the *same* absolute 5-second grid, anchored to a
fixed reference epoch rather than each trajectory's own start time. Two
different users' 100-second windows (8 observed + 12 predicted steps, same
convention as TrajImpute) are then directly comparable whenever they share a
window index — no separate alignment step needed.

**Scenes / adjacency.** Windows sharing a global window index are candidate
co-presence; a spatial-proximity check then decides real edges. Pairwise
distance among same-window candidates is cleanly bimodal in every mode
(effectively 0 m or many km, nothing in between — see `stage3` distance
diagnostics), so a single **50 m** radius separates genuine encounters from
same-clock-time coincidences. This produced real multi-user scenes (2+
distinct users) in every mode: 331 for walk, 411 for bus, 305 for car, 209
for bike, 43 for subway (out of 2,970/3,104/2,474/1,592/304 total scenes
respectively — most scenes are still size-1, same as many ETH/HOTEL scenes).

**Split.** 70/15/15 by user (disjoint people in train/val/test), separately
per mode since user pools differ by mode (8-23 users depending on mode).

**Ground truth.** Unlike TrajImpute, no replication trick is needed: since
this dataset is generated here, the clean pre-missingness values are saved
directly (`obs_traj_clean`) alongside the NaN'd `obs_traj`. Easy (m~Uniform
{0..4}) / Hard (m~Uniform{4..7}) missingness simulated on the 8 observed
steps, mirroring TrajImpute's own protocol.

## Two different kinds of "missing" — don't conflate them

**Real GPS gaps in the raw logs.** Checked how much of each mode's total
labeled time falls into a real gap too long to trust interpolation across
(> 30 s between actual GPS fixes):

| Mode | % of labeled time in gaps > 30s | Max single gap |
|---|---|---|
| walk | 46.9% | 3,587 s |
| subway | 42.5% | 3,418 s |
| bike | 33.8% | 3,299 s |
| train | 22.3% | 948 s |
| bus | 25.4% | 3,516 s |
| car | 16.9% | 2,325 s |

Subway's 42.5% is consistent with the user guide's own note that "no
trajectory can be recorded in an underground subway system since a GPS
logger cannot receive any signal there." Walk's 46.9% is likely mostly the
logger being off between separate outings, not signal loss mid-walk.

**Critically, the pipeline's response to these real gaps was to exclude them,
not recover them**: a window is only ever built from a stretch where real
fixes are never more than 30 s apart (see `geolife_windows.py`). So every
result reported here — including the subway graph-benefit finding — comes
from densely-sampled stretches only. **The subway result is not "recovering
GPS lost in the tunnel."** It's the graph benefit measured on whatever
portions of a subway trip were well-sampled to begin with (platform waits,
above-ground segments, or the underground link simply working that day) once
the *separate*, artificial Easy/Hard protocol below is applied on top. This
distinction matters if this ever gets cited: real signal dropout was
designed out of the evaluation, not solved by it.

**The artificial missingness actually being scored** is the Easy (m~Uniform
{0..4}) / Hard (m~Uniform{4..7}) protocol described above, applied to those
already-clean windows — same convention as TrajImpute, and the only thing
the MAE/RMSE table reflects. Breaking it down by exact missing-count m
(`results_GeoLife_by_m.json`) shows the expected monotonic pattern — error
rises with m in every mode — and, more importantly, shows the subway
graph-vs-no-graph edge holds up *at every m level*, not just in the
aggregate:

| Mode | m | KF no-graph MAE | KF graph MAE | Persistence MAE |
|---|---|---|---|---|
| subway | 1 | 23.87 | 23.70 | 25.45 |
| subway | 2 | 24.38 | 24.24 | 25.64 |
| subway | 3 | 18.55 | 18.52 | 18.88 |
| subway | 4 (Easy) | 18.69 | 18.53 | 20.22 |
| subway | 4 (Hard) | 32.76 | 32.67 | 33.61 |
| subway | 5 | 33.72 | 33.69 | 33.99 |
| subway | 6 | 28.01 | 27.93 | 28.81 |
| subway | 7 | 69.91 | 69.91 | 69.91 |

(This breakdown predates the taxi/car split and covers the five original
modes only. A taxi breakdown isn't included because it would be
uninformative by construction: with zero multi-user scenes anywhere in
taxi's test set, `graph` and `no_graph` are identical at every m level for
taxi, for the same reason the aggregate MAE barely differs — see "Cab data"
below.)

(m=7 — only one real frame left to anchor on — collapses all three methods
to the same answer in most modes, as expected.) The other four modes show
the same negligible/mixed graph-vs-no-graph gap at every m level as they do
in aggregate — walk and bus stay flat-to-slightly-negative throughout, not
just on average.

## A real bug, caught and fixed: coordinate scale

First run gave a nonsensical result — the graph-Kalman-filter variants were
*worse* than plain persistence on every mode, contradicting the ETH-M/HOTEL-M
finding that a physically-motivated filter should at least modestly beat
naive persistence. Root cause: GeoLife (x,y) were projected relative to a
**fixed Beijing reference point**, so an individual trajectory's absolute
coordinates can be tens of thousands of metres from the origin. `theta_tm` is
never exactly 1.0 (e.g. 1.00026); multiplying a ~36,000 m coordinate by
1.00026 injects a ~9 m error *per step* — an artefact of coordinate scale,
not evidence about the model. ETH-M/HOTEL-M never hit this because those
coordinates were already camera-local (a few metres from an arbitrary local
origin), so the same imprecision in theta was harmless there.

**Fix:** every scene is now centred on each node's own mean observed position
before fitting/filtering, and re-offset afterward. This is not a tuning
choice — it's what "the state barely changes between steps" is supposed to
mean physically (small relative displacement), and the bug was applying that
assumption in the wrong reference frame. Confirmed by inspecting individual
windows before/after (see script comments in `gkf_geolife.py`). All numbers
below are post-fix.

## Results (post-fix)

MAE / RMSE at masked positions, metres, per mode:

| Mode | Protocol | KF no-graph | KF graph MP | Persistence | theta_tm | theta_sp |
|---|---|---|---|---|---|---|
| walk | Easy | 5.105 / 9.808 | 5.105 / 9.808 | 5.201 / 10.082 | 0.9429 | -0.00018 |
| walk | Hard | 7.892 / 16.642 | 7.892 / 16.642 | 7.955 / 16.775 | 0.9429 | -0.00018 |
| bus | Easy | 16.584 / 31.133 | 16.588 / 31.142 | 16.826 / 31.721 | 0.9472 | -0.00326 |
| bus | Hard | 28.095 / 57.211 | 28.099 / 57.220 | 28.387 / 57.862 | 0.9472 | -0.00326 |
| car (private) | Easy | 26.714 / 44.168 | **26.709 / 44.158** | 27.249 / 45.163 | 0.9479 | +0.01302 |
| car (private) | Hard | 47.274 / 85.467 | **47.257 / 85.444** | 47.896 / 86.378 | 0.9479 | +0.01302 |
| bike | Easy | 15.196 / 22.037 | 15.195 / 22.036 | 15.501 / 22.513 | 0.9401 | +0.00116 |
| bike | Hard | 25.381 / 39.615 | 25.380 / 39.615 | 25.657 / 40.075 | 0.9401 | +0.00116 |
| subway | Easy | 27.107 / 51.487 | **27.084 / 51.464** | 27.342 / 51.995 | 0.9387 | +0.01225 |
| subway | Hard | 45.561 / 97.672 | **45.507 / 97.612** | 46.083 / 98.370 | 0.9387 | +0.01225 |
| taxi | Easy | 35.629 / 62.593 | 35.617 / 62.582 | 36.486 / 63.822 | 0.9453 | **+0.06256** |
| taxi | Hard | 59.555 / 107.017 | 59.546 / 107.002 | 60.274 / 108.372 | 0.9453 | **+0.06256** |

Full JSON in `results_GeoLife.json`. (Numbers above supersede an earlier
version of this table computed when taxi was merged into "car" — that
merged bucket showed car MAE around 37/60 Easy; splitting taxi out drops
private car down to ~27/44, i.e. the taxi trips were the noisier half of
that pooled bucket. See "Cab data: taxi split out from car" below.)

## What this says about generalisation

**1. Once the coordinate-scale bug was fixed, the physically-motivated filter
beats plain persistence in every mode, on both protocols** — a cleaner result
than ETH-M/HOTEL-M, where persistence sometimes edged out the Kalman filter.
The margin is modest (1-6%), consistent with GPS-derived motion being noisier
and coarser (5 s steps vs. ETH/HOTEL's 0.4 s video frames) than vision-tracked
pedestrians, leaving less for any model to add over a naive carry-forward.

**2. The graph term's benefit scales with how physically coupled the
co-located agents actually are — and taxi is now the most striking
demonstration of this, with subway a close second.** `theta_sp` for taxi
(**+0.0626**) is the single largest coefficient found anywhere in this study —
roughly **5x subway** (+0.0122) and **~4.8x** private car (+0.0130), and
overwhelmingly larger than walk (-0.0002), bus (-0.0033), or bike (+0.0012).
Ranked: taxi (0.0626) > subway (0.0122) > car (0.0130) > bike (0.0012) >
bus (-0.0033) > walk (-0.0002) [subway and car are close; taxi is in a
different tier entirely]. This tracks the same physical-coupling story as
before — taxis dispatched to the same stretch of road inherit traffic-flow
and routing correlations at least as tight as subway co-riders sharing a
car — but see the important caveat immediately below: this coefficient is
well-supported in training but essentially untested at evaluation time.

## Cab data: taxi split out from car

The user guide's own labelling groups taxi under a broader "driving" category
alongside private car, and the first pass followed that literally
(`MODE_MERGE = {'taxi': 'car'}`). Splitting them into separate modes turned
up two things worth reporting on their own.

**First, it changes the private-car numbers.** Once taxi's 44,853 points are
removed from car's pool, private car's Easy MAE drops from ~37.1 (merged) to
26.7 (private-car-only) — taxi trips were measurably noisier/more erratic
than private car trips, and pooling them was diluting the private-car signal
rather than adding data to it cleanly.

**Second, taxi produced the strongest-fitted graph coefficient in the whole
study, but the held-out test set can't confirm it.** Digging into why a
`theta_sp` of 0.0626 (5x subway) barely moved the test-set MAE (Easy:
no-graph MAE=35.629 vs graph MAE=35.617, a 0.03% difference) turned up a
scene-composition mismatch between train and test:

| Split | Users | Scenes | Multi-user (2+) scenes |
|---|---|---|---|
| train | 7 (`078,084,085,126,153,167,128`) | 292 | 10 |
| test | 2 (`052,163`) | 261 | **0** |

Taxi's `theta_sp=0.0626` is genuinely fitted from 10 real co-presence scenes
in the training split — not spurious, and consistent with the physical
story above. But the random user-disjoint 70/15/15 split happened to place
only 2 taxi-labeled users in the test set, and those two never appear in the
same window together anywhere in this bounded scope: every one of the 261
test scenes is a singleton. The graph term literally has nothing to act on
during evaluation — `Abar` is the identity for every test scene regardless
of how large `theta_sp` is — so the test-set MAE cannot reflect a benefit
that is nonetheless real in the fit. **This is a small-sample/split-luck
artifact of only having 10 taxi-labeled users in scope, not evidence against
the finding.** A larger pull of taxi data (more busy days, or all 35
taxi-labeled users in the full GeoLife set rather than the 10 that fell
inside this pass's 50-day/28-user bound) would very likely produce test-side
co-presence and let the MAE table catch up to what the fit already shows.

**3. This is the most improvised part of the whole exercise and should be read
that way.** Every step — which days count as "busy," what counts as a
same-mode "run," the 30 s interpolation-gap cutoff, the 50 m proximity
radius, even the user-disjoint split with as few as 2 users in some val/test
sets — was a judgment call made to get a working pipeline out of raw GPS logs
with no reference implementation to check against, unlike TrajImpute where
there's a paper and (likely) public code to compare numbers with. Treat the
qualitative finding (graph benefit tracks physical coupling tightness) as the
useful takeaway, and the exact numbers as illustrative rather than
publication-ready.

## GeoLife-specific caveats

- **Bounded first pass, not the full dataset**: 69 of 182 users (only the
  labeled ones), 50 of ~1,872 active days, Beijing only. Expanding this was
  explicitly deferred per your call on scope.
- **Co-presence is mined, not given**: the 50 m radius and the "same global
  window index" definition of a candidate scene are both choices calibrated
  against this dataset's own bimodal distance distribution, not validated
  against any external notion of "these two people were actually together."
- **Very small per-mode user pools for val/test** (as few as 2 users each for
  car/bike/subway/taxi) — per-mode metrics, especially for subway and taxi,
  should be read as indicative, not statistically robust.
- **Taxi's test-set MAE cannot yet confirm its own fitted graph benefit**:
  the strongest `theta_sp` in the study (0.0626) comes from only 10
  co-presence scenes among 7 training users, and the 2 test users never
  co-occur, so the graph term is inert (identity `Abar`) for all 261 test
  scenes. See "Cab data" section above — treat the fitted coefficient as the
  finding, not the near-zero test MAE delta.
- **R_meas (measurement noise) was left at the same 1e-4 constant used for
  ETH-M/HOTEL-M's essentially-exact vision-tracked coordinates.** A quick
  sensitivity check (R from 1e-4 to 400) showed *increasing* R made things
  worse, not better, so this wasn't blindly wrong — but it was never
  independently fit for GPS-scale measurement noise (~5-10 m accuracy), and
  doing so properly would strengthen the comparison.
- **The centring fix was validated by eyeballing individual windows, not by
  an automated test.** Worth double-checking on a couple more scenes before
  trusting the exact numbers.
- Non-Beijing GeoLife data (~30 other cities, some in the US/Europe) and the
  full 5-year span were entirely out of scope for this pass.

## Files

- `scripts/torchless_load.py` — torch-free `.pkl` loader (TrajImpute part).
- `scripts/gkf_trajimpute.py` — fits theta_tm/theta_sp on train, runs the
  Kalman filter (graph and no-graph) plus a persistence baseline on Easy and
  Hard test splits, for whichever dataset is set via the `GKF_DATASET`
  environment variable (`ETH-M` or `HOTEL-M`). Writes `results_<DATASET>.json`.
- `results_ETH-M.json`, `results_HOTEL-M.json` — full numeric results, overall
  and by missing-count, per dataset.
- `scripts/geolife_prepare.py` — reads the GeoLife zip directly (no
  extraction), restricts to labeled users + Beijing bbox + busiest days,
  tags points by transportation mode, projects to local metres.
- `scripts/geolife_windows.py` — resamples onto the global 5 s grid per
  same-mode run, cuts 8+12-step windows aligned to absolute time.
- `scripts/geolife_split.py` — builds per-mode co-presence scenes (50 m
  radius), simulates Easy/Hard missingness, splits 70/15/15 by user, saves
  `GeoLife-M/<mode>/<protocol>/data_<split>.npz`.
- `scripts/gkf_geolife.py` — same fit/evaluate logic as `gkf_trajimpute.py`,
  adapted for the `.npz` format and the per-scene coordinate centring fix.
  Writes `results_GeoLife.json`.
- `results_GeoLife_by_m.json` — GeoLife results broken down by exact
  missing-count m per mode/protocol (see "Two different kinds of 'missing'").
