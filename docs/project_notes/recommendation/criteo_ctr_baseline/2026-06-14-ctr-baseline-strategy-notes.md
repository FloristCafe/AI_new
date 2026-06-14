# Criteo CTR Baseline Strategy Notes

## Why `docs/project_notes/recommendation/criteo_ctr_baseline/`

This project note lives under:

- `docs/project_notes/`: project-facing notes, not code and not roadmap
- `recommendation/`: track-level grouping
- `criteo_ctr_baseline/`: one folder per concrete project

Rule to keep using later:

- roadmap and workspace rules stay in `docs/roadmaps/`
- project-specific reasoning, experiment notes, and strategy reviews stay in `docs/project_notes/<track>/<project>/`
- do not scatter notes into `src/`, `artifacts/`, or the workspace root

This keeps "what we built" separate from "why we chose it".

## Strategy Decision 1: Dense Missing Value Sentinel

Current decision:

- dense feature missing values are filled with `-1`

Reasoning:

- In this CTR dataset, dense features are closer to count-like or non-negative behavior than to symmetric real-valued measurements.
- A real `0` can mean "observed but zero".
- A missing value often means "this field was unavailable or absent".
- Filling with `0` mixes these two meanings together.
- Filling with `-1` creates a simple first-pass missing sentinel that a linear model can distinguish.

What this decision does well:

- preserves the difference between "zero" and "missing"
- is easy to implement in a baseline
- is usually more faithful than `0` for non-negative dense fields

What it does not solve:

- `-1` is still a heuristic, not a learned representation
- the model may still treat missingness too crudely
- a stronger later version should consider adding explicit missing-indicator columns

Practical conclusion:

- For the current baseline, `-1` is a reasonable improvement over `0`
- For later versions, "dense value + missing flag" is likely better

## Strategy Decision 2: Why One-Hot Can Be Biased in Ad Click Modeling

Important point:

- One-hot is not only "high dimensional"
- In ad click prediction, it can be systematically biased by serving logic and human behavior

### 1. One-hot treats IDs as isolated facts

Ad CTR data contains many categorical IDs:

- user-related buckets
- ad / campaign / creative IDs
- publisher / page / slot IDs
- device / context IDs

One-hot gives each category its own separate coordinate.

Problem:

- it does not express similarity between categories
- two creatives that are semantically close are treated as unrelated
- two placements with similar audience intent are treated as unrelated

Result:

- the model memorizes sparse coincidences instead of learning reusable structure

### 2. Clicks are affected by exposure policy, not only user preference

An ad click is not a pure preference label.
It is the result of a serving pipeline:

- auction and bidding decisions
- targeting rules
- placement constraints
- time and page context

This means a category may look "good" because it was shown under better conditions, not because it is intrinsically better.

One-hot tends to absorb these accidental correlations very directly.

Example:

- a certain campaign ID may get high CTR because it was mostly exposed to high-intent users
- one-hot may then learn "this ID is good"
- but what actually mattered was targeting and exposure condition, not the ID alone

### 3. Human behavior creates unstable correlations

Ad clicks are influenced by many human factors:

- novelty effect
- ad fatigue
- accidental clicks
- trust in brand or site
- time pressure
- repeated exposure changing response

One-hot cannot represent these dynamics well.
Instead, it freezes a category into a static indicator.

Result:

- yesterday's useful ID pattern can become today's noise
- repeated exposures may reduce click probability, but one-hot alone cannot express that evolving effect

### 4. Sparse one-hot magnifies small-sample accidents

With high-cardinality features and small sample size:

- many categories appear only a few times
- a few lucky or unlucky clicks can dominate their estimated weight

This is especially dangerous in ad data because:

- positive labels are rare
- exposure is selective, not random
- many IDs are operational artifacts rather than stable semantic objects

So one-hot can exaggerate random local history into apparently meaningful signals.

### 5. Important interactions are missing

CTR is often driven by interactions such as:

- user intent x ad creative
- page context x device
- slot position x campaign type
- frequency of exposure x user response

Plain one-hot plus linear model mostly learns additive effects.
It misses many interaction patterns unless we engineer them manually.

That creates another source of distortion:

- the model may assign blame or credit to a single ID
- but the real effect may only exist under a specific combination

## Project-Level Takeaway

For this project, one-hot with logistic regression is still useful as a baseline because it helps us:

- verify the pipeline
- establish a simple reference point
- observe overfitting and calibration behavior

But it has clear structural weaknesses in ad click modeling:

- memorization of sparse IDs
- sensitivity to exposure bias
- inability to represent semantic similarity
- weak handling of evolving human response
- weak interaction modeling

So this baseline should be treated as:

- a pipeline sanity check
- a calibration and experiment starting point
- not a strong final representation for recommendation or ad CTR modeling

## Likely Next Modeling Directions

More suitable later directions include:

- adding explicit missing indicators for dense features
- expanding sample size before over-interpreting small-micro results
- factorization-based models such as FM / FFM
- embedding-based models such as DeepFM
- features that encode exposure count, recency, or interaction structure

## Current Code Decision

Applied in code now:

- dense missing values use `-1` instead of `0`

Not changed yet:

- the overall logistic regression plus one-hot baseline remains, because it is still useful as a controlled reference model
