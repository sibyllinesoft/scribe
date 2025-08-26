# FastPath V5 — `todo.md`

**TL;DR:** Tighten baselines, ground evaluation in real data/tasks, verify citations, scope paper narrative, and add engineering details for scalability and updates.

## Invariants (do not change)

* Must clearly state baselines, datasets, and evaluation protocol.
* Ground truth for relevance must be reproducible and auditable.
* No fabricated or unverifiable citations.
* Hermetic spin-up for experiments (seeded, containerized).
* Oracles (contracts/properties) enforce runtime correctness.

## Assumptions & Scope

* **Assumption:** Evaluation repositories are open-source (≥5, diverse in size/language).
* **Assumption:** “Relevance” derived from PR/task annotations (files edited/linked).
* **Assumption:** Mutation threshold ≥0.80; property/metamorphic coverage ≥0.70; 0 high/critical SAST.
* **Scope:** Research paper (ICSE-ready) + runnable prototype code + reproducibility kit.

## Objectives

1. Document all baselines (TF-IDF, semantic, naive recency/random) with reproducible configs.
2. Publish dataset table (repos, size, language, domain, task count).
3. Define and implement ground-truth protocol with inter-rater reliability or automated proxy.
4. Verify all citations; correct or replace inaccuracies.
5. Add scalability discussion: static vs incremental PageRank, update cost, 10M+ files.
6. Meet reproducibility standards: hermetic spin-up, artifact hashes, signed boot transcript.

## Risks & Mitigations

* Missing/weak baselines → **Mitigation:** Add naive + commercial system discussion.
* Ground-truth ambiguity → **Mitigation:** Explicit annotation protocol with reliability stats.
* Scalability skepticism → **Mitigation:** Engineering detail on incremental graph updates.
* Citation errors → **Mitigation:** Cross-check each against DBLP/ACM/IEEE sources.
* Reviewer distrust (too polished) → **Mitigation:** Present nuanced, messy results (per-repo slices).

## Method Outline (idea → mechanism → trade-offs → go/no-go)

### Workstream A — Baseline Expansion

* **Idea:** Anchor gains vs weak/strong baselines.
* **Mechanism:** Implement random, recency, TF-IDF, semantic, commercial-inspired.
* **Trade-offs:** More runs; stronger comparisons.
* **Go/No-Go:** CI lower bound >0 vs strongest baseline.

### Workstream B — Ground-Truth Protocol

* **Idea:** Explicit reproducible method for labeling relevant files.
* **Mechanism:** Use PR-modified files + human annotators; compute Cohen’s kappa.
* **Trade-offs:** Expensive but credible.
* **Go/No-Go:** κ ≥0.7 (substantial agreement).

### Workstream C — Scalability & Updates

* **Idea:** Address incremental graph updates for large repos.
* **Mechanism:** Use personalized PR with delta updates; benchmark on 10k, 100k, 10M files.
* **Trade-offs:** Engineering effort; runtime cost.
* **Go/No-Go:** ≤2× baseline time at 10M files.

### Workstream D — Citation & Related Work Audit

* **Idea:** Ensure correctness and credibility.
* **Mechanism:** Manually verify all refs; update mismatched systems.
* **Trade-offs:** Time-consuming.
* **Go/No-Go:** Zero incorrect refs.

## Run Matrix

| ID | Method/Variant    | Budget        | Inputs      | Expected Gain      | Promote if…              |
| -- | ----------------- | ------------- | ----------- | ------------------ | ------------------------ |
| V1 | Random baseline   | ±5% runtime   | Repo corpus | Anchor floor       | CI>0 vs random           |
| V2 | Recency baseline  | ±5% runtime   | Repo corpus | Show temporal bias | CI>0 vs recency          |
| V3 | TF-IDF baseline   | ±5% runtime   | Repo corpus | Anchor mid         | CI>0 vs TF-IDF           |
| V4 | Semantic baseline | ±5% runtime   | Repo corpus | Anchor high        | CI>0 vs semantic         |
| V5 | FastPath V5       | System budget | Repo corpus | Target 8–12%       | CI lower bound >0 vs all |

## Implementation Notes

* **APIs:** Oracle (ranking), Clustering, Controller attach points.
* **Graph:** Support incremental PageRank updates; personalized PR for queries.
* **Caching:** Persist PageRank vectors; invalidate on `git pull`.
* **Telemetry:** Latency (p50/p95), throughput, memory; per-repo metrics.
* **Repro:** Pin container, seed data; record SHAs/hashes.

## Acceptance Gates

* Baselines documented with configs.
* Dataset table published; reproducibility kit built.
* Ground truth protocol defined; κ ≥0.7 or proxy justified.
* Scalability: 10M files ≤2× baseline runtime.
* Mutation ≥0.80; property coverage ≥0.70; 0 high/critical SAST.
* All citations verified.

## “Make-sure-you” Checklist

* Add naive baselines (random, recency).
* Publish repo/task/dataset table.
* Define relevance annotation protocol.
* Verify all citations; update incorrect.
* Add scalability and incremental update section.
* Sign boot transcript and pin environment.

## File/Layout Plan

```
fastpath_v5/
  paper/
    draft.tex
    figures/
    refs.bib
  eval/
    datasets/
    tasks/
    results/
    ground_truth_protocol.md
  src/
    oracle/
    clustering/
    controller/
  scripts/
    run_baselines.sh
    run_fastpath.sh
    eval_metrics.py
  artifacts/
    boot_transcript.json
    metrics/
```

## Workflows (required)

```xml
<workflows project="fastpath_v5" version="1.0">

  <workflow name="building">
    <contracts id="B1">
      <desc>Verify citations and compile oracles</desc>
      <commands>
        <cmd>crosscheck_refs.sh refs.bib</cmd>
        <cmd>compile_oracles.py spec/</cmd>
      </commands>
      <make_sure>
        <item>0 incorrect refs</item>
        <item>Contracts generated</item>
      </make_sure>
    </contracts>
    <spinup id="B2">
      <desc>Hermetic boot</desc>
      <commands>
        <cmd>docker build -t fastpath:lock .</cmd>
        <cmd>docker run fastpath:lock spinup_smoke.sh</cmd>
        <cmd>write_boot_transcript.sh</cmd>
      </commands>
      <make_sure>
        <item>Boot transcript signed</item>
      </make_sure>
    </spinup>
  </workflow>

  <workflow name="running">
    <baseline id="R1">
      <desc>Run baselines</desc>
      <commands>
        <cmd>./scripts/run_baselines.sh --random</cmd>
        <cmd>./scripts/run_baselines.sh --recency</cmd>
        <cmd>./scripts/run_baselines.sh --tfidf</cmd>
        <cmd>./scripts/run_baselines.sh --semantic</cmd>
      </commands>
      <make_sure>
        <item>Configs logged</item>
      </make_sure>
    </baseline>
    <fastpath id="R2">
      <desc>Run FastPath V5</desc>
      <commands>
        <cmd>./scripts/run_fastpath.sh</cmd>
        <cmd>./scripts/eval_metrics.py</cmd>
      </commands>
      <make_sure>
        <item>Metrics JSON generated</item>
      </make_sure>
    </fastpath>
  </workflow>

  <workflow name="tracking">
    <harvest id="T1">
      <desc>Collect results and compute CI</desc>
      <commands>
        <cmd>collect_results.py eval/results</cmd>
        <cmd>bootstrap_ci.py --n=10000 --bca</cmd>
      </commands>
      <make_sure>
        <item>95% CI per metric</item>
      </make_sure>
    </harvest>
  </workflow>

  <workflow name="evaluating">
    <gatekeeper id="E1">
      <desc>Apply acceptance gates</desc>
      <commands>
        <cmd>check_mutation_score.sh</cmd>
        <cmd>check_property_cov.sh</cmd>
        <cmd>verify_ci_bounds.sh</cmd>
      </commands>
      <make_sure>
        <item>All objectives met before promotion</item>
      </make_sure>
    </gatekeeper>
  </workflow>

  <workflow name="refinement">
    <manual id="N1">
      <desc>Manual QA if evaluation fails</desc>
      <commands>
        <cmd>open_tracking_dashboard.sh</cmd>
        <cmd>assign_owner.sh</cmd>
      </commands>
      <make_sure>
        <item>Owner assigned; repros attached</item>
      </make_sure>
    </manual>
  </workflow>

</workflows>
```

## Next Actions (strict order)

1. Implement and document naive baselines (random, recency).
2. Build dataset + task table; publish ground-truth protocol doc.
3. Verify all citations in refs.bib.
4. Add scalability experiments (incremental PageRank at 10M files).
5. Run evaluation pipeline; compute CIs; apply Gatekeeper.
