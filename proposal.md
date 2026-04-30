# ESoC 2026 Proposal

**GC.OS + sktime STEP**

# AgenticForecaster for sktime

**An LLM-driven, tool-calling forecasting agent built on sktime-mcp with explainable outputs and extensible LLM backend support**

| Field | Details |
| --- | --- |
| Name | Saloni Sharma |
| GitHub | [Saloni-0465](https://github.com/Saloni-0465) |
| Email | salonivashisht6564@gmail.com |
| Timezone | UTC+5:30 (India) |
| University | Rishihood University - B.Tech CSE |

## 1. Project Overview

### Abstract

Time-series forecasting in practice is difficult not because forecasting algorithms are unavailable, but because each dataset requires a chain of fragile decisions: which model family to try, whether the index supports frequency-based assumptions, whether exogenous variables are available for the forecast horizon, and which evaluation metric should guide selection. These choices are often handled manually through trial and error, and mistakes around index handling, future exogenous data, or evaluation setup can lead to silent failures or misleading forecasts.

This project proposes AgenticForecaster, an experimental forecasting agent for sktime-mcp with an sktime-compatible estimator-style interface. The agent will use an LLM as a bounded reasoning component while relying on sktime-mcp tools as the only execution layer. Instead of generating free-form Python code, the LLM will operate inside a constrained finite-state loop: summarize dataset metadata without transmitting raw time-series values, rank candidates from a prevalidated registry-derived menu, execute candidates through MCP tools, evaluate their performance, select the best successful model, and return both forecasts and a structured explanation.

The MVP will support an OpenAI backend, using gpt-4o or the current recommended OpenAI production model, through a small backend interface designed to allow future provider support. Anthropic Claude support is planned as a stretch goal if the core system stabilizes ahead of schedule.

AgenticForecaster differs from single-family automated methods such as AutoARIMA by performing guarded family-level model selection across eligible sktime forecasting estimators. Deterministic rules will take priority over LLM suggestions: non-datetime indices will disable calendar-frequency assumptions, models requiring future exogenous data will be skipped or marked when future X is unavailable, and every candidate will run in an isolated failure boundary. The final output will include not only the selected forecast, but also candidate metrics, warnings, and an audit log of tool calls, making the forecasting process more reproducible, inspectable, and safer for common sktime-mcp forecasting workflows.

### Problem Statement

There are four interconnected problems this project addresses:

- **High friction in model selection:** Users must reason about seasonality, trend, horizon, and metric choice before writing a single line of fitting code. There is no guided path in sktime today.
- **Silent failure modes:** Index/frequency mismatches and exogenous misalignment are among the most common sources of forecasting bugs, and they often produce wrong results silently rather than raising clear errors.
- **LLMs without guardrails produce fragile code:** A naive LLM-based forecaster that generates Python scripts will hallucinate API signatures, skip evaluation, and produce results that cannot be audited or reproduced.
- **sktime-mcp is an underused foundation:** The MCP layer already exposes robust tool primitives for estimator discovery, fitting, prediction, and evaluation. This project turns that foundation into a user-facing agentic forecasting interface.

## 2. Evidence of Technical Competence

### A) sktime-mcp - 9 PRs (Primary Codebase)

These contributions are directly relevant because they address the exact failure modes AgenticForecaster must handle safely. Each PR strengthens the tool layer the agent will rely on.

| PR | Description | Status |
| --- | --- | --- |
| #246 | [MNT] Replace deprecated ExpandingWindowSplitter import - updated deprecated import path to current sktime API to avoid future breakage | Open |
| #133 | fix: restrict auto-format to datetime-like indices - prevented unsafe calendar gap-filling on non-datetime indices; improved infer_freq and index .freq robustness | Open |
| #128 | fix: include job_id in all failure paths of fit_predict_async - async tool responses always include job_id so failures remain debuggable | Open |
| #119 | fix: correct handling of exogenous variables in fit_predict workflows - training X must not be passed to predict(); added future exogenous support via future_data_handle | Open |
| #107 | fix: update export_code to generate task-aware examples - generated code now matches estimator task type (forecasting vs classification) | Open |
| #105 | fix: fail fast in fit_predict_async for invalid estimator handles - invalid handles fail early instead of creating doomed background jobs | Open |
| #104 | Update invalid list_data_handles reference to list_available_data - fixed tool description so LLMs call the correct registered tool | Merged |
| #103 | fix(docs): add missing instantiation step in forecasting workflow - corrected the Hello World workflow to create estimator handle before fit/predict | Merged |
| #95 | Add detection estimators to MCP registry via TASK_MAP - expanded registry coverage for detector scitype with tests and docs | Open |

How these directly establish the base for AgenticForecaster:

- PR #119 directly informs the agent rule: never pass training X to predict(); never select an exogenous-requiring model unless future X is aligned to the horizon.
- PR #133 maps to the agent's metadata-driven index constraint: non-datetime indices disable calendar-frequency assumptions.
- PR #128 and #105 ensure the agent loop can fail gracefully: async jobs are always traceable and invalid handles are rejected before wasting tokens.
- PR #104 and #103 fix the exact schema guidance the LLM uses to call tools correctly - foundational for an agent that must call tools without hallucinating names.

### B) sktime - 4 PRs (Ecosystem Depth)

| PR | Description | Status |
| --- | --- | --- |
| #9833 | Relax dtype checks in long-format loader - broader format compatibility | Open |
| #9826 | [DOC] Fix np.darray typo to np.ndarray - documentation correctness | Merged |
| #9825 | [DOC] Fix incorrect reference in SlidingWindowSplitter docstring | Open |
| #9806 | [DOC] Expand developer guide: document _safe_import for soft dependencies | Open |

### C) SHAP - 3 PRs (Explainability Experience)

These contributions demonstrate I can ship reliable, tested changes in explainability tooling - directly relevant to the explain() output goal.

| PR | Description | Status |
| --- | --- | --- |
| #4355 | DOC: expand shap.plots.bar docstring + type hints - improved API documentation and typing for a key explanation entrypoint | Open |
| #4332 | Fix text plot color rendering by converting numpy types to float - fixed real bug causing invalid CSS from NumPy scalars, with tests | Merged |
| #4334 | fix(plots): avoid stacked colorbars on repeated summary_plot - robust fix with tests and refactor for repeated explanation plotting | Open |

## 3. Detailed Design & Implementation

### Architecture Overview

AgenticForecaster is a sktime-compatible estimator that wraps a finite-state agent loop. The LLM, using an OpenAI backend in the MVP, is the reasoning engine; sktime-mcp tools are the only execution path. The LLM never generates Python training code. It only selects from a safe menu and interprets structured tool results.

The design below describes my current thinking on each component. These are not final implementation specs - they represent how I have reasoned through the key decisions so far, informed by my prior work in sktime-mcp. I expect details to be refined with mentors during the bonding period.

### Agent Loop: Five States with Explicit Failure Handling

#### State 0 - Summarize Inputs (No Raw Data Leakage)

The agent begins by computing a compact metadata summary. Raw y values are never transmitted to the LLM. The summary includes:

- Index type (DatetimeIndex, PeriodIndex, integer, other) and inferred frequency if applicable
- Series length, missingness fraction, univariate vs. multivariate target shape
- Presence and shape of X (training exogenous), and whether future X is available for the forecast horizon
- Any detected seasonality signal, such as autocorrelation at candidate seasonal lags

This summary is transmitted as a fixed-schema JSON object so prompts stay stable and testable.

#### State 1 - Propose Candidates (Bounded, Rule-Guided + LLM-Ranked)

Candidate generation is constrained by deterministic heuristics that always take priority over LLM suggestions:

- Always include a naive or seasonal naive baseline, which is guaranteed to run and provide a lower-bound comparison.
- If X is present in training but no future X is available, exclude any model requiring future exogenous data, or include it with `candidate_status="needs_future_X"` and skip it in selection.
- If the index is non-datetime-like, exclude any model requiring calendar frequency gap-filling.
- Candidate pool is capped at 3-6 models to control token cost and runtime.

Within this constrained pool, the LLM ranks candidates by suitability given the metadata summary. When LLM ranking conflicts with a deterministic heuristic, the heuristic wins. The LLM reasons within a safe menu, not over an unbounded space.

#### State 2 - Execute with Per-Candidate Isolation

Each candidate is executed inside its own error boundary. A single failed candidate does not crash the loop:

```python
for candidate in candidates:
    try:
        handle = mcp.instantiate_estimator(candidate.name, candidate.params)
        fit_result = mcp.fit(handle, y_train_handle, X_train_handle)
        pred_result = mcp.predict(handle, fh, future_X_handle)
        eval_result = mcp.evaluate(pred_result, y_test_handle, metric)
        candidate.status = "success"
        candidate.metrics = eval_result
    except FitError as e:
        candidate.status = "fit_failed"
        candidate.error = str(e)
    except PredictError as e:
        candidate.status = "predict_failed"
        candidate.error = str(e)
    except EvalError as e:
        candidate.status = "eval_failed"
        candidate.metrics = fallback_metrics(pred_result)
```

#### State 3 - Select Best Candidate

The agent chooses the best successful candidate by primary metric, such as MASE or sMAPE, with tie-breakers on runtime and model simplicity. If all candidates fail, a graceful failure is returned containing per-candidate error diagnostics and recommended next actions, such as "provide future X", "reduce horizon", or "switch index type".

#### State 4 - Return Forecast + Explanation

The final output is both the predictions and a structured report from `explain()`.

### explain() Output Schema

The `explain()` method returns a structured dict, also serializable to JSON for logging, with the following fields:

```python
{
  "selected": {
    "estimator": "AutoARIMA",
    "params": {"sp": 12, "d": None},
    "reason": "Lowest MASE (0.87) among successful candidates; seasonal signal detected at lag 12."
  },
  "data_summary": {
    "index_type": "DatetimeIndex",
    "freq": "M",
    "length": 120,
    "missing_fraction": 0.0,
    "exogenous": True,
    "future_X_available": True
  },
  "candidates": [
    {"estimator": "NaiveForecaster", "status": "success", "MASE": 1.42, "runtime_s": 0.3},
    {"estimator": "AutoARIMA", "status": "success", "MASE": 0.87, "runtime_s": 4.1},
    {"estimator": "DirectTabularRegressionForecaster", "status": "fit_failed", "error": "..."}
  ],
  "warnings": ["future X was aligned to fh by truncation - verify alignment is correct."],
  "audit_log": [
    {"tool": "instantiate_estimator", "args": {...}, "ts": "2026-05-01T10:00:01Z"},
    {"tool": "fit", "args": {...}, "ts": "2026-05-01T10:00:02Z"}
  ]
}
```

### LLM Backend Design

The MVP targets an OpenAI backend, using GPT-4o or the current recommended OpenAI production model. The agent loop is written against a thin `BaseLLMBackend` interface so that adding a second backend does not require touching the core logic - only a new adapter class. Anthropic Claude support is a stretch goal, planned for week 11 if the core is stable ahead of schedule.

```python
# MVP: OpenAI only
forecaster = AgenticForecaster(model="gpt-4o", metric="MASE", max_candidates=5)

# Stretch goal, week 11 if time permits
forecaster = AgenticForecaster(backend="anthropic", model="claude-sonnet-4-20250514")
```

### Proposed User-Facing API

```python
from sktime_mcp.agentic import AgenticForecaster

forecaster = AgenticForecaster(
    backend="openai",
    model="gpt-4o",
    metric="MASE",
    max_candidates=5,
)

forecaster.fit(y, X=X_train)
y_pred = forecaster.predict(fh=12, X=X_future)
report = forecaster.explain()
```

If no viable candidate is found, `fit()` raises `ForecastingAgentError` with per-candidate diagnostics. `explain()` remains available, even on failure, with suggested next actions.

### Non-Goals

To set reviewer expectations clearly, the following are explicitly out of scope:

- Streaming inference or real-time forecasting pipelines
- Neural architecture training; deep learning models are not in the candidate pool for v1
- Transmitting raw time-series data to any LLM; the metadata-only contract is a hard constraint
- General-purpose code generation; the agent calls tools only and never generates executable Python

### MVP vs. Stretch Goals

This distinction is the most important scoping decision in the proposal. The MVP is what I guarantee to deliver regardless of any slowdowns. Stretch goals are additions if the core is stable ahead of schedule.

| MVP (Guaranteed by Week 10) | Stretch Goals (Weeks 11-12 if ahead) |
| --- | --- |
| AgenticForecaster class with sktime estimator interface<br>Metadata summarizer (no raw data leakage)<br>Deterministic candidate proposal with heuristic rules<br>Per-candidate isolated execution via MCP tools<br>Evaluation-driven selection (MASE/sMAPE)<br>explain() with selection rationale + warnings<br>Exogenous-aware candidate filtering<br>OpenAI GPT-4o backend only<br>Benchmarked vs NaiveForecaster + AutoARIMA<br>Tutorial notebook + docs | Anthropic Claude backend (BaseLLMBackend abstraction already designed for this)<br>Async execution path via fit_predict_async<br>Audit log export (JSON) for external tooling<br>Cross-backend regression test suite<br>Additional tutorial: multivariate + exogenous workflow |

The MVP alone is a complete, usable, and testable contribution to sktime. The stretch goals extend backend coverage and performance paths but do not affect the correctness of the core agent loop.

## 4. Project Timeline (12 Weeks)

Weeks 1-10 deliver the MVP. Weeks 11-12 are reserved for stretch goals, documentation hardening, and release. Every week has one focused milestone and one concrete testable output.

| Week | Milestone / Focus | Testable Deliverable |
| --- | --- | --- |
| 1 | Bonding: spec + acceptance tests + benchmark datasets agreed | Written spec; acceptance tests for all 5 loop states; 3 benchmark datasets confirmed with mentors |
| 2 | Metadata summarizer + no-raw-data enforcement | MetadataSummarizer class with tests; prompt template verified to pass zero raw y values |
| 3 | Candidate proposal policy + baseline-only end-to-end | CandidateProposer with deterministic heuristics; single end-to-end run (NaiveForecaster) on 1 dataset |
| 4 | Multi-candidate execution with per-candidate isolation | ExecutionLoop class; test: one failing candidate does not crash loop; 3-candidate run on demo data |
| 5 | Evaluation-driven selection on 3 benchmark datasets | SelectionPolicy class; integration tests passing on airline, M4-monthly, and one custom dataset |
| 6 | Exogenous-aware candidate filtering + alignment validation | Future X path; tests for needs_future_X warning and exclusion rule; end-to-end with exogenous dataset |
| 7 | explain() report + audit log + snapshot tests | explain() returns correct schema on all test cases; snapshot tests lock output structure |
| 8 | Cost controls + regression tests + API stabilization | max_candidates enforced; full regression suite; public API locked for review |
| 9 | Benchmarking harness vs NaiveForecaster + AutoARIMA | Benchmark script; results table on agreed datasets; documented performance comparison |
| 10 | MVP docs + tutorial notebook + failure recovery guide | Tutorial notebook (univariate workflow); failure modes guide; MVP declared complete |
| 11 | Stretch: Anthropic Claude backend + async execution path | AnthropicBackend adapter class (if time); async path wired to fit_predict_async (if time) |
| 12 | Stabilization, API review, final test hardening, release PR | 90%+ test coverage; release-ready PR with changelog; cross-backend tests if stretch done |

Phase summary:

- Weeks 1-2: Infrastructure and bonding
- Weeks 3-6: Core agent loop (MVP foundation)
- Weeks 7-10: Explainability, evaluation, benchmarking, docs (MVP complete)
- Weeks 11-12: Stretch goals + release hardening

## 5. Why I Am Well-Positioned for This Project

I am not proposing to build an agent on top of a codebase I have read from the outside. I have already contributed to the exact failure modes the agent must handle safely, and each major design decision in this proposal traces directly to something I have shipped.

**Exogenous variable correctness (PR #119)** -> Agent rule: never pass training X to predict(); never select an exogenous-requiring model unless future X is available and aligned to the horizon. I discovered and fixed this bug in sktime-mcp, so I know exactly where it will surface in the agent's execution loop and how to route around it.

**Non-datetime index handling (PR #133)** -> Agent constraint: the metadata summarizer detects index type first, and non-datetime indices suppress calendar-frequency assumptions for all downstream candidate proposals. This fix was non-trivial because infer_freq silently succeeds on some integer indices - the agent needs the same defensive check.

**Async job traceability (PR #128) and early validation (PR #105)** -> The agent loop must be able to fail gracefully on any candidate without losing diagnostic context. My work ensuring job_id is always present in failure paths and invalid handles are rejected early means the agent's audit log will remain complete even under partial failure.

**Tool schema accuracy (PR #104, #103)** -> The agent calls tools by name. If tool descriptions are wrong or the documented workflow skips a step, the LLM will hallucinate a call sequence. I fixed both of these: the tool name reference and the missing instantiation step. This is foundational for an agent that must call tools without generating free-form code.

**Explainability tooling (SHAP PRs #4332, #4334, #4355)** -> The explain() output is not just a dict of metrics. It must be consistently structured, testable, and visually clear when surfaced to users. My SHAP contributions show I can deliver reliable, tested changes to explanation outputs, including fixing real rendering bugs and improving API documentation with type hints.

## 6. Community Impact & Usability

AgenticForecaster will lower the barrier to forecasting for three distinct user groups:

- New sktime users who do not know which model family to start with. The agent provides a guided path with a working result and explanation on the first call.
- Intermediate users who know forecasting theory but hit silent bugs, especially index and exogenous-variable issues. The agent surfaces these as explicit warnings rather than wrong predictions.
- Researchers and practitioners who want a reproducible baseline selection method. The audit log of tool calls makes every decision traceable and replicable.

The project also benefits the sktime-mcp ecosystem by exercising the MCP tool layer at scale. Any reliability gaps discovered during agent testing will produce additional PRs that improve the tool layer for all users, not just agent users.

The backend abstraction is designed to support multiple providers, with OpenAI in the MVP and Anthropic Claude as a stretch goal. This keeps the core project focused while leaving a clear path for additional providers later.

## 7. Risks and Mitigations

| Risk | Why It Matters | Mitigation |
| --- | --- | --- |
| Token limits / API cost | Many candidates times long contexts can hit rate limits or become expensive | Transmit only metadata summaries; cap candidates at 3-6; cache intermediate results between calls |
| LLM hallucinating tool calls | Agent could call non-existent tools or construct invalid arguments | Enforce tool-calls-only execution mode; strict schema validation on every tool call; explicit failure on schema mismatch |
| Wrong model selected | Agent selects a model that performs poorly on the user's data | Evaluation-based selection over a controlled candidate set; all candidates and metrics logged; user can inspect and override |
| Index/exogenous edge cases | Unusual index types or misaligned exogenous inputs can produce silent wrong results | Metadata summarizer validates index type before candidate proposal; alignment checks derived from PR #119 and #133 fixes |

