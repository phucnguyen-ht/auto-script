---
name: new-bench-ticket
description: Scaffold a new auto-script bench/profile ticket (bench_mv<id>/) from a reference bench script, wiring it into the framework (common/auto_bench_template.sh + env.yaml + run_all). The framework is only scaffolding — the generated benchmark MUST reproduce the reference's exact benchmark command, arguments, and run flow, whatever tool it uses (vllm bench serve, a custom python bench, curl, sglang, ...). Invoke as /new-bench-ticket @reference_bench.sh [ticket-id].
disable-model-invocation: true
user-invocable: true
argument-hint: "@reference-bench-script [ticket-id]"
allowed-tools: Read, Write, Edit, Bash
---

# Generate a new bench/profile ticket from a reference script

Arguments:
- `$ARGUMENTS[0]` = the reference bench script (passed as `@file`; contents are in
  context — also Read it directly to be sure).
- `$ARGUMENTS[1]` = ticket id (e.g. `mv4600` or `4600`). **Optional — if missing,
  ASK the user.** The new folder is `bench_mv<id>` (use the id as-is if it already
  starts with `mv` or `bench_`).

Run inside the `auto-script` repo.

## ⛔ THE ONE RULE THAT MATTERS: benchmark-script correctness

Benchmark experiments are critical. The generated ticket must run the benchmark
**exactly like the reference** — same tool, same command, same arguments, same
order/conditions, same flow:

- **Reproduce the reference's benchmark invocation verbatim.** Do NOT assume it is
  `vllm bench serve`. It may be a custom python bench, `curl`, `sglang.bench_serving`,
  an in-tree `bench_serving*.py`, etc. Whatever the reference runs to produce one
  measurement, `run_one` must run the **same binary/script with identical flags and
  values** (dataset args, input/output lens, num-prompts and its derivation,
  concurrency, request-rate, seed, prefix-cache reset, warmup, headers, etc.).
- **Reproduce the reference's metric extraction faithfully.** Do NOT assume
  `agg_bench.py`. If the reference has its own extractor / summary format /
  aggregation, **port it** (keep the exact metrics, keys, order, and computation).
  Use `agg_bench.py` only if it yields equivalent output. (Example: `bench_mv4433`
  uses its OWN `bench_serving_glm4p5_65k.sh` + an inline `append_aggregates` python,
  NOT vllm bench serve / agg_bench.py — that's a valid, faithful ticket.)
- **Allowed adaptations (scaffolding only):** serving the model (framework serves
  via preset → `serve_backend`), the scenario loop / run-major structure, log/file
  layout, and `env.yaml`-driving the scenario params. Profiling: add the reference's
  profile switch (e.g. `--profile`) **only** in `MODE=profile`. A
  `--trust-remote-code` (or similar) guard if the model needs it.
- **Anything you cannot reproduce 1:1 → STOP and ask the user.** Never silently
  change a benchmark argument or the run flow.
- **Verify:** show a side-by-side of the generated `run_one` command vs the
  reference command before claiming done. **Never run real benchmarks (no GPU)** —
  validate with `bash -n` + a dry scenario-count check (`AUTO_SERVE=0`).

## Steps

1. **Read the framework anchors** (don't guess — read them):
   - `common/auto_bench_template.sh` — contract: a concrete sets `METHOD` and
     defines `load_scenarios` / `run_one <idx> <run>` / `aggregate`, then sources
     the template (template owns config/serve/kill + the run-major loop).
   - Example concretes to copy the *shape* from (each uses a DIFFERENT bench tool +
     extraction — pick the closest to the reference):
     - `bench_mv4526/auto_bench.sh` — `vllm bench serve`, custom dataset, cross-product
       scenarios, `agg_bench.py`.
     - `common/auto_bench_random.sh` — `vllm bench serve`, random dataset, list-of-dicts.
     - `bench_mv4433/auto_bench.sh` — custom `bench_serving_glm4p5_65k.sh` + inline
       `append_aggregates` (the "different tool + different extractor" example).
   - `common/helper.sh` — `resolve_backend/resolve_preset/resolve_model_path`,
     `serve_backend`, `reset_prefix_cache` (vLLM `/reset_prefix_cache` needs
     `VLLM_SERVER_DEV_MODE=1`; sglang `/flush_cache`), `profiler_config_json`,
     `yaml_get`/`yaml_list`, `is_enabled`.
   - `bench_mv4526/{env.yaml,run_all.sh,auto_profile.sh,run_all_full_presets_vllm.sh}`
     + base `env.yaml` — override/merge + preset-family resolution.

2. **Dissect the reference** ($ARGUMENTS[0]): the exact benchmark command + every
   arg, the scenario axes and how derived values (osl, num-prompts, …) are computed,
   dataset prep, prefix-cache reset, seed/warmup, and how it extracts/aggregates
   metrics (tool + metric keys + order).

3. **`bench_mv<id>/env.yaml`** (overrides on `../env.yaml`): `backends`,
   `phases: [bench, profile]`, and `bench:`/`profile:` `method: <name>` + a
   `<name>:` block encoding the reference's scenario params (cross-product lists or
   list-of-dicts — match the reference). Add new model families to base `env.yaml`,
   not here (family = preset's top-level folder).

4. **`bench_mv<id>/auto_bench.sh`** (concrete): set `METHOD`, env wiring
   (`ENV_YAML/LOG_ROOT/DATA_DIR`), and the reference's knobs verbatim. Define:
   - `load_scenarios` — build `SCENARIOS` + state arrays, reproducing the reference's
     loops/derivation exactly.
   - `run_one <idx> <run>` — emit the reference's benchmark command **verbatim** (its
     tool + args) writing into `<RUN_DIR>/run<run>/...`; add the profile switch only
     when `MODE=profile`; reset cache as the reference did.
   - `aggregate` — reproduce the reference's extraction (port its extractor, or
     `agg_bench.py "$RUN_DIR" "<exact,ordered,cols>"` if equivalent).
   - `source "${COMMON_DIR}/auto_bench_template.sh"`.

5. **`bench_mv<id>/auto_profile.sh`**: `MODE=profile exec bash "${SCRIPT_DIR}/auto_bench.sh"`.

6. **`bench_mv<id>/run_all.sh`**: copy `bench_mv4526/run_all.sh` (RUN_BENCH/RUN_PROFILE,
   `ticket_phase` → the two wrappers, sglang skip, cache clear) — only names differ.

7. **`bench_mv<id>/run_all_full_presets_vllm.sh`**: `PRESET=<family>/... bash run_all.sh`.

8. **Validate (no GPU):** `bash -n` all scripts (+ `ast.parse` any python); dry
   `AUTO_SERVE=0 SERVER_WAIT_TIMEOUT=1 bash bench_mv<id>/auto_bench.sh` → `scenarios=N`
   equals the reference's cross-product size; print the run_one-vs-reference command diff.

## Report

Files created, scenario count, and the benchmark-command parity diff. Flag anything
not reproduced 1:1 and ask before finishing.
