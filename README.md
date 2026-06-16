# Auto Scripts

Automated serve + downstream (bench / profile / readable / eval) for vLLM and
SGLang, shared across tickets.

## Run container

```bash
bash docker.sh        # or: bash podman.sh <CONTAINER_NAME>
```

## Layout

```
auto-script/
├── env.yaml                 # GLOBAL base config (model.paths, defaults)
├── serve.sh                 # vLLM server startup (preset-driven)
├── presets/<family>/...     # vLLM presets; <family> = model.paths key (glm5, deepseek)
├── datasets/                # datasets that need a manually-added path/repo
│   └── repobench/
│       ├── repobench_65k.json   # dataset_path
│       └── repobench/           # repobench_dir (eval tool repo, has eval.py)
├── common/                  # REUSABLE, ticket-agnostic
│   ├── helper.sh            # paths, env merge, serve/kill/wait, resolvers
│   ├── run_all.sh           # shared driver (readable + eval) with ticket_phases hook
│   ├── auto_readable_template.sh    # readable serve/kill + shared prompts
│   ├── auto_readable_{completion,chat,pychat}.sh  # readable methods (concretes)
│   ├── readable_prompts.txt         # shared smoke-test prompts
│   ├── readable_pychat.py           # client-side chat-template readable
│   └── auto_eval.sh         # repobench (generate+eval) + lm_eval, run-count driven
└── bench_<ticket>/          # ticket-specific
    ├── env.yaml             # OVERRIDES merged on top of ../env.yaml
    ├── run_all.sh           # thin wrapper -> common/run_all.sh
    ├── auto_bench.sh        # (mv4433) bench sweep — ticket-specific
    ├── auto_profile.sh      # (mv4433) profile sweep — ticket-specific
    └── serve_sglang_*.sh, data files, ...
```

## Config — base + override

A ticket's `env.yaml` is **deep-merged on top of** the global `env.yaml`
(arrays replace, nested maps merge, base values inherited). A ticket only lists
what it changes — usually `backends` and the `eval.datasets.*.runs` it needs.

Global `env.yaml` (base): `model.paths`, profiler flags, and every dataset at
`runs: 0` with repobench's `dataset_path` / `repobench_dir` pointing into
`datasets/` (relative paths resolve against `auto-script/`).

`bench_mv4433/env.yaml` (override):

```yaml
backends:
  - name: vllm
  # vLLM gets its model from the preset family. sglang has no preset, so its
  # entry must supply a serve_script (relative to auto-script/) and a model key
  # (-> model.paths.<model>):
  # - name: sglang
  #   serve_script: bench_mv4433/serve_sglang_ds3.2.sh
  #   model: deepseek
eval:
  datasets:
    repobench: { runs: 1 }
    mmlu:      { runs: 1 }
    gsm8k:     { runs: 1 }
```

Phase order is per-ticket via `phases:` in the ticket env.yaml (default
`[readable, eval]` if omitted). `readable`/`eval` are handled by the shared
driver; any other name (e.g. `bench`, `profile`) is delegated to the ticket's
`ticket_phase()`. Example — eval before readable: `phases: [eval, readable]`.
`RUN_*=0` still toggles a phase off regardless of order.

`runs` = number of times to run (0 = skip). One repobench run = generate
predictions + compute metrics (wrapped). mmlu/gsm8k/longbench/longbench2 are
lm_eval tasks run N times each.

gsm8k supports a `method` field: `lm_eval` (default, `local-completions`) or
`script` — the latter runs `datasets/gsm8k/gsm8k.py` (chat-completions runner
with a custom task yaml). Example: `gsm8k: { runs: 1, method: script }`.

`eval.readable` filters which readable smoke tests the readable phase runs:
`completion` (raw /v1/completions), `chat` (/v1/chat/completions), `pychat`
(chat template applied client-side, then /v1/completions — sidesteps a wrong
server chat template). Example: `readable: [pychat]`.

## Run

```bash
cd bench_mv4476
bash run_all.sh                                  # all backends + phases from env.yaml
PRESET=glm5/dp8ep8/bs64-moreh.yaml bash run_all.sh
BACKEND=sglang RUN_EVAL=0 bash run_all.sh
RUN_BENCH=0 RUN_PROFILE=0 bash run_all.sh        # (mv4433) skip ticket phases
```

Run a single phase standalone (it serves/kills its own server):

```bash
PRESET_YAML=$PWD/../presets/glm5/dp8ep8/bs64-dg.yaml bash ../common/auto_eval.sh
AUTO_SERVE=0 BASE_URL=http://localhost:8000 bash ../common/auto_readable_chat.sh
```

## Debug (split serve vs downstream)

To debug a ticket, split serving from the downstream task — same log/result layout
as the auto_* phases. `serve.sh` stays in `common/` (ticket has a thin wrapper);
`eval.sh` / `readable.sh` are per-ticket and just run against the live server
(`AUTO_SERVE=0`), driven by whatever you set in `env.yaml`:

```bash
cd bench_mv4476
bash serve.sh                      # terminal 1: serve only (foreground, eval-ready)
PRESET=kimi2.6/... bash serve.sh   # match the preset you want to debug
# terminal 2 — uses the running server, logs under logs/<preset>/...:
PRESET=kimi2.6/... bash eval.sh        # runs the eval datasets enabled in env.yaml
PRESET=kimi2.6/... bash readable.sh    # runs the readable smoke tests
```

## Add a new ticket

1. `mkdir bench_<id>` and add an `env.yaml` with only the overrides (backends +
   the `eval.datasets.*.runs` you want — copy `bench_mv4476/env.yaml`).
2. Add a `run_all.sh`: for readable+eval only, copy `bench_mv4476/run_all.sh`
   (two lines). For extra ticket phases, define a `ticket_phases <backend>`
   function then `source ../common/run_all.sh` (see `bench_mv4433/run_all.sh`).
3. For a new dataset that needs a repo, add it under `datasets/` and point the
   `dataset_path` / `repobench_dir` fields at it (relative to `auto-script/`).
