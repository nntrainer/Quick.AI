# CLAUDE.md

This file briefs Claude Code (and any new contributor) on Quick.AI's
conventions before making changes. Read it top-to-bottom — it is
deliberately short.

The bulk of this document is contributor-facing guidance that applies
to humans and AI coding agents alike. Agent-specific rules (Claude
Code, other coding agents) live in [a single dedicated section at the
bottom](#for-ai-coding-agents).

---

## Project at a glance

Quick.AI is a production-grade **on-device causal-LM inference engine**
built on top of [NNTrainer](https://github.com/nntrainer/nntrainer). It
targets Linux and Android (arm64-v8a) with hand-tuned ARMv8.2-a (FP16,
dotprod, i8mm) and AVX2 kernels, and runs MoE models (Qwen3-MoE 30B,
GPT-OSS 20B/120B) on a phone via Flash Storage Utilization (FSU) —
experts stream from disk only when their tokens fire.

| Item | Value |
|---|---|
| Language | C++17 (and a little C for the public API) |
| Build system | Meson + Ninja (`meson_version >= 0.55.0`) |
| Submodule | `subprojects/nntrainer/` (pinned commit, meson subproject) |
| C++ namespace | `quick_dot_ai` (do **not** reintroduce the old `causallm`) |
| Brand spelling | "Quick.AI" in human copy, `quick_dot_ai` in identifiers |
| License | Apache-2.0 |

Sources of truth that go deeper than this file:

- [`README.md`](README.md) — user-facing intro, demos, quick start.
- [`docs/architecture.md`](docs/architecture.md) — Mermaid diagram, per-binary / per-plugin breakdown, design rationale.
- [`models/README.md`](models/README.md) — model author guide.
- [`api/README.md`](api/README.md) — C API reference.
- [`benchmarks/README.md`](benchmarks/README.md) — Android benchmark tooling.

---

## Repository map

```
api/             Stable C API surface (libquick_dot_ai_api.so).
                 ABI-stable — do NOT rename symbols or change enum values.
layers/          Per-layer plugin .so's (rms_norm, swiglu, qkv,
                 mha_core, lm_head, tie_word_embedding, embedding_*,
                 reshaped_rms_norm). Each builds as its own
                 libquick_dot_ai_<name>_layer.so.
models/          causal_lm + transformer base classes; per-family
                 causal LMs (qwen2, qwen3, qwen3_moe, gpt_oss,
                 gemma3). The *_cached_slim and *_slim_moe variants
                 enable FSU.
factory.h        Model registry. Every new family must be wired here
                 so loadModel can dispatch by ModelType.
main.cpp         quick_dot_ai_run executable.
quantize.cpp     quick_dot_ai_quantize executable
                 (Q4_0 / Q4_K / Q6_K / FP16).
huggingface_tokenizer.{cpp,h}  Tokenizer adapter over tokenizers-cpp.
llm_util.{cpp,hpp}             Generation-loop helpers.
jni/             Android.mk + prepare_encoder.{sh,ps1}.
build_android.sh         Core Android build (NDK + Rust target).
build_api_lib.sh         libquick_dot_ai_api.so for Android.
build_test_app.sh        quick_dot_ai_test_api for Android.
install_android.sh       adb push to /data/local/tmp/quick_dot_ai/.
benchmarks/      Android perf tooling (benchmark_android.py et al.).
docs/            architecture.md + demo GIFs.
res/             Drop model directories here (config.json,
                 tokenizer.json, nntr_config.json, weight .bin, …).
.clang-format    clang-format 14 style — CI fails on diffs.
.github/workflows/   ci-linux, ci-android, cpp-linter, codeql,
                     check_count, labeler.
subprojects/nntrainer/   Vendored NNTrainer, built lean
                         (enable-app=false, enable-test=false,
                         enable-tflite-{backbone,interpreter}=false).
meson.build / meson_options.txt   Top-level build wiring.
```

---

## Build & verify

### Linux (mirrors `ci-linux.yml`)

```bash
sudo apt-get install -y libopenblas-dev libflatbuffers-dev \
    flatbuffers-compiler libiniparser-dev libomp-dev cmake \
    build-essential pkg-config
pip install meson ninja

meson setup build -Denable-fp16=true -Dthread-backend=omp \
                  -Domp-num-threads=4
ninja -C build
```

Expected artifacts under `build/`:

- `libquick_dot_ai.so`
- `quick_dot_ai_run`, `quick_dot_ai_quantize`, `quick_dot_ai_test_api`
- `layers/libquick_dot_ai_*_layer.so` (one per layer plugin)

Smoke test the runner with a model directory under `res/<name>/`:

```bash
export OMP_NUM_THREADS=4 OMP_WAIT_POLICY=active \
       OMP_PROC_BIND=true OMP_PLACES=cores
./build/quick_dot_ai_run ./res/qwen3/qwen3-4b/
```

### Android (mirrors `ci-android.yml`)

Prereqs: NDK r26d (export `ANDROID_NDK`), CMake, Rust with the
`aarch64-linux-android` target, `adb`.

```bash
export ANDROID_NDK=/path/to/android-ndk
./build_android.sh        # core: libquick_dot_ai_core.so + binaries
./build_api_lib.sh        # libquick_dot_ai_api.so
./build_test_app.sh       # quick_dot_ai_test_api
./install_android.sh      # adb push to /data/local/tmp/quick_dot_ai/
```

### Format every changed C/C++ file before committing

```bash
clang-format-14 -i path/to/changed.cpp path/to/changed.h
```

The `cpp-linter.yml` workflow runs `clang-format 14` against
`.clang-format` and gates the PR; `subprojects/` is excluded.

---

## Commit rules (HARD — CI / DCO will block violations)

Every commit MUST end with a `Signed-off-by:` trailer. Use a real name
and an email reachable by the author. The trailer is the project's
DCO sign-off and is non-negotiable.

### Format

```
[<Component>] <imperative summary, <= 72 chars, no trailing period>

<Body explaining WHY the change exists, wrapped at ~72 chars.
Use bullets ("- ...") for multiple points. Reference behavior
and motivation, not the file list — git already shows the diff.>

Signed-off-by: Your Name <you@example.com>
```

Always sign off with `git commit -s` (or include the trailer manually
when using the GitHub API). Co-authored work gets one additional
`Signed-off-by:` line per author.

### Subject conventions

- **Imperative mood**: "add", "fix", "rename" — not "added" / "adds".
- **Component prefix** when the change is local to a subsystem.
  Prefixes already used in the history (re-use, don't invent):
  `[CausalLM]`, `[api]`, `[Android.mk]`, `[neuralnet]`, `[script]`,
  `[Docs]`, `ci`, `ci(android)`, `ci(codeql)`, `ci(linux)`. A bare
  imperative subject (no bracket) is also accepted for repo-wide
  changes — see commit `526f361` ("Rename project to Quick.AI; …").
- **<= 72 chars**, no trailing period.
- Keep brand spelling consistent: human copy → "Quick.AI", code
  identifiers → `quick_dot_ai`.

### Body conventions

- Explain *why* (motivating bug, requirement, constraint), not
  *what the diff does*.
- Wrap at ~72 chars. One blank line between subject and body, and
  between body and the trailers.
- Bullets start with `- ` and stay short.
- Call out explicitly when the change touches CI, the build system,
  the public API/ABI, or the on-device install layout.

### Examples drawn from the history

```
[CausalLM] fix mmap read in tie-word-embedding
ci(android): pass user-writable prefix to nntrainer package build
ci(codeql): pin source-root and exclude vendored trees from scan
Rename project to Quick.AI; unify identifiers to quick_dot_ai
```

### Don'ts

- Do **not** use `--no-verify` to skip hooks.
- Do **not** `--amend` an already-pushed commit; create a new one.
- Do **not** force-push to `main` or any shared branch.
- Do **not** add `Co-Authored-By:` trailers that imply authorship a
  tool does not have.

---

## Branching & PR workflow

- Work on a topic branch; never commit directly to `main`. Branch
  naming patterns observed in the repo:
  `feat/...`, `bugfix/...`, `ci/...`, `unittest/...`, `claude/...`.
- A PR is gated by:
  | Workflow | What it does |
  |---|---|
  | `ci-linux.yml` | Meson + Ninja on Ubuntu 22.04 & 24.04. |
  | `ci-android.yml` | NDK r26d, arm64-v8a, Rust `aarch64-linux-android`. |
  | `cpp-linter.yml` | clang-format 14 against `.clang-format` (subprojects/ ignored). |
  | `codeql.yml` | CodeQL c-cpp + python; vendored trees excluded by `.github/codeql/codeql-config.yml`. |
- `check_count.yml` + `labeler.yml` toggle `Need Review` and
  `PR/READY2MERGE` based on **2 approving reviewers** (Quick.AI's
  threshold; nntrainer uses 3).

---

## Code style essentials

Read `.clang-format` for the canonical rules. The non-obvious bits:

- 2-space indent, **80-column** hard limit, tabs forbidden.
- `BreakBeforeBraces: Attach`, `PointerAlignment: Right`
  (`int *p`, not `int* p`).
- `SortIncludes: CaseSensitive` — keep includes sorted; let
  clang-format do it.
- C++17 (`-std=c++17`), C `gnu89` for the public C API.
- Keep new C++ symbols inside `namespace quick_dot_ai`.

### ABI stability

`api/causal_lm_api.h` is the integration seam used by Android JNI,
iOS, and server embedders. The following are **frozen**:

- Symbols: `loadModel`, `runModel`, `getPerformanceMetrics`.
- Enums: `BackendType`, `ModelType`, `ModelQuantizationType`
  (and the `CAUSAL_LM_*` enumerator names / values).

Don't rename them, don't reorder enumerators, don't change numeric
values. If you need a new entry, append at the end.

### Layers as plugins

Each transformer building block under `layers/` builds as its own
`shared_library` (named `libquick_dot_ai_<name>_layer.so`). New
layers must:

1. Add `<name>.{cpp,h}` under `layers/`.
2. Declare the build target in `layers/meson.build`.
3. Append the resulting `quick_dot_ai_<name>_dep` to
   `quick_dot_ai_layer_dependencies` in the top-level `meson.build`.

---

## Adding a new model family

1. `models/<family>/<family>_causallm.{h,cpp}` deriving from the
   appropriate base in `models/causal_lm.{h,cpp}` /
   `models/transformer.{h,cpp}`.
2. `models/<family>/meson.build`; append the family's sources to
   `models/meson.build` (and to `quick_dot_ai_src` /
   `quick_dot_ai_inc` at the top level if needed).
3. Register the family + a new `ModelType` enumerator in
   [`factory.h`](factory.h) so `loadModel` can dispatch to it.
4. Optional: implement custom layers under `layers/` and wire their
   deps into the top-level `meson.build`.
5. Drop a runnable model directory under `res/<family>/<variant>/`
   containing `config.json`, `generation_config.json`,
   `tokenizer.json`, `tokenizer_config.json`, `vocab.json`,
   `nntr_config.json`, and the NNTrainer `.bin` weight file
   referenced from `nntr_config.json`.

Verify: `meson setup build && ninja -C build && \
./build/quick_dot_ai_run ./res/<family>/<variant>/`.

---

## Anti-patterns / things to NOT do

- Do not reintroduce the old `causallm` C++ namespace, the
  `nntr_causallm` / `nntr_quantize` / `test_api` binary names, the
  `libcausallm.so` library name, or the
  `/data/local/tmp/nntrainer/causallm` install path. They were all
  renamed in commit `526f361` and the rename is enforced by CI.
- Do not add system deps Quick.AI doesn't actually use (e.g.
  `tensorflow2-lite-dev`, `nnstreamer-dev`). NNTrainer is pulled in
  with `enable-tflite-backbone=false`, `enable-tflite-interpreter=false`,
  `enable-app=false`, `enable-test=false` — keep that surface lean.
- Do not commit meson auto-generated wrap redirects under
  `subprojects/*.wrap` — they are regeneration artifacts and are
  already gitignored.
- Do not relax `BreakBeforeBraces`, `IndentWidth`, or `ColumnLimit`
  in `.clang-format` to make a diff fit; reformat the diff instead.
- Do not assume Q4_0 quantized files are portable across
  architectures. Quantize on the same ISA (ARM vs x86_64) you serve
  from.
- Do not push to `main` directly, and do not force-push to any
  shared branch.

---

## For AI coding agents

This section collects rules that apply specifically to AI coding
agents (Claude Code, and any other tool that drives commits/PRs on
behalf of a human). It supplements — does not replace — every other
rule above. If you are a human reading this file, you can skip it.

- **Stay on the topic branch the user named for the session.** Do
  not open new branches, rebase shared branches, or rewrite history
  on branches the user did not authorize.
- **Do not open a pull request unless the user explicitly asks for
  one.** Pushing the branch is fine; opening / merging the PR is the
  user's call.
- **Never force-push to `main` or any shared branch**, and avoid
  force-push on user-visible feature branches unless the user asks.
- **Keep the DCO trailer honest.** Sign off as the human running the
  session (or as instructed). Do not invent `Co-Authored-By:`
  trailers.
- **Treat `api/causal_lm_api.h` and `factory.h` as load-bearing.**
  Surface any change to those files explicitly in the PR / commit
  message so a human reviewer can sanity-check the ABI impact.
- **Run the local format / build smoke tests** described above
  before pushing, even when CI will rerun them — failing CI on
  trivial whitespace wastes a review round trip.
