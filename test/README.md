# Quick.AI unit tests

GoogleTest-based unit tests for the Quick.AI runtime. Built only when the
meson option `enable-test` is set:

```bash
meson setup build -Denable-test=true
ninja -C build
meson test -C build --print-errorlogs
```

## Test suites

| File | What it covers | Needs model weights? |
|---|---|---|
| `unittest_llm_util.cpp` | Logits post-processing helpers declared in `llm_util.hpp` (`generate_multi_tokens`, `applyRepetitionPenalty`, `applyBadWordsPenalty`, `applyTKP`). | No |
| `unittest_factory.cpp` | `quick_dot_ai::Factory` register / create / override / print. Uses a `UT/` key prefix so it does not collide with production model registrations. | No |
| `unittest_model_config.cpp` | `registerModelArchitecture`, `registerModel`, `register_builtin_model_configs` (Qwen3-0.6B built-in), `setOptions` flag combinations, null-parameter validation. | No |
| `unittest_tokenizer.cpp` | HuggingFace tokenizer wrapper from `tokenizers_cpp.h`: `FromBlobJSON`, `Encode`/`Decode` roundtrip, `IdToToken`/`TokenToId`, `EncodeBatch`. | Only `tokenizer.json` |
| `unittest_causal_lm_api.cpp` | Causal-LM C API lifecycle (error codes for out-of-order calls, null parameters) plus an end-to-end smoke test that loads the model, runs a prompt and fetches `PerformanceMetrics`. | Yes, Qwen3-0.6B Q4_0 |

Tests that need files on disk call `GTEST_SKIP` when the files are absent, so
the suite remains green on machines where the model bundle has not been
fetched.

## Staging the Qwen3-0.6B Q4_0 weights

```bash
./test/scripts/download_qwen3_0.6b.sh
```

This script:

1. Shallow-clones `github.com/eunjuyang/nntrainer-causallm-models` into
   `.test_cache/` at the repo root.
2. Runs the shipped `combine.sh` to reassemble
   `nntr_qwen3_0.6b_w4e4a32.bin` from the ~95 MB split parts and verifies
   the embedded SHA256.
3. Runs `sha256sum -c` against `SHA256SUMS` for the remaining assets.
4. Copies the reassembled binary together with `config.json`,
   `nntr_config.json`, `generation_config.json`, `tokenizer.json`,
   `tokenizer_config.json`, `vocab.json`, `merges.txt` into
   `./models/qwen3-0.6b-w16a16/`.

The directory name `qwen3-0.6b-w16a16` matches
`resolve_model_path("QWEN3-0.6B", CAUSAL_LM_QUANTIZATION_W16A16)`, a
combination that is intentionally unregistered in the built-in config
table. This routes `loadModel` onto the "external config" code path
(CASE 2 in `api/causal_lm_api.cpp`), where the API reads
`config.json` / `nntr_config.json` from the staged directory and uses the
actual weights file name (`nntr_qwen3_0.6b_w4e4a32.bin`) declared inside
`nntr_config.json`.

## One-shot runner

```bash
./test/scripts/run_unittests.sh [build_dir]
```

Stages the model (unless `QUICKAI_SKIP_MODEL_DOWNLOAD=1`), reconfigures
meson with `-Denable-test=true`, runs `ninja` and finally `meson test`.

## Known limitation: Qwen3-0.6B Q4_0 + tied embedding

The public Qwen3-0.6B bundle ships with `tie_word_embeddings=true` and a Q4_0
LM head, but `layers/tie_word_embedding.cpp` currently accepts only Q6_K or
FP32 weights for the tied path and throws `Tieword embedding is not supported
yet for the data type` at the first decode step. `unittest_causal_lm_api`
recognises this specific failure (`runModel` returns `INFERENCE_FAILED`),
emits `GTEST_SKIP` and the suite remains green. Once NNTrainer adds Q4_0
support for the tied path the test will switch back to PASS automatically.

## Environment overrides

- `QUICKAI_TEST_MODEL_DIR` - point `unittest_causal_lm_api` at an
  alternate model directory.
- `QUICKAI_TEST_TOKENIZER_JSON` - point `unittest_tokenizer` at an
  alternate `tokenizer.json`.
- `QUICKAI_SKIP_MODEL_DOWNLOAD=1` - tell `run_unittests.sh` not to fetch
  the model (model-dependent tests will then `SKIP`).
