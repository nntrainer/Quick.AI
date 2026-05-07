# OpenAI-Compatible Server

Quick.AI can run a local HTTP server with an OpenAI-compatible REST surface for
causal LM models. The server loads one model directory at startup and exposes
completion endpoints.

## Build

```bash
meson setup build -Denable-fp16=true -Dthread-backend=omp -Domp-num-threads=4
ninja -C build quick_dot_ai_server
```

On Windows, use the existing helper:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_windows.ps1
```

The executable is built as `quick_dot_ai_server` or
`quick_dot_ai_server.exe`.

Android builds include the same executable target through `jni/Android.mk`.
Run `./build_android.sh` to produce `jni/libs/arm64-v8a/quick_dot_ai_server`
alongside the existing Android binaries.

When running directly from `build-win`, make sure the generated DLL directories
are discoverable in `PATH`. One convenient development-shell setup is:

```powershell
$dllDirs = Get-ChildItem -Path build-win -Recurse -Filter *.dll |
  ForEach-Object { $_.Directory.FullName } |
  Sort-Object -Unique
$env:PATH = ($dllDirs -join ';') + ';' + $env:PATH
```

## Run

```bash
./build/quick_dot_ai_server ./res/qwen3/qwen3-4b \
  --host 127.0.0.1 \
  --port 8000 \
  --model-id qwen3-4b
```

Options:

- `--host`: IPv4 address to bind. Defaults to `127.0.0.1`.
- `--port`: TCP port to bind. Defaults to `8000`.
- `--model-id`: model id returned in OpenAI-compatible responses. Defaults to
  `quick.ai`.
- `--no-chat-template`: disables `tokenizer_config.json` chat-template
  rendering for chat requests.
- `--verbose`: prints prompt/output and model metrics during generation.

## Endpoints

### `GET /health`

Returns server health and the loaded model id.

### `GET /v1/models`

Returns a single OpenAI-style model entry for the loaded model.

### `POST /v1/chat/completions`

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "Explain Quick.AI in one sentence."}
    ]
  }'
```

Set `"stream": true` to receive an OpenAI-style Server-Sent Events response.
The current implementation is buffered: generation still completes internally
before the server emits SSE chunks, then finishes with `data: [DONE]`.

### `POST /v1/completions`

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b",
    "prompt": "Quick.AI is"
  }'
```

## Compatibility Notes

- `stream: true` is accepted on `/v1/chat/completions` and
  `/v1/completions`, but it is buffered compatibility streaming. The server
  emits OpenAI-style SSE after generation finishes; it does not yet stream each
  token as soon as it is produced.
- Per-request `max_tokens`, `top_p`, and `temperature` do not override the
  loaded model's `nntr_config.json` and `generation_config.json` values yet.
  `temperature > 0` or `do_sample: true` enables sampling for the run.
- Requests are accepted concurrently, but model inference is serialized behind
  a mutex because the loaded model instance is stateful.
- The server implementation is native C++ and uses platform socket APIs guarded
  for Windows and POSIX-style targets. It is intended to build on Windows,
  Ubuntu/Linux, and Android NDK; Android deployment still needs normal runtime
  setup such as pushing model files to the device and allowing the chosen bind
  address/port for the process.

## Streaming TODO

- Add a token callback in the CausalLM generation loop so generated text can be
  delivered as soon as `registerOutputs()` produces it.
- Add a socket streaming path that writes SSE chunks without building the full
  response body first.
- Emit optional `stream_options.include_usage` usage chunks compatible with
  OpenAI clients.
- Decide how to surface generation errors after an SSE stream has already
  started.
- Add integration tests that consume `text/event-stream` responses from an
  OpenAI-compatible SDK/client.
