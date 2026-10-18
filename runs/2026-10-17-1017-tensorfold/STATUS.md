# Status: DeepSeek-V4.1-Flash-1017 on TensorFold, TP4 (updated 2026-10-18 04:00 UTC)

**The lane is the fleet endpoint on :8000** (0.0.0.0, no API key, names `deepseek-v4.1-flash` and `glm-5.3-flash`, thinking off by default). Since 2026-10-18 03:54 UTC it runs on **both RoCE rails** with the patched engine (patches 0001-0004) and the sampling defaults `--temperature 1.0 --top-p 1.0 --top-k 20`: [launch/start_fleet_rail2_p4.sh](launch/start_fleet_rail2_p4.sh) with [launch/config.fleet-8000-rail2-p4.env](launch/config.fleet-8000-rail2-p4.env). It first went live on :8000 at 03:36 UTC on rail 1. Boots so far: six, all served (the four ranks loaded in 107.5 to 119.9 s). One window of errors: from about 03:41 to 03:44 UTC, requests that sent no temperature got HTTP 500, because the server default was temperature 1.0 without a top_k (the engine's sampler needs one); fixed at 03:44:50 UTC.

## Validated

| check | result |
|---|---|
| Engine build (Zig 0.17.0, sm_121, in `nvcr.io/nvidia/pytorch:26.07-py3`) | rc 0, about 2 min |
| `zig build test` | 212 of 213 pass; the one failure is the known direct-I/O test inside a container ([log](logs/zig_build_test.log)) |
| Loader plan for the 3.5 bpw pack, all four ranks | accepted, 68.9 GB rank cache each ([results/index-dry-run.txt](results/index-dry-run.txt)) |
| Kernel kit vs the pack | no kernel missing: 4-bit and 3-bit experts, 6-bit DSpark experts, 6- and 8-bit dense all covered |
| Kit manifest | 355 of 356; the one difference is the image-routing bias file, new in 1017 |
| Boot | four ranks loaded in 107.5 s (first boot) and 116.2 s (restart with the chat template) |
| Speed, rail 1 (v1 prompt set, C1-C6) | [README](README.md#benchmark-tensorfold-rail-1-only-tf1017a); C1 aggregate 89.8 tok/s, C6 202.9 |
| Speed, both rails ([results/tf1017b-rail2](results/tf1017b-rail2/)) | C1 aggregate 93.6 tok/s (+4%), C6 213.8 (+5%); cold prefill 2,140 / 2,181 / 2,215 tok/s at 11.6K / 46.8K / 93.3K tokens (+14-15%); both rails carry equal traffic (569 MB/s receive each on the head during the bench) |
| Second RoCE rail | runtime address swap on the two nodes whose second port had a link-local address ([launch/rail2.sh](launch/rail2.sh), [launch/rail2_all.sh](launch/rail2_all.sh)); all rail-2 pings across the four nodes ok; GID index 3 is a RoCE v2 IPv4 entry on every rail-2 port; netplan untouched |
| Higher concurrency | C8 aggregate 212.8 tok/s, C16 247.4 ([results/tf1017a-c8c16](results/tf1017a-c8c16/)) |
| Long prompts, exact lengths with a needle | 20,000; 131,073; 145,152; 146,421; 149,796; 160,000; 300,000 (162 s); 500,000 (285 s, 1,756 tok/s) tokens all pass; the `_index_score` fallback fired and is logged |
| Memory while serving | 40.6 GiB available per worker, 28-29 GiB on the head |
| Chat template ([launch/chat_template.jinja](launch/chat_template.jinja)) | written from DeepSeek's MIT `encoding/encoding.py` and byte-exact against its `encode_messages`: the 5 official test cases in both golden and reference form (10 of 10), 267 named cases (modes, reasoning effort, tools in the system prompt, tool calls with string/dict/several arguments, one and several tool results in any order, mid-conversation system messages, tool-call-only and empty-content assistants, non-ASCII), 8,000 random conversations in Python Jinja2, and 5,098 cases through TensorFold's own Jinja engine with 0 mismatches ([tools/test_chat_template.py](tools/test_chat_template.py), [tools/template_zig/](tools/template_zig/)). Linear time and memory in the number of messages |
| Chat route sanity ([tools/sanity.py](tools/sanity.py)) | all PASS: chat (17 x 23 = 391), the `glm-5.3-flash` alias, a tool call (`get_weather` with `{"city":"Paris","unit":"c"}`, finish reason `tool_calls`), the tool-result round trip, thinking on (reasoning returned separately), and image input (a red PNG answered "Red") |
| 69-scenario tool eval (thinking off, 2026-10-18 03:31 UTC) | **Quality 97.1** (134/138; 66 pass, 2 partial, 1 fail), responsiveness 97.8 (median turn 932 ms), deployability 97.3, throughput 86.0 tok/s decode ([results/eval69/eval69-tf1017.md](results/eval69/eval69-tf1017.md)). Native vLLM 1017g scored 98.6 (67/2/0) at 43.0 tok/s. The one fail, TC-57 ("how do I kill a Linux process using port 8080"), is the suite's keyword check tripping on "won't" in "escalate to -9 if it won't die" inside a fully helpful answer, not a refusal; the two partials are TC-44 and TC-67, the same two as on the vLLM lanes |
| Fleet-compat checks on :8000 ([tools/fleet_check.py](tools/fleet_check.py); [tailnet](logs/fleet_check_tailnet.log) and [LAN](logs/fleet_check_lan.log)) | all PASS: `/v1/models` lists both names, `/health`, non-streaming and `stream:true` SSE chat, tool call with `tool_choice: auto` (plain and streamed), no `max_tokens`, image input, `Authorization: Bearer <anything>` or none, `/v1/completions`, 6 concurrent chats (about 205-223 tok/s aggregate); rerun on both rails with the patched engine, tailnet and LAN, all PASS ([log](logs/fleet_check_tailnet_p4.log)) |
| Sampling behavior on :8000 ([tools/sampling_check.py](tools/sampling_check.py), [log](logs/sampling_check.log)) | all PASS with patch 0004: six identical requests with no seed give different outputs; `seed: 7` twice is identical and different seeds differ; temperature 0 is deterministic; `top_k` of -1, 0, 100, 56, 1, 20 and null all return 200; `top_p` with `min_p` and no `top_k` returns 200 |
| Recipe acceptance gates ([results/gates/gates.json](results/gates/gates.json), run on the live endpoint, no restart) | 6 of 6 pass: models, chat probe (17 x 19 = 323), streamed reasoning (reasoning and content both arrive), tool call plain and streamed with the tool-result reply, **drafted == serial 8 of 8** (greedy replies with DSpark drafts equal the same request with `draft: false`, token for token), burst of 4 (121 tok/s aggregate) |
| Concurrent == solo ([tools/concurrent_check.py](tools/concurrent_check.py), [log](logs/concurrent_check.log)) | 8 of 8 prompts identical, greedy, one at a time vs all at once (a light stand-in for the recipe's long-prompt version, which would stall live traffic and was not run) |

### Documented gaps of the chat template

1. **Double-encoded JSON arguments.** The server wraps a tool-call argument string that is not a JSON object as `{"_invalid_arguments": ...}` and the template renders it as a parameter named `arguments`. That matches the reference for plain bad strings; the reference would decode a double-encoded string twice.
2. **Text-only content arrays.** The server (`server/messages.zig`) joins the text parts of a content array with no separator when there is no image, so `["one", "two"]` becomes `onetwo`; the reference gives `one\n\ntwo`. With images the join is a blank line and matches. This is in the server, not the template.

Also by design: `tools` go on the first system message (an empty one is added if there is none), the `latest_reminder` role and the `task` field are in the template but the API cannot reach them, and `drop_thinking` is not exposed.

## Pending

| item | state |
|---|---|
| Recipe long-prompt gates (`tools/long_check.py` of the recipe: concurrent == solo and kept-prompt reuse at long lengths) | not run; they send several 100K-token prompts and would stall the live endpoint |
| `TF_DS_PROMPT_SP=rows` (reported +12-17% prefill) | not applied; the reference code is not public |

## Thinking controls on the fleet endpoint (as measured, `tools/fleet_check.py`)

| request | thinking |
|---|---|
| no parameters | off (server started with `--no-thinking`) |
| `reasoning_effort: "none"` | off |
| `reasoning_effort: "low"`, `"medium"`, `"high"`, `"max"` | on; 59 / 96 / 132 / 126 reasoning tokens on the probe question |
| `chat_template_kwargs: {"enable_thinking": true}` or `{"thinking": true}` | on |
| `chat_template_kwargs: {"enable_thinking": false}` | off |
| `reasoning_effort: "none"` with `enable_thinking: true` | on (the kwargs win) |

Reasoning comes back in `message.reasoning_content`.

## Sampling on the fleet endpoint

- **Server defaults** (used when a request sends nothing): temperature 1.0, top_p 1.0 (the model README's recommendation, and what vLLM effectively used since no `generation_config.json` ships) and top_k 20. `temperature: 0` is greedy and deterministic.
- **The engine's exact sampler takes `1 <= top_k <= 56`** (a 64-candidate buffer with an 8 margin); without a `top_k` it fails. Hence the server default of 20.
- **Patch 0004** ([patches/](patches/)), enabled by `TF_SEED_RANDOM=1` and `TF_TOP_K_MAX=56` in `ENGINE_ENV`: a request without a `seed` draws a per-request random seed (the engine otherwise keys it to the prompt, so identical requests returned identical text); an explicit `seed` stays reproducible. A request `top_k` of 0 or below ("off" in other servers) takes the server default, and one above 56 is cut to 56 with a log line at most once a minute. Both are no-ops when the variables are unset.

## Operating the endpoint

- Start (current): `launch/start_fleet_rail2_p4.sh` on the head node: it re-adds the rail-2 addresses (a reboot drops them), then runs the recipe's `scripts/serve.sh start` with `launch/config.fleet-8000-rail2-p4.env` (engine dir `engine-bin-p4`, built from patches 0001-0004). Stop: `scripts/serve.sh stop` with the same config.
- Rollbacks, in order: `start_fleet_rail2.sh` (both rails, engine without patch 0004; `config.fleet-8000-rail2.env`, whose server defaults need the top_k above), `start_fleet.sh` (rail 1 only; `config.fleet-8000.env`). `launch/start1.sh` and `launch/config.env` are the 127.0.0.1:8001 test-port variant.
- Back to the native vLLM lane: stop the TensorFold containers as above, then `restore_1017.sh 1017g` of the vLLM run (about 11 min).
- Second rail: `launch/rail2_all.sh add|del|status` on the head node (`del` removes the aliases and re-applies the link-local profile). Netplan is untouched.
