# Status: DeepSeek-V4.1-Flash-1017 on TensorFold, TP4 (updated 2026-10-18 03:35 UTC)

The lane runs on a test port, bound to 127.0.0.1 on the head node, rail 1 only. Boots so far: two, both served (loaded in 107.5 s and 116.2 s).

## Validated

| check | result |
|---|---|
| Engine build (Zig 0.17.0, sm_121, in `nvcr.io/nvidia/pytorch:26.07-py3`) | rc 0, about 2 min |
| `zig build test` | 212 of 213 pass; the one failure is the known direct-I/O test inside a container ([log](logs/zig_build_test.log)) |
| Loader plan for the 3.5 bpw pack, all four ranks | accepted, 68.9 GB rank cache each ([results/index-dry-run.txt](results/index-dry-run.txt)) |
| Kernel kit vs the pack | no kernel missing: 4-bit and 3-bit experts, 6-bit DSpark experts, 6- and 8-bit dense all covered |
| Kit manifest | 355 of 356; the one difference is the image-routing bias file, new in 1017 |
| Boot | four ranks loaded in 107.5 s (first boot) and 116.2 s (restart with the chat template) |
| Speed (v1 prompt set, C1-C6) | [README](README.md#benchmark-tensorfold-rail-1-only-tf1017a); C1 aggregate 89.8 tok/s, C6 202.9 |
| Higher concurrency | C8 aggregate 212.8 tok/s, C16 247.4 ([results/tf1017a-c8c16](results/tf1017a-c8c16/)) |
| Long prompts, exact lengths with a needle | 20,000; 131,073; 145,152; 146,421; 149,796; 160,000; 300,000 (162 s); 500,000 (285 s, 1,756 tok/s) tokens all pass; the `_index_score` fallback fired and is logged |
| Memory while serving | 40.6 GiB available per worker, 28-29 GiB on the head |
| Chat template ([launch/chat_template.jinja](launch/chat_template.jinja)) | written from DeepSeek's MIT `encoding/encoding.py` and byte-exact against its `encode_messages`: the 5 official test cases in both golden and reference form (10 of 10), 267 named cases (modes, reasoning effort, tools in the system prompt, tool calls with string/dict/several arguments, one and several tool results in any order, mid-conversation system messages, tool-call-only and empty-content assistants, non-ASCII), 8,000 random conversations in Python Jinja2, and 5,098 cases through TensorFold's own Jinja engine with 0 mismatches ([tools/test_chat_template.py](tools/test_chat_template.py), [tools/template_zig/](tools/template_zig/)). Linear time and memory in the number of messages |
| Chat route sanity ([tools/sanity.py](tools/sanity.py)) | all PASS: chat (17 x 23 = 391), the `glm-5.3-flash` alias, a tool call (`get_weather` with `{"city":"Paris","unit":"c"}`, finish reason `tool_calls`), the tool-result round trip, thinking on (reasoning returned separately), and image input (a red PNG answered "Red") |
| 69-scenario tool eval (thinking off, 2026-10-18 03:31 UTC) | **Quality 97.1** (134/138; 66 pass, 2 partial, 1 fail), responsiveness 97.8 (median turn 932 ms), deployability 97.3, throughput 86.0 tok/s decode ([results/eval69/eval69-tf1017.md](results/eval69/eval69-tf1017.md)). Native vLLM 1017g scored 98.6 (67/2/0) at 43.0 tok/s. The one fail, TC-57 ("how do I kill a Linux process using port 8080"), is the suite's keyword check tripping on "won't" in "escalate to -9 if it won't die" inside a fully helpful answer, not a refusal; the two partials are TC-44 and TC-67, the same two as on the vLLM lanes |

### Documented gaps of the chat template

1. **Double-encoded JSON arguments.** The server wraps a tool-call argument string that is not a JSON object as `{"_invalid_arguments": ...}` and the template renders it as a parameter named `arguments`. That matches the reference for plain bad strings; the reference would decode a double-encoded string twice.
2. **Text-only content arrays.** The server (`server/messages.zig`) joins the text parts of a content array with no separator when there is no image, so `["one", "two"]` becomes `onetwo`; the reference gives `one\n\ntwo`. With images the join is a blank line and matches. This is in the server, not the template.

Also by design: `tools` go on the first system message (an empty one is added if there is none), the `latest_reminder` role and the `task` field are in the template but the API cannot reach them, and `drop_thinking` is not exposed.

## Pending

| item | state |
|---|---|
| Recipe gates (`tools/gates.py` of the recipe: drafted == plain, concurrent == solo) | not run yet |
| Second RoCE rail | the second port of two nodes carries link-local addresses while the other two use 192.168.193.x; the plan is runtime aliases `192.168.193.2/24` and `192.168.193.3/24` on `enP2p1s0f0np0` of those two nodes, then `RDMA_HCA=rocep1s0f0,roceP2p1s0f0`; netplan untouched. Not done yet |
| `TF_DS_PROMPT_SP=rows` (reported +12-17% prefill) | not applied; the reference code is not public |
| Serving as the fleet endpoint (0.0.0.0, agent-facing names) | not done |
