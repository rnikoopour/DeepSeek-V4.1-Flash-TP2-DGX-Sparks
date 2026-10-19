# Phase B runbook: TP2 lane on Reddie + Bluey (run only on the go from Tony / main)

All commands run on Reddie (`ssh spark-reddie`) unless marked **Mac**. Everything lives in `/var/tmp/tf1017-tp2/`. Nothing here deletes data; both lanes are containers (`tp4-lane` on four nodes, `tp2-lane` on two).

## 0. Before the window (no downtime)
```
/var/tmp/tf1017-tp2/precheck_tp2.sh     # expected to FAIL only on "tp4-lane" and MemAvailable while TP4 is up; everything else must be OK
```
Both rank caches and the token map must be in place (Phase A step 3). Tell the fleet owner that `:8000` will serve the plain (non-overlay) 2.9 bpw model, 4 streams, for the length of the window.

## 1. Stop TP4, start TP2 (outage starts)
```
/var/tmp/tf1017-tp2/stop_tp4.sh                              # removes tp4-lane on the 4 nodes; prints containers and GPU procs (expect none)
nohup /var/tmp/tf1017-tp2/start_tp2.sh > /var/tmp/tf1017-tp2/logs/start_tp2.out 2>&1 < /dev/null &
tail -f /var/tmp/tf1017-tp2/logs/tp2-lane.log                # on Reddie; rank 1's log is the same path on Bluey
```
`start_tp2.sh` re-adds the rail-2 address, runs the precheck, starts rank 1 then rank 0, waits for `/v1/models`, then a 4-token chat. Expected: load 1 to 2 minutes cold (99 GiB per rank from NVMe), ready in about 2 to 4 minutes. **If rank 0 or rank 1 fails**: copy both logs to `/var/tmp/tf1017-tp2/logs/failed-<time>/`, run `/var/tmp/tf1017-tp2/restore_tp4.sh` (below) and report; do not retry blindly.

## 2. Startup facts for the doc (from the log)
```
grep -E "pool|KV|positions|streams|loaded|available|GiB|gid_index|path_mtu|fallback" /var/tmp/tf1017-tp2/logs/tp2-lane.log | head -60
ssh -n tonyspark1@192.168.192.1 'grep -E "pool|loaded|GiB|gid_index|path_mtu" /var/tmp/tf1017-tp2/logs/tp2-lane.log | head -30'
for h in tonyspark2@192.168.192.2 tonyspark1@192.168.192.1; do ssh -n $h 'hostname; awk "/MemAvailable/ {printf \"%.1f GiB available\n\", \$2/1048576}" /proc/meminfo; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'; done
```
Record: load seconds per rank, the pool size line, MemAvailable per node, both `gid_index=3` and two rails in the ring line.

## 3. Gates (copy the tools' logs into the run folder's logs/)
```
cd /var/tmp/tf1017-tp2/tools
python3 sanity.py http://127.0.0.1:8000                         # chat, alias, tool call + round trip, thinking, image
python3 fleet_check.py http://100.113.138.96:8000               # (from Reddie, tailnet) all PASS expected
python3 concurrent_check.py http://127.0.0.1:8000               # 4 streams: concurrent == solo
python3 sampling_check.py http://127.0.0.1:8000                 # patch behaviour
python3 long_check.py http://127.0.0.1:8000 20000 131073 300000 500000 1000000   # needles; 1M takes about 7 minutes; stop at the first failure and report the largest passing length
```
All of these tools take the base URL as the first argument; check each tool's header if it differs. Also grep the log once for the Triton fallback line (expected for lengths not in the kit's recordings).

## 4. Bench (v1 prompt set, same method as the TP4 run)
```
cd /var/tmp/tf1017-tp2
mkdir -p bench/tp2a
nohup python3 tools/v41bench_tf.py --base http://127.0.0.1:8000/v1 --model deepseek-v4.1-flash --label tp2a --out bench/tp2a \
  --levels 1,2,3,4 --prefill 2000,8000,32000,64000,128000 > bench/tp2a/bench.txt 2>&1 < /dev/null &
```
The lane has 4 streams, so levels above 4 would only queue; the requested `1,2,4,8,12,16,32` is not a valid concurrency sweep on this config. Optional second pass only if step 2 shows at least 12 GiB available on rank 0: stop the lane, `DSV41_CONFIG=/var/tmp/tf1017-tp2/config.tp2-s8.env /var/tmp/tf1017-tp2/serve_tp2.sh start`, rerun with `--levels 1,2,4,6,8 --label tp2b-s8`, and report the memory floor. Copy `bench/tp2a/*` into `results/tp2a/` in the run folder.

## 5. 69-scenario eval (Mac)
```
EP=http://100.113.138.96:8000/v1 ~/.openclaw/workspace/dsv41-flash-vllm-dgx-spark/runs/2026-10-18-1017-tensorfold-tp2/tools/run_eval69_tf_tp2.sh tf1017-tp2 "TP2 2x Spark, LibertAI 2.9 bpw, thinking off"
```
Result files land in `~/.openclaw/workspace/2wild-model-eval-results/dsv41-1017-tensorfold-tp2/`.

## 6. Restore TP4 (outage ends when this answers)
```
/var/tmp/tf1017-tp2/restore_tp4.sh          # stops tp2-lane on both nodes, runs start_fleet_rail2_p4_ablit.sh (the live config), waits for /v1/models
python3 /var/tmp/tf1017-tp2/tools/fleet_check.py http://100.113.138.96:8000
```
Expected TP4 ready 2 to 3 minutes after the script starts (the earlier boots took 107 to 120 s to load). Success = `fleet_check` all PASS and the 4-node line in `docker ps` on each node.

## Downtime (estimates)
- Full outage: TP4 stop to TP2 ready, about 3 to 5 minutes; TP2 stop to TP4 ready, about 3 minutes.
- Degraded window between them (TP2 on `:8000`: plain model, 4 streams): about 2 hours with the 1M needle, bench to 128K and the eval; 1 hour without the 1M and 500K needles.

## Risks
- First GPU run of the new engine build, rank-cache writer and kit on this pack; a load error is the likely failure and costs about 10 minutes before rollback.
- Reddie memory floor of about 5 GiB (head also serves HTTP and logs). A kernel OOM would kill the lane, not the host; restore is the same.
- The fleet gets the non-overlay model during the window.
- `rail2.sh add 2` runs `sudo ip addr` on Reddie (runtime only; same call the TP4 starter makes).
