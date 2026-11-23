# bench log acceptance/calls attribution

`tools/moe/bench_attrib.py` turns llama.cpp `slot print_timing` /
`statistics draft-dspark` / `[ Prompt: .. | Generation: .. ]` output into one row
per speculative-decoding session, plus per-arm medians/min/max.  Pure stdlib,
offline: it does not build, touch a phone, or read a model.

The motivating case is T-0032: a single log can contain several runs
(`abba-4c.log` is A B B A), and hand-grepping easily mixes their acceptance /
`#calls` counters.

## Usage

```sh
python3 tools/moe/bench_attrib.py \
  --logs /path/to/runs \
  --out sessions.tsv --json > attrib.json

# explicit arm naming for files like A-1.log / B-2.log (this is the default):
python3 tools/moe/bench_attrib.py --logs 'runs/*.log' \
  --arm-regex '^([AB])(?:[-_.]|$)' --out sessions.tsv --json
```

`--logs` accepts files, directories (recursive `*.log`) and glob patterns.
`--out` writes the fixed-column TSV; `--json` prints `sessions` + `arms` JSON.
Without either option the TSV is printed to stdout.

## Sessions and arms

* A session is one `slot print_timing` block, anchored at its `n_decoded` line
  and ended by the next block, the next `[bench]` header, or the next
  `adding speculative implementation 'draft-dspark'` line.  Counters never cross
  a session boundary.
* Arm order is:
  1. `--arm-regex` match on the file (or parent directory) name;
  2. otherwise `[run-start] <arm>` markers in the same log.  If the number of
     successful marker runs equals the number of timing sessions, arms are
     assigned positionally; if all successful markers agree, that arm is used
     for all sessions;
  3. otherwise `unknown`.
* T-0032's `abba-4a.log` / `abba-4b.log` contain only successful A timing runs
  (their B runs failed at model load), so they become A.  `abba-4c.log` contains
  A B B A and its four timing sessions are paired with the four successful
  `[run-start]` markers.

## TSV fields

| column | source | notes |
|---|---|---|
| `file` | log basename | |
| `session` | 1-based timing block index | |
| `arm` | see above | `unknown` if not inferable |
| `n_decoded` | `n_decoded = N, tg = X t/s` | |
| `tg_tps` | `tg` on the `n_decoded` line | fallback `[Generation: X t/s]` |
| `prompt_tps` | `prompt eval time ... tokens per second` | fallback `[Prompt: X t/s]` |
| `eval_ms` | decode `eval time = X ms` | not `prompt eval time` |
| `graphs_reused` | `graphs reused = N` | |
| `acceptance` | `draft acceptance = R (A accepted / G generated)` | |
| `accepted` / `generated` | same line | |
| `calls_g` / `calls_a` | `#calls(b,g,a) = b g a` | g and a fields |
| `per_call_ms` | `dur_g / calls_g` | see below; empty if `calls_g == 0` |
| `gen_tokens` / `acc_tokens` | `#gen tokens` / `#acc tokens` | |
| `draft_gen` / `draft_acc` | `#gen drafts` / `#acc drafts` | |

Missing or malformed fields are empty in the TSV and listed in the JSON
session's `missing` array; they are **never** filled with 0.

## Two `per_call_ms` definitions (do not mix)

The sample line is:

```text
dur(b,g,a) = 0.001, 589.643, 0.004 ms
```

* **This tool's `per_call_ms` (TSV + JSON)** = `dur_g / calls_g` =
  `589.643 / 34` = 17.3424 ms.  This follows the explicit unit-test definition
  in T-0070 (`per_call_ms=589.643/34`) and measures the generation/draft-call
  duration per call.
* **T-0054's per-verification-call 136.8 ms** = `eval_ms / calls_g` =
  `4623.54 / 34` = 136.0–137.2 ms.  This includes the whole target eval interval
  (draft + verification + any idle gaps within eval), so it is a different
  metric.  It is exposed as `per_call_ms_eval` in JSON for comparison, but it is
  **not** the TSV `per_call_ms` column.

The `a` field of `dur(b,g,a)` (0.004 ms) is the accept-phase total; dividing it
by `calls_a` gives a sub-microsecond number that is not the T-0054 metric.  The
spec's prose says `dur_a / calls_a`, while its concrete unit test says
`589.643/34`; this implementation follows the concrete test and reports the
alternative (`per_call_ms_eval`) separately.

## T-0032 regression (A unweighted Q1_0 vs B weighted Q2_K)

Command (offline, read-only):

```sh
python3 tools/moe/bench_attrib.py \
  --logs /home/yydh/.local/state/dpsk-q2/runs/T-0032/run-20261122T195853-w5-2/artifacts/runs \
  --out "$QUEUE_RUN_DIR/artifacts/t0032-sessions.tsv" --json \
  > "$QUEUE_RUN_DIR/artifacts/t0032-attrib.json"
```

8 sessions: A n=6 (2 each from `abba-4a`, `abba-4b`, `abba-4c`), B n=2
(`abba-4c` only).  Medians [min, max]:

| metric | A | B | B/A |
|---|---:|---:|---:|
| `tg_tps` | 27.425 [27.38, 27.51] | 19.825 [19.75, 19.90] | 0.7229 |
| `eval_tps` | 27.52 [27.45, 27.68] | 19.605 [19.55, 19.66] | **0.7124** |
| `acceptance` | 0.90196 | 0.55634 | 0.6168 |
| `calls_g` (`calls_a`) | 34 | 48 | 1.4118 |
| `graphs_reused` | 33 | 46 | 1.3939 |
| `per_call_ms` (`dur_g/calls_g`) | 17.294 | 17.250 | 0.9975 |
| `per_call_ms_eval` (`eval_ms/calls_g`) | 136.794 | 136.002 | **0.9942** |
| `gen_tokens` / `acc_tokens` | 102 / 92 | 142 / 79 | |

This reproduces T-0054/T-0032: **acceptance 0.902→0.556, calls 34→48, and the
observed t/s ratio 0.7124 (T-0054's `Generation t/s` = eval-time t/s)**.  The
`n_decoded` `tg` gauge gives 0.7229 because it is a slightly different interval;
both are reported, and 0.7229 is still within the task's 0.713±0.02 tolerance.
The `per_call_ms_eval` ratio 0.9942 matches T-0054's 136.8→136.0 ms (−0.6%)
statement.  The 34/48 = 0.7083 call ratio is the causal explanation of the
0.7124 t/s ratio; acceptance, not kernel time, is the driver.

## Tests

```sh
python3 -m unittest discover -s tools/moe/bench -p 'test_bench_attrib.py' -v
```

Covers: full single session, two sessions without cross-talk, missing acceptance,
malformed/empty input, zero `calls_g`, filename and `[run-start]` arm inference,
failed-B marker handling, aggregation, and the fixed TSV header/empty cells.
