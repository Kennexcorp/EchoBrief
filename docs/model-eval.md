# Model Evaluation — choosing the default Ollama model

Comparing `llama3.1:8b` and `qwen2.5:7b` on the three bundled sample transcripts
(short single-speaker, medium dialogue, long dialogue — see `scripts/benchmark.py`).
All runs: temperature 0, 8k context, Ollama JSON-schema-constrained decoding,
Apple M5 Pro (18-core, 24 GB). `mistral:7b` was cut from scope mid-eval to save
bandwidth; the harness supports adding it later with two commands.

Reproduce with:

```bash
uv run python scripts/benchmark.py validity --runs 9 --model qwen2.5:7b
uv run python scripts/benchmark.py briefs --model qwen2.5:7b
```

## Criteria

1. **JSON validity** — % of runs producing a schema-valid brief on the first attempt
   (9 runs per model, 3 per transcript; llama additionally scored 20/20 in the main benchmark).
2. **Quote fidelity** — % of `supporting_quote` values found verbatim in the source
   transcript (automated substring check, whitespace-normalized).
3. **Action-item recall** — of 14 ground-truth action items across the medium and long
   transcripts (add confidence intervals, restructure chapter four, book the statistics
   consultant, submit the conference abstract, chapter four due on the 15th; plus rework
   the literature review, send a sample section, prepare the March seminar talk, collect
   training records), how many did the model extract?
4. **Latency** — seconds per brief (short / medium / long transcript).

## Results

| Criterion | llama3.1:8b | qwen2.5:7b |
|---|---|---|
| JSON validity (first attempt) | **100 %** (9/9, and 20/20) | **100 %** (9/9) |
| Quote fidelity (verbatim) | 13/14 (one light paraphrase) | **14/14** |
| Action-item recall | 11/14 (79 %) | **12/14 (86 %)** |
| Latency per brief (s) | **5.2 / 12.3 / 17.5** | 6.5 / 13.3 / 18.7 |

Notes: neither model invented an action item — every extracted task traces to the
transcript (llama's one "extra" on the medium transcript, *email the statistics
consultant today*, is genuinely in the dialogue). Both models most often missed the
same ground-truth item: the chapter-four deadline stated as a bare date ("due on the
fifteenth"), which tends to get folded into the restructuring task instead of
surfacing separately.

## Verdict

**Default stays `llama3.1:8b`; `qwen2.5:7b` is an equally recommended alternative.**
Qwen edges the recall and quote-fidelity scores, but by one item each on a
three-transcript sample — within noise. Both are 100% schema-valid, which is the
property the product actually depends on. Llama keeps the default on latency and
ecosystem familiarity; switching is one dropdown click in the UI or
`OLLAMA_MODEL=qwen2.5:7b` in `.env`, and detail-critical users should try qwen.
