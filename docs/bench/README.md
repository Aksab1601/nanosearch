# Benchmark artifacts

Committed on purpose: the numbers in the project README should be checkable
rather than taken on trust.

| File | What it is |
|---|---|
| `sweep_n1000000_d128_m32_final.json` | The run the README quotes: 1M × 128-d, `nlist=1024`, `m=32`. Machine-readable rows plus the environment the numbers were produced on. |
| `sweep_n1000000_d128_m32_final.md` | The same run as a table. |
| `recall_vs_qps[_dark].png` | Recall against throughput, drawn from that JSON by `benchmarks/plot.py`. |
| `memory[_dark].png` | Resident memory per backend. |
| `raw/1m_m16.log` | The same sweep at `m=16` (64 bytes → 16 bytes per vector). Kept because it is the evidence that `m`, not `nprobe`, sets the recall ceiling: strict recall@10 falls from 0.52 to 0.25 while shortlist recall holds at 0.77. |
| `raw/1m_m32_nprobe16.log` | An earlier `m=32` run that re-ranks at `nprobe=16` instead of 8. Slightly higher recall, roughly 1.6× the latency. |

Reproduce the headline run with `make bench-big` (several minutes), then redraw
the charts with:

```bash
python benchmarks/plot.py docs/bench/sweep_n1000000_d128_m32_final.json
```

Hardware matters for the latency columns and not at all for the recall ones:
recall is a property of the index and the data, so it should reproduce
anywhere. The run here was a 2-core container.
