# Design notes

How nanosearch works, why each piece is shaped the way it is, and what it does
not do. The README has the results; this is the reasoning.

---

## 1. The problem

Given a corpus of `n` vectors in `d` dimensions and a query vector `q`, return
the `k` vectors closest to `q`.

Exact search is three lines of NumPy and costs `O(n · d)` per query. At one
million 128-dimensional vectors that is 128 million multiply-adds, which
measures at **55 ms** and needs the whole 512 MB corpus resident. Both numbers
are problems: 55 ms is most of a latency budget before any of the real work
(re-ranking, generation) starts, and 512 MB per million vectors means a
100-million-vector corpus is a cluster rather than a container.

So there are two separate costs to attack, and they need different ideas:

| Cost | Idea | Mechanism |
|---|---|---|
| Too many vectors scored | **IVF** | partition the space, score only nearby cells |
| Each vector too big and too slow to score | **PQ** | store `m` bytes per vector, score by table lookup |

They compose, and the composition is the interesting part.

---

## 2. One distance function

Everything reduces to squared Euclidean distance, expanded so the work lands in
one matrix multiply:

```
‖q − x‖² = ‖q‖² − 2·q·x + ‖x‖²
```

`‖x‖²` is cached at insert time, because the database does not change between
queries — only the queries do. The naive
`((q[:, None] - x[None]) ** 2).sum(-1)` is mathematically identical and
allocates an `(nq, n, d)` intermediate; the expansion hands BLAS a single
`(nq, d) × (d, n)` job.

**Cosine similarity needs no second code path.** For L2-normalised vectors,

```
‖q − x‖² = 2 − 2·cos(q, x)
```

so squared distance is a strictly decreasing function of cosine similarity and
ranking by either is the same ranking. nanosearch normalises on the way in when
`metric="cosine"` and converts back to a similarity only when results are
returned. One code path, two metrics, and no opportunity for them to disagree.

The other primitive is top-k selection. `np.argpartition` is `O(n)`
(quickselect); only the `k` selected entries are then sorted, for
`O(n + k log k)`. A full sort would be `O(n log n)` — strictly more work than
the question asks for.

---

## 3. IVF: score fewer vectors

**Build.** Run k-means on a sample of the corpus to get `nlist` centroids. Each
vector is assigned to its nearest centroid, and each cell keeps a list of the
vectors it owns — an *inverted list*, by analogy with the term→documents lists
of text search.

**Query.** Compute distances to the `nlist` centroids, take the `nprobe`
nearest, and score only the vectors in those cells:

```
O(nlist · d)                    find the nearest centroids
  + O(n · nprobe/nlist · d)     score the candidates
```

The two terms balance around `nlist ≈ √n`, which is where the usual
"`nlist` between `√n` and `4√n`" advice comes from. Too few cells and each one
is enormous; too many and finding them costs more than searching them.

**The approximation is exactly one thing:** a true neighbour sitting in a cell
that was not probed can never be returned. That is the whole error model, and
it has three consequences worth knowing:

1. `nprobe = nlist` reduces to exact search. The test suite asserts this
   (`test_ivf.py::test_probing_every_cell_equals_exact_search`) — if it ever
   fails, vectors are being lost in the assignment, the lists, or the merge,
   and no amount of tuning would fix that.
2. `nprobe` is a **runtime** dial on an already-built index. The same index can
   serve a cheap autocomplete path and an expensive high-recall path.
3. Recall depends on how well the partition matches the data. Clustered data
   does well; uniform noise has no cluster structure to exploit.

**Why list balance matters.** Query cost is driven by the size of the lists a
query actually probes, not by the average. One cell holding 20% of the corpus
turns every query that lands near it into a near-exhaustive scan, which shows
up as a p95 far above the p50 rather than as a bad mean. `stats()` reports the
imbalance factor `n · Σsᵢ² / (Σsᵢ)²` — 1.0 is perfectly even, and the measured
value here is 1.46.

---

## 4. PQ: make each vector smaller

Split a `d`-dimensional vector into `m` contiguous sub-vectors and run k-means
with 256 centroids *independently* on each slice. A vector becomes `m` bytes —
one codebook index per slice.

At `d = 128, m = 32`: 128 bytes becomes 32, and the codebooks jointly describe
`256³² ≈ 10⁷⁷` distinct points. The representable set is enormous even though
each slice is coarse — that is the trick. The assumption being made is that the
slices carry roughly independent information, which is not quite true of a
learned embedding; OPQ (a learned rotation applied before splitting) is the
standard fix and is not implemented here.

### Asymmetric Distance Computation

The naive way to use compressed codes would be to decompress each candidate.
ADC avoids that. For a query `q`, precompute a table of distances from each of
its sub-vectors to each of the 256 centroids in that subspace — `m × 256`
numbers — then for any stored vector:

```
d(q, x) ≈ Σⱼ table[j][ code_x[j] ]
```

Scoring a candidate is `m` byte-indexed lookups and `m` adds instead of `d`
multiply-adds. The table costs `256 · d` flops per query, paid once and
amortised over every candidate scored — **which is why PQ wins on long
candidate lists and loses on short ones.**

"Asymmetric" means the query stays in full precision while only the database
side is quantised; quantising both would be strictly less accurate for no gain.

### Residual encoding

PQ is applied to `x − centroid(x)`, not to `x`. Residuals are smaller and more
tightly concentrated than raw vectors, so the same 256-entry codebooks resolve
them far more finely — the coarse quantiser has already said *where* the vector
is, leaving PQ to describe only the local offset. Free accuracy at identical
storage cost.

It works because the residual is taken against the same centroid on both sides:

```
‖q − x‖² = ‖(q − c) − (x − c)‖²
```

A query is compared with a cell's codes by first forming its own residual
against that cell's centroid. Distances from different cells stay comparable,
because each approximates the same `‖q − x‖²`.

---

## 5. What PQ actually costs — and the two-stage fix

The measured behaviour of IVF-PQ at 1M × 128, `m = 32`:

| metric | value |
|---|---|
| recall@10 (strict) | **0.52** |
| recall@10 inside a 100-item shortlist | **0.97** |

That gap *is* product quantisation. The codes locate the right neighbourhood
reliably and cannot order it finely, because 10% distortion is enormous
relative to the distance gaps between the 10th and the 100th nearest neighbour
in a million vectors. Raising `nprobe` does not help: the ceiling is set by `m`,
not by how hard you search. Recall is flat from `nprobe = 4` onward.

The conclusion is not that PQ is bad — it is that **a compressed index is a
candidate generator, not a final ranking.** So `RefineIndex` asks it for
`k × factor` candidates and re-scores just those few hundred against
full-precision vectors:

| configuration | recall@10 | p50 | RAM |
|---|---|---|---|
| IVF-PQ alone, `nprobe=8` | 0.519 | 3.14 ms | 32.7 MB |
| \+ exact re-rank, `factor=32` | **0.9975** | 3.38 ms | 32.7 MB |

Where the full vectors live is the design decision. Keeping them in RAM would
undo the compression, so they sit in a flat float32 file read through
`np.memmap`: the OS pages in only the rows touched — roughly `k × factor`
random reads of `4d` bytes per query. RAM holds the `m`-byte codes; the cold
bytes stay on disk. FAISS's `IndexRefineFlat` and every serious vector database
implement the same tiering, and it is why "approximate index" rarely means
"approximate answers" in production.

Two honest costs: disk reads add latency (fine on SSD, bad on network storage),
and total footprint is unchanged — what changes is how much must be in *memory*.

---

## 6. Data structures

| Structure | Where | Why |
|---|---|---|
| Geometrically grown array | `FlatIndex`, `IVFIndex`, `IVFPQIndex` | doubling keeps appends amortised `O(1)`; growing by a constant would make repeated `add` calls `O(n²)` overall |
| Inverted lists → CSR on save | `inverted.py` | per-list arrays for cheap appends while building; one flat array plus `nlist+1` offsets when serialised — compact, and how production indexes lay it out in memory |
| `(m, 256, d/m)` codebooks + transposes | `pq.py` | the transposed copy and `‖centroid‖²` are needed on every query and never change, so they are computed once at fit time |
| Hand-rolled CSR document-term matrix | `text.py` | a document-term matrix is >99% zeros; CSR turns `n_docs × vocab` into `nnz` |
| Bounded ring buffer | `service/app.py` | p50/p95/p99 over recent requests at fixed memory cost |

---

## 7. Two optimisations found by profiling

Both came from measuring, not from guessing, and both are the same mistake:
**many small BLAS calls instead of one batched call.**

**1. Distance tables.** The obvious implementation loops over the `m` subspaces,
each doing its own `(1, d/m) × (d/m, 256)` distance computation. At `m = 16`
that is 16 BLAS calls on tiny matrices, which spend all their time in call
overhead. Expressing it as one batched matmul over the subspace axis:

| | per call (16 residual queries) |
|---|---|
| loop over subspaces | 338 µs |
| one batched matmul | **111 µs** (3.0× faster) |

End-to-end this cut IVF-PQ p50 at 50k from 0.49 ms to 0.29 ms. The readable
version is kept as `reference_distance_tables()` and a test asserts the fast
one agrees with it — the batched version is a transpose-and-reshape puzzle,
exactly the kind of code that is wrong in a way output-level tests would not
localise.

**2. Grouped ADC.** Residual encoding means a candidate's lookup table depends
on which cell it lives in, so the natural implementation loops over probed
cells — making per-query interpreter cost scale with `nprobe` and wiping out
the algorithmic win. `asymmetric_distances_grouped()` instead takes a `groups`
array naming the table for each candidate and scores the whole probed set in
`m` vectorised gathers: **O(m) interpreter work per query regardless of
`nprobe`.**

---

## 8. Choosing parameters

| Parameter | Effect | Guidance |
|---|---|---|
| `nlist` | cells; build cost and cell size | `√n` to `4√n`; keep ≥ ~39 vectors per cell |
| `nprobe` | **runtime** recall/latency dial | start at `nlist/64`, raise until recall is enough |
| `m` | bytes per vector; recall ceiling | must divide `d`; `d/4` is a good start, `d/m ≥ 4` |
| `factor` | re-rank shortlist depth | 8–32; cheap recall once the shortlist contains the answer |

And the first question, which is not about parameters at all:

| Corpus | Use |
|---|---|
| < 10k | `FlatIndex`. Exact search is sub-millisecond and always right; an approximate index here buys nothing and costs recall. |
| 10k–50k | `IVFIndex`. The partition saves real time and nothing about the vectors is approximated. |
| > 50k | `IVFPQIndex`, with `RefineIndex` if strict top-k ordering matters. Also the point where there are finally enough vectors to fit 256-entry codebooks per subspace. |

`engine.choose_index()` implements this.

---

## 9. The text layer

The index does not care where vectors come from, but a search engine you cannot
type a sentence into is hard to believe in. `text.py` implements LSA —
TF-IDF then truncated SVD — on NumPy, so the demo and the service work offline
with no model download.

Three things in it were decided by measurement
(`benchmarks/bench_text_quality.py`, 16 labelled queries over 8 topics,
precision@3 against a 0.125 random baseline):

1. **Document-frequency filtering is not optional.** Without `max_df`, the top
   singular directions are dominated by "the", "a" and "and" — a word in every
   document describes the average document rather than distinguishing any of
   them. IDF downweights such terms but cannot remove them, and after row
   normalisation their small weights still sum to a direction the SVD
   faithfully learns. Filtering by document frequency is a corpus-driven
   stopword list that needs no word list. On this corpus "the" appears in 75 of
   96 documents, so a lax `max_df=0.4` dropped only three terms and changed
   nothing; `max_df=0.2` is the default.
2. **The reduction has to actually reduce.** 94 dimensions over 96 documents is
   a rotation of TF-IDF space, not a compression, and scored *worse* than 48.
   `choose_embedding_dim()` targets roughly half the corpus size.
3. **Bigrams hurt on a small corpus.** Almost every bigram occurs once, adding
   thousands of features that cannot support cross-document similarity.
   Unigrams are the default; turn bigrams on when the corpus is large enough
   for them to repeat.

Measured: **precision@3 = 0.63, MRR = 0.79** — about 5× the random baseline.
For real semantic quality, `SentenceTransformerEmbedder` swaps a transformer in
behind the same two-method interface, and nothing else changes. That
substitutability is the architectural point: the hard parts of this project are
independent of where vectors come from, which is why the index benchmarks can
run on synthetic vectors and still mean something.

---

## 10. Testing strategy

191 tests. The ones that matter are not the shape-and-dtype checks:

- **Exact reduction.** `nprobe = nlist` must equal brute force *exactly*.
- **Fast path vs reference path.** The batched table builder is asserted against
  the readable loop; grouped ADC against per-group scoring.
- **Monotonicity.** Recall must rise with `nprobe`, with re-rank depth, and with
  `m`. These catch a whole class of subtle breakage that a single-point recall
  assertion would not.
- **Approximation quality, not just approximation.** ADC distances are asserted
  to track true distances, and the true top-5 to land in a 50-item shortlist.
- **Aggregate quality metrics, not single queries.** Text retrieval is asserted
  on precision@3 and MRR over a labelled set. One query's ranking on a
  96-document LSA model is noisy enough that asserting on it would be a flaky
  test dressed up as a quality bar.
- **Round-trips are bit-identical,** and `allow_pickle=False` holds, so loading
  an index someone else produced cannot execute their code.
- **Documented limits are asserted.** Out-of-vocabulary input must return a
  well-formed response rather than crash the service.

---

## 11. What this does not do

Stated plainly, because the gaps are as informative as the features:

- **No HNSW.** Graph indexes dominate the recall/latency frontier in-memory and
  are the obvious next build. IVF-PQ wins on memory and on build time.
- **No OPQ.** A learned rotation before splitting would reduce the inter-subspace
  correlation PQ assumes away, and is the cheapest available recall win.
- **No SIMD.** Real ADC implementations use 4-bit codes and SIMD shuffles to keep
  lookup tables in registers. NumPy gathers are the floor here, and it is why
  IVF-PQ is not faster than IVF at these scales — the compression is real, the
  per-candidate speedup mostly is not.
- **No deletion or in-place update.** Both need tombstones plus periodic
  compaction; IVF cell membership is part of how a vector is stored.
- **Single process, no concurrency control.** `add` and `search` are not safe to
  call concurrently. A real service would snapshot or use a read-write lock.
- **No metadata filtering.** "Nearest neighbours *where* topic = X" interacts
  badly with partitioning — a selective filter can empty the probed cells — and
  is an active research area.
- **Build time is not optimised.** Coarse k-means dominates it (20 s of the
  ~75 s build at 1M); subsampling bounds it, nothing parallelises it.

---

## 12. References

- Jégou, Douze & Schmid (2011), *Product Quantization for Nearest Neighbor
  Search* — PQ and IVFADC.
- Ge, He, Ke & Sun (2013), *Optimized Product Quantization* — OPQ.
- Malkov & Yashunin (2018), *Efficient and robust approximate nearest neighbor
  search using HNSW graphs*.
- Halko, Martinsson & Tropp (2011), *Finding structure with randomness* —
  randomized SVD.
- Arthur & Vassilvitskii (2007), *k-means++: The Advantages of Careful Seeding*.
- Johnson, Douze & Jégou (2019), *Billion-scale similarity search with GPUs* —
  the FAISS paper.
