"""nanosearch -- a vector search engine built from scratch.

Three index backends over one interface:

======================  ==========  ====================  =============================
Index                   Recall      Bytes per vector      Use it when
======================  ==========  ====================  =============================
``FlatIndex``           1.0         ``4 * dim``           under ~50k vectors, or as
                                                          the ground truth for recall
``IVFIndex``            tunable     ``4 * dim``           speed matters, memory does not
``IVFPQIndex``          tunable     ``m``                 both matter (the usual case)
======================  ==========  ====================  =============================

Quick start::

    import numpy as np
    from nanosearch import IVFPQIndex

    vectors = np.random.default_rng(0).normal(size=(50_000, 128)).astype("float32")

    index = IVFPQIndex(dim=128, nlist=256, m=16, metric="cosine")
    index.train(vectors)          # learn the partition + codebooks
    index.add(vectors)            # 16 bytes per vector instead of 512
    ids, scores = index.search(vectors[:5], k=10, nprobe=16)
"""

from __future__ import annotations

from .base import BaseIndex, SearchResult
from .evaluate import LatencyReport, measure_latency, recall_at_k
from .flat import FlatIndex
from .inverted import InvertedLists
from .ivf import IVFIndex
from .ivfpq import IVFPQIndex
from .kmeans import KMeans
from .pq import ProductQuantizer
from .refine import RefineIndex

__version__ = "0.1.0"

__all__ = [
    "BaseIndex",
    "FlatIndex",
    "IVFIndex",
    "IVFPQIndex",
    "InvertedLists",
    "KMeans",
    "LatencyReport",
    "ProductQuantizer",
    "RefineIndex",
    "SearchResult",
    "__version__",
    "load_index",
    "measure_latency",
    "recall_at_k",
]


def load_index(path: str) -> BaseIndex:
    """Load any saved index, dispatching on the class recorded in the file."""
    return BaseIndex.load(path)
