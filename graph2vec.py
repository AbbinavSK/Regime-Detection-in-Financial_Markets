import zlib
import numpy as np
import networkx as nx
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# Weisfeiler-Lehman relabeling: hash each node's label with its sorted neighbours' labels, wl_iterations times, pooling every round's labels into one feature bag.
def _wl_graph_features(G: nx.Graph, wl_iterations: int) -> list[str]:
    labels = {}
    for node in G.nodes():
        if "feature" in G.nodes[node]:
            labels[node] = str(G.nodes[node]["feature"])
        else:
            labels[node] = str(G.degree(node))

    all_features = list(labels.values())
    for _ in range(wl_iterations):
        new_labels = {}
        for node in G.nodes():
            neighbour_labels = sorted([labels[n] for n in G.neighbors(node)])
            raw = labels[node] + "_" + "_".join(neighbour_labels)
            new_labels[node] = str(zlib.crc32(raw.encode()))
        labels = new_labels
        all_features += list(labels.values())

    return all_features


# Self-contained Graph2Vec: WL graph features fed into a gensim Doc2Vec (PV-DBOW) model, one document per graph.
class Graph2Vec:
    def __init__(
        self,
        wl_iterations : int   = 2,
        dimensions    : int   = 128,
        epochs        : int   = 10,
        learning_rate : float = 0.025,
        min_count     : int   = 5,
        down_sampling : float = 0.0001,
        workers       : int   = 4,
        seed          : int   = 42,
    ):
        self.wl_iterations = wl_iterations
        self.dimensions    = dimensions
        self.epochs        = epochs
        self.learning_rate = learning_rate
        self.min_count     = min_count
        self.down_sampling = down_sampling
        self.workers       = workers
        self.seed          = seed
        self.model         = None

    def _to_documents(self, graphs):
        return [TaggedDocument(words=_wl_graph_features(G, self.wl_iterations), tags=[str(i)]) for i, G in enumerate(graphs)]

    def fit(self, graphs):
        documents = self._to_documents(graphs)
        self.model = Doc2Vec(
            documents,
            vector_size = self.dimensions,
            window      = 0,
            min_count   = self.min_count,
            dm          = 0,
            sample      = self.down_sampling,
            workers     = self.workers,
            epochs      = self.epochs,
            alpha       = self.learning_rate,
            seed        = self.seed,
        )
        self._embeddings = np.array([self.model.dv[str(i)] for i in range(len(documents))])

    def get_embedding(self) -> np.ndarray:
        return self._embeddings

    def infer(self, graphs) -> np.ndarray:
        assert self.model is not None, "Call fit() before infer()."
        features = [_wl_graph_features(G, self.wl_iterations) for G in graphs]
        return np.array([self.model.infer_vector(f, alpha=self.learning_rate, min_alpha=1e-5, epochs=self.epochs) for f in features])

    def oov_rate(self, graphs) -> list[float]:
        assert self.model is not None, "Call fit() before oov_rate()."
        vocab = self.model.wv.key_to_index
        rates = []
        for G in graphs:
            features = _wl_graph_features(G, self.wl_iterations)
            oov = sum(1 for f in features if f not in vocab)
            rates.append(oov / len(features) if features else 0.0)
        return rates