import numpy as np

from .ontology import Ontology


NAMESPACES = {"bp": "biological_process", "mf": "molecular_function", "cc": "cellular_component"}
ROOTS = {"bp": "GO:0008150", "mf": "GO:0003674", "cc": "GO:0005575"}


class Evaluator:
    def __init__(self, go_file, ontology, terms):
        self.go = Ontology(str(go_file), with_rels=True)
        labels = sorted(self.go.get_namespace_terms(NAMESPACES[ontology]) - {ROOTS[ontology]})
        index = {term: i for i, term in enumerate(labels)}
        self.projection = [[index[parent] for parent in self.go.get_anchestors(term) if parent in index] for term in terms]
        self.label_count = len(labels)

    def __call__(self, labels, scores, protocol="original"):
        truth = np.zeros((len(labels), self.label_count), dtype=bool)
        predictions = np.full((len(labels), self.label_count), -1.0, dtype=np.float64 if protocol == "original" else np.float32)
        for column, ancestors in enumerate(self.projection):
            for ancestor in ancestors:
                truth[:, ancestor] |= labels[:, column] > 0
                predictions[:, ancestor] = np.maximum(predictions[:, ancestor], scores[:, column])
        if protocol not in {"original", "fast"}:
            raise ValueError(f"Unknown evaluation protocol: {protocol}")
        true_count = truth.sum(axis=1)
        eligible = true_count > 0
        if not eligible.any():
            raise ValueError("No annotated proteins remain after GO namespace propagation")
        precision_values = []
        recall_values = []
        best_f, best_threshold = 0.0, 0.0
        for threshold in np.arange(101) / 100.0:
            predicted = predictions >= threshold
            count = predicted.sum(axis=1)
            tp = (predicted & truth).sum(axis=1)
            precision_mask = count > 0
            if protocol == "original":
                precision_mask &= eligible
            precision = float(np.mean(tp[precision_mask] / count[precision_mask])) if precision_mask.any() else 0.0
            recall = float(np.mean(tp[eligible] / true_count[eligible]))
            precision_values.append(precision)
            recall_values.append(recall)
            fscore = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            if fscore > best_f:
                best_f, best_threshold = fscore, float(threshold)
        order = np.argsort(recall_values)
        integrate = getattr(np, "trapezoid", np.trapz)
        area = float(integrate(np.asarray(precision_values)[order], np.asarray(recall_values)[order]))
        return {"fmax": best_f, "aupr": area, "threshold": best_threshold, "protocol": protocol}
