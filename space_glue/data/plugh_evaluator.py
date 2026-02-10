# Adapted from https://github.com/altsoph/PLUGH/blob/main/calc_metrics.py
import re
from Levenshtein import distance as ldist


def normalized_distance(p1, p2):
    return ldist(p1, p2) / max(len(p1), len(p2))


def normalize(n):
    n = n.lower()
    n = n.replace("_", " ").replace("'", " ")
    n = f" {n} ".replace(" the ", " ").replace(" a ", " ")
    n = n.strip().strip(";").strip('"').strip()
    n = re.sub(r"\[[^\]]+\]", " ", n).strip()
    return n


def extract_graph(raw):
    if raw is None:
        return [], []
    if "assistant\n\n" in raw:
        raw = raw.split("assistant\n\n", 1)[0]
    if "{" in raw:
        _, d = raw.split("{", 1)
    else:
        d = raw
    if "}" in d:
        d = d.split("}")[-2]
    edges = []
    nodes = set()
    for l in d.split("\n"):
        l = l.replace(" -- ", " -> ")
        if " -> " in l:
            n1, n2 = l.split(" -> ", 1)
            n1 = normalize(n1)
            n2 = normalize(n2)
            if (n1, n2) not in edges and (n2, n1) not in edges:
                edges.append((n1, n2))
                nodes.add(n1)
                nodes.add(n2)
    return list(sorted(edges))


def match_nodes(n1, n2):
    if (
        n1
        and n2
        and (
            not (set(n1.split()) - set(n2.split()))
            or not (set(n2.split()) - set(n1.split()))
        )
    ):
        return True
    return False


def fuzzy_intersect_nodes(gt_n, pr_n):
    intersection = 0
    seen = set()
    for n1 in gt_n:
        for n2 in pr_n:
            if n2 not in seen and match_nodes(n1, n2):
                intersection += 1
                seen.add(n2)
                break
    return min(intersection, len(gt_n), len(pr_n))


def fuzzy_intersect_edges(gt_e, pr_e):
    intersection = 0
    seen = set()
    for e1 in gt_e:
        for e2 in pr_e:
            if e2 not in seen and fuzzy_intersect_nodes(e1, e2) == 2:
                intersection += 1
                seen.add(e2)
                break
    return min(intersection, len(gt_e), len(pr_e))


def task1(ground_truth, predicted):
    # Nodes
    intersection = 0
    pred_len = 0
    gt_len = 0
    gt_n = set(
        [normalize(n) for n, _ in ground_truth]
        + [normalize(n) for _, n in ground_truth]
    )
    pr_n = set(
        [normalize(n) for n, _ in predicted] + [normalize(n) for _, n in predicted]
    )
    pred_len = len(pr_n)
    gt_len = len(gt_n)
    intersection = fuzzy_intersect_nodes(gt_n, pr_n)
    if not pred_len:
        f1_nodes = 0.0
    prec_nodes = intersection / pred_len if pred_len else 0.0
    rec_nodes = intersection / gt_len if gt_len else 0.0
    if prec_nodes + rec_nodes:
        f1_nodes = 2 * prec_nodes * rec_nodes / (prec_nodes + rec_nodes)
    else:
        f1_nodes = 0.0

    # Edges
    intersection = 0
    pred_len = 0
    gt_len = 0
    gt_e = set([frozenset([normalize(n) for n in e]) for e in ground_truth])
    pr_e = set([frozenset([normalize(n) for n in e]) for e in predicted])
    pred_len = len(pr_e)
    gt_len = len(gt_e)
    intersection = fuzzy_intersect_edges(gt_e, pr_e)

    if not pred_len:
        f1_edges = 0.0
    prec_edges = intersection / pred_len if pred_len else 0.0
    rec_edges = intersection / gt_len if gt_len else 0.0
    if prec_edges + rec_edges:
        f1_edges = 2 * prec_edges * rec_edges / (prec_edges + rec_edges)
    else:
        f1_edges = 0.0

    return f1_nodes, f1_edges, prec_nodes, rec_nodes, prec_edges, rec_edges


def task2(ground_truth, predicted):
    if predicted is None:
        return 1.0
    return normalized_distance(ground_truth, predicted.split("\n"))


# Same evaluation for task 3 and 4
def task3_4(ground_truth, predicted):
    return min(map(lambda x: task2(x, predicted), ground_truth))
