"""Метрика качества ранжирования MAP@10

Формулу беру из задания. Для запроса q с множеством правильных статей R_q
и ранжированным списком предсказаний p_1, ..., p_k (k <= 10):

    AP@10 = (1 / min(|R_q|, 10)) * sum_i [p_i in R_q] * Precision@i

где Precision@i — доля релевантных среди первых i предсказаний
MAP@10 — среднее AP@10 по всем запросам
"""

from __future__ import annotations
from collections.abc import Iterable, Sequence

def ap_at_k(
        predicted: Iterable[int],
        relevant: Sequence[int],
        k: int = 10,
) -> float:
    """Average Precision@k для одного запроса

    predicted — ранжированный список article_id (важен порядок)
    relevant  — множество правильных article_id
    """
    relevant = set(relevant)
    if not relevant:
        raise ValueError("relevant set is empty")

    predicted = list(predicted)[:k]
    hits = 0
    score = 0.0
    for i, p in enumerate(predicted, start=1):
        if p in relevant:
            hits += 1
            score += hits / i  # Precision@i в момент попадания
    return score / min(len(relevant), k)


def map_at_k(
        predictions: dict[int, Sequence[int]],
        ground_truth: dict[int, Iterable[int]],
        k: int = 10,
) -> float:
    """MAP@k: среднее AP@k по всем запросам из ground_truth

    predictions  — {query_id: [article_id, ...]} (ранжированные)
    ground_truth — {query_id: {article_id, ...}}
    Запрос без предсказаний получает AP = 0
    """
    if not ground_truth:
        raise ValueError("ground_truth is empty")
    total = 0.0
    for qid, rel in ground_truth.items():
        total += ap_at_k(predictions.get(qid, []), rel, k=k)
    return total / len(ground_truth)