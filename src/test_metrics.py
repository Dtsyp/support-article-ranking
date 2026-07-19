"""Проверка метрики на примерах, посчитанных вручную."""

import math
from metrics import ap_at_k, map_at_k


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-12)


def run() -> None:
    # 1. Идеальное ранжирование: один релевантный документ на 1-м месте
    assert close(ap_at_k([1, 2, 3], {1}), 1.0)

    # 2. Единственный релевантный на 2-м месте: Precision@2 = 1/2
    assert close(ap_at_k([9, 1, 3], {1}), 0.5)

    # 3. Два релевантных на местах 1 и 2: (1/2) * (1/1 + 2/2) = 1.0
    assert close(ap_at_k([1, 2, 9], {1, 2}), 1.0)

    # 4. Два релевантных на местах 1 и 3: (1/2) * (1/1 + 2/3) = 5/6
    assert close(ap_at_k([1, 9, 2], {1, 2}), 5 / 6)

    # 5. Релевантный за пределами топ-10 не учитывается
    assert close(ap_at_k(list(range(100, 110)) + [1], {1}), 0.0)

    # 6. Ни одного найденного: AP = 0
    assert close(ap_at_k([5, 6], {1}), 0.0)

    # 7. Хвост из нерелевантных не портит уже набранное:
    #    релевантный на 1-м месте, дальше мусор
    assert close(ap_at_k([1] + list(range(100, 109)), {1}), 1.0)

    # 8. |R_q| > 10: нормировка на min(|R_q|, 10) = 10
    rel = set(range(1, 16))  # 15 релевантных
    pred = list(range(1, 11))  # нашли 10 из них, все сверху

    assert close(ap_at_k(pred, rel), 1.0)

    # 9. MAP — среднее по запросам: (1.0 + 0.5) / 2
    preds = {1: [1], 2: [9, 2]}
    gt = {1: {1}, 2: {2}}

    assert close(map_at_k(preds, gt), 0.75)

    # 10. Запрос без предсказаний в MAP даёт 0: (1.0 + 0) / 2
    assert close(map_at_k({1: [1]}, {1: {1}, 2: {2}}), 0.5)

    print("metrics: all 10 checks passed")

if __name__ == "__main__":
    run()
