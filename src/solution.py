"""Полный пайплайн решения: валидация на calibration + генерация answer.csv

Запуск:
    python solution.py # офлайн: BM25 + LSA
    python solution.py --use-emb # + нейроэмбеддинги e5

Что делает:
  1. Чистит HTML статей, строит сигналы поиска (BM25, LSA, эмбеддинги)
  2. Подбирает веса объединения RRF по 5-fold кросс-валидации на calibration
     (честно: веса выбираются на train-фолдах, оцениваются на held-out)
  3. Печатает MAP@10 каждого сигнала и итогового объединения
  4. Прогоняет лучшую конфигурацию на test.f и пишет answer.csv
  5. Строго проверяет формат answer.csv

Детерминировано (фикс SEED) — повторный запуск даёт тот же answer.csv
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from metrics import map_at_k  # noqa: E402
from retrieval import (  # noqa: E402
    SEED,
    BM25Signal,
    EmbeddingSignal,
    LSASignal,
    fuse,
    lexical_documents,
    rrf_contributions,
    topk_ids,
)

TOP_K = 10
RRF_K = 10
N_FOLDS = 5

_DATA_SUBDIRS = ("data", "candidate_data", "candidate_public/candidate_data")


def find_data_dir(explicit: str | None) -> Path:
    """Находит папку с articles.f. Приоритет: --data, затем автопоиск"""
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if (p / "articles.f").exists():
            return p
        raise FileNotFoundError(f"В {p} нет articles.f")
    here = Path(__file__).resolve().parent
    bases = [here, here.parent, here.parent.parent, Path.cwd()]
    for base in bases:
        for sub in _DATA_SUBDIRS:
            cand = base / sub
            if (cand / "articles.f").exists():
                return cand
    raise FileNotFoundError(
        "Не нашёл articles.f. Укажи папку с данными явно: "
        "python solution.py --data ПУТЬ_К_ПАПКЕ_С_articles.f"
    )


def map_for_weights(contribs, weights, article_ids, qids, gt, idx):
    """MAP@10 для подмножества запросов idx при заданных весах"""
    sub = {k: v[idx] for k, v in contribs.items()}
    fused = fuse(sub, weights)
    preds = dict(zip([qids[i] for i in idx], topk_ids(fused, article_ids, TOP_K)))
    sub_gt = {qids[i]: gt[qids[i]] for i in idx}
    return map_at_k(preds, sub_gt)


def weight_grid(signal_names):
    """Сетка весов RRF по доступным сигналам. BM25 — опорный (вес 1.0),
    остальные подбираем относительно него"""
    axes = []
    for name in signal_names:
        if name == "bm25":
            axes.append([1.0])
        else:
            axes.append([0.4, 0.6, 0.8, 1.0, 1.2])
    for combo in itertools.product(*axes):
        yield dict(zip(signal_names, combo))


def cross_validate(contribs, article_ids, qids, gt):
    """5-fold CV: на каждом фолде выбираем лучшие веса по train, меряем на test
    Возвращает (средний held-out MAP, std, финальные веса на всех данных)"""
    names = list(contribs)
    n = len(qids)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    held_out = []
    for tr, te in kf.split(np.arange(n)):
        best_w, best_m = None, -1.0
        for w in weight_grid(names):
            m = map_for_weights(contribs, w, article_ids, qids, gt, tr)
            if m > best_m:
                best_m, best_w = m, w
        held_out.append(map_for_weights(contribs, best_w, article_ids, qids, gt, te))

    # финальные веса — лучшие на всей калибровке
    best_w, best_m = None, -1.0
    all_idx = np.arange(n)
    for w in weight_grid(names):
        m = map_for_weights(contribs, w, article_ids, qids, gt, all_idx)
        if m > best_m:
            best_m, best_w = m, w
    return float(np.mean(held_out)), float(np.std(held_out)), best_w, best_m


def build_signals(articles, use_emb):
    lex_docs = lexical_documents(articles)
    signals = {"bm25": BM25Signal(lex_docs), "lsa": LSASignal(lex_docs)}
    if use_emb:
        signals["emb"] = EmbeddingSignal(articles)  # чанкование внутри
    return signals


def validate_answer(answer: pd.DataFrame, test: pd.DataFrame, valid_ids: set[int]) -> None:
    """Строгая проверка формата answer.csv (требования задания)."""
    assert list(answer.columns) == ["query_id", "answer"], "колонки должны быть query_id, answer"
    assert set(answer.query_id) == set(test.query_id), "должны быть ВСЕ query_id из test, без лишних"
    assert len(answer) == len(test), "число строк != числу запросов (дубли/пропуски)"
    for _, row in answer.iterrows():
        ids = row.answer.split()
        assert len(ids) <= TOP_K, f"больше {TOP_K} статей у query_id={row.query_id}"
        assert len(ids) == len(set(ids)), f"повтор article_id у query_id={row.query_id}"
        for a in ids:
            assert int(a) in valid_ids, f"article_id {a} нет в articles.f"
    print("формат answer.csv: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-emb", action="store_true", help="добавить эмбеддинги e5")
    ap.add_argument("--data", default=None, help="папка с articles.f/calibration.f/test.f")
    ap.add_argument("--out", default="answer.csv")
    args = ap.parse_args()

    data_dir = find_data_dir(args.data)
    print(f"Данные: {data_dir}")
    articles = pd.read_feather(data_dir / "articles.f")
    cal = pd.read_feather(data_dir / "calibration.f")
    test = pd.read_feather(data_dir / "test.f")
    article_ids = articles.article_id.tolist()
    valid_ids = set(article_ids)
    gt = {r.query_id: {int(x) for x in r.ground_truth.split()} for r in cal.itertuples()}
    cal_qids = cal.query_id.tolist()

    print(f"Статей: {len(articles)} | calibration: {len(cal)} | test: {len(test)}")
    print(f"Сигналы: {'BM25 + LSA + эмбеддинги' if args.use_emb else 'BM25 + LSA'}\n")

    signals = build_signals(articles, args.use_emb)

    # оценки и RRF-вклады на калибровке
    cal_contribs, cal_solo = {}, {}
    for name, sig in signals.items():
        S = sig.scores(cal.query_text.tolist())
        cal_contribs[name] = rrf_contributions(S, RRF_K)
        preds = dict(zip(cal_qids, topk_ids(S, article_ids, TOP_K)))
        cal_solo[name] = map_at_k(preds, gt)

    print("MAP@10 каждого сигнала по отдельности (вся калибровка):")
    for name, m in cal_solo.items():
        print(f"  {name:5}: {m:.4f}")

    cv_mean, cv_std, best_w, full_m = cross_validate(cal_contribs, article_ids, cal_qids, gt)
    print("\nОбъединение RRF:")
    print(f"веса: {best_w}")
    print(f"MAP@10 на всей калибровке: {full_m:.4f}")
    print(f"MAP@10 честно по CV: {cv_mean:.4f} +- {cv_std:.4f}")

    # применяем к test
    test_contribs = {name: rrf_contributions(sig.scores(test.query_text.tolist()), RRF_K)
                     for name, sig in signals.items()}
    fused = fuse(test_contribs, best_w)
    preds = topk_ids(fused, article_ids, TOP_K)

    answer = pd.DataFrame({
        "query_id": test.query_id,
        "answer": [" ".join(map(str, ids)) for ids in preds],
    })
    validate_answer(answer, test, valid_ids)
    answer.to_csv(args.out, index=False)
    print(f" сохранено: {args.out} ({len(answer)} строк)")


if __name__ == "__main__":
    main()