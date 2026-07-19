"""Ядро поиска: сигналы (BM25, LSA, эмбеддинги) и их объединение через RRF

Идея: BM25 и LSA/эмбеддинги ошибаются в разных местах, поэтому
их объединение через Reciprocal Rank Fusion (по рангам, а не по сырым
оценкам разного масштаба) даёт результат лучше каждого по отдельности.

Каждый сигнал — это объект, который умеет по списку запросов вернуть матрицу
оценок (n_queries x n_articles): чем больше, тем релевантнее
"""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from text_processing import clean_html, tokenize

SEED = 42


# Сигналы
class BM25Signal:
    """Лексический поиск BM25 (без стемминга - на этих данных он вредит)"""

    def __init__(self, docs, k1: float = 1.2, b: float = 0.4):
        self._tok = lambda s: tokenize(s, stem=False)
        self._bm = BM25Okapi([self._tok(d) for d in docs], k1=k1, b=b)

    def scores(self, queries) -> np.ndarray:
        return np.vstack([self._bm.get_scores(self._tok(q)) for q in queries])


class LSASignal:
    """Латентно-семантический анализ: TF-IDF -> SVD -> косинус в пространстве тем"""

    def __init__(self, docs, n_components: int = 100):
        self._vec = TfidfVectorizer(
            analyzer=lambda s: tokenize(s, stem=True), sublinear_tf=True, min_df=2
        )
        matrix = self._vec.fit_transform(docs)
        self._svd = TruncatedSVD(n_components=n_components, random_state=SEED)
        self._docs = normalize(self._svd.fit_transform(matrix))

    def scores(self, queries) -> np.ndarray:
        q = normalize(self._svd.transform(self._vec.transform(queries)))
        return cosine_similarity(q, self._docs)


class EmbeddingSignal:
    """Нейросетевые эмбеддинги би-энкодер с чанкованием статей.

    Требует sentence-transformers и интернета для первой загрузки модели.
    Модель <1B параметров, локально на CPU — соответствует правилам задания.
    Для семейства e5 обязательны префиксы 'query:' и 'passage:'.

    Почему чанки: e5 читает ~512 токенов, а статьи длинные (до 900k символов).
    Без чанкования модель обрезает статью до начала и теряет суть. Поэтому
    режем тело на куски по chunk_words слов (с перекрытием), к каждому куску
    приписываем заголовок, а похожесть статьи к запросу берём как максимум по
    её кускам (max-pooling) — нужный абзац не теряется.
    """

    def __init__(
        self,
        articles,
        model_name: str = "intfloat/multilingual-e5-base",
        chunk_words: int = 180,
        overlap: int = 40,
        batch_size: int = 64,
        max_seq_length: int = 256,
    ):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._model.max_seq_length = max_seq_length
        self._n_articles = len(articles)

        texts, owners = [], []
        titles = articles["title"].tolist()
        bodies = articles["body"].tolist()
        step = max(1, chunk_words - overlap)
        for idx, (title, body) in enumerate(zip(titles, bodies)):
            words = clean_html(body).split()
            if len(words) <= chunk_words:
                chunks = [" ".join(words)]
            else:
                chunks = [" ".join(words[i : i + chunk_words])
                          for i in range(0, len(words), step)]
                chunks = [c for c in chunks if c] or [""]
            for c in chunks:
                texts.append(f"passage: {title}. {c}")
                owners.append(idx)

        owners = np.asarray(owners)
        # первый индекс чанка каждой статьи (owners отсортированы по построению)
        self._starts = np.searchsorted(owners, np.arange(self._n_articles))
        self._chunk_emb = self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
        )

    def scores(self, queries) -> np.ndarray:
        q = self._model.encode(
            [f"query: {x}" for x in queries],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sim = q @ self._chunk_emb.T  # (n_queries x n_chunks)
        # max-pool по чанкам каждой статьи (блоки идут подряд)
        return np.maximum.reduceat(sim, self._starts, axis=1)


# Объединение по рангам (RRF)
def _rank_contribution(score_row: np.ndarray, rrf_k: int) -> np.ndarray:
    """1/(rrf_k + позиция) для каждого документа по одному запросу"""
    order = np.argsort(score_row)[::-1]
    rank = np.empty(len(score_row), dtype=int)
    rank[order] = np.arange(len(score_row))
    return 1.0 / (rrf_k + rank)


def rrf_contributions(score_matrix: np.ndarray, rrf_k: int = 10) -> np.ndarray:
    """Матрица вкладов RRF (n_queries x n_articles) для одного сигнала"""
    return np.vstack([_rank_contribution(score_matrix[i], rrf_k) for i in range(score_matrix.shape[0])])


def fuse(contribs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Взвешенная сумма RRF-вкладов сигналов -> итоговая оценка"""
    keys = list(contribs)
    total = np.zeros_like(contribs[keys[0]])
    for k in keys:
        total = total + weights.get(k, 0.0) * contribs[k]
    return total


def topk_ids(fused: np.ndarray, article_ids, k: int = 10):
    """Из матрицы оценок -> для каждого запроса топ-k article_id (ранжированные)"""
    article_ids = np.asarray(article_ids)
    return [article_ids[np.argsort(fused[i])[::-1][:k]].tolist() for i in range(fused.shape[0])]


# Подготовка текстов документов
def lexical_documents(articles, title_boost: int = 3):
    """Для BM25/LSA: заголовок усилен повтором (важнее тела) + чистый текст."""
    clean = articles["body"].map(clean_html)
    return ((articles["title"] + " ") * title_boost + clean).tolist()


def semantic_documents(articles):
    """Для эмбеддингов: заголовок + чистый текст (без искусственных повторов -
    трансформеру повторы не нужны, модель сама взвесит важность)"""
    clean = articles["body"].map(clean_html)
    return (articles["title"] + ". " + clean).tolist()