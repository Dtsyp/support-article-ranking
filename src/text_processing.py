"""Очистка HTML статей и токенизация текста для поиска"""
import re
from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer
from stop_words import get_stop_words

_STEMMER = SnowballStemmer("russian")
STOPWORDS = frozenset(w.replace("ё", "е") for w in get_stop_words("russian"))

def clean_html(html):
    """HTML статьи в плоский текст"""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]): # выкидываем код
        tag.decompose()
    text = soup.get_text(separator=" ") # separator=' ' — не склеивать слова
    return re.sub(r"\s+", " ", text).strip() # схлопнуть пробелы


_WORD_RE = re.compile(r"[а-яёa-z0-9]+")

def tokenize(text, stem=True, drop_stopwords=True):
    """нижний регистр → слова → стоп-слова → стемминг"""
    words = _WORD_RE.findall(text.lower().replace("ё", "е"))  # выкидываем заглушки типа <MONEY>
    if drop_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    if stem:
        words = [_STEMMER.stem(w) for w in words]
    return words