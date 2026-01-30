from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from gensim.models import Word2Vec
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchtext.datasets import AG_NEWS


@dataclass(frozen=True)
class TextDataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    input_dim: int
    num_classes: int


_TOKEN_RE = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _split_train_val(
    texts: list[str],
    labels: list[int],
    val_split: float,
    seed: int,
) -> tuple[list[str], list[str], list[int], list[int]]:
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=val_split,
        random_state=seed,
        stratify=labels,
    )
    return train_texts, val_texts, train_labels, val_labels


def _build_tfidf_features(
    train_texts: list[str],
    val_texts: list[str],
    test_texts: list[str],
    max_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    val_matrix = vectorizer.transform(val_texts)
    test_matrix = vectorizer.transform(test_texts)

    return (
        train_matrix.toarray().astype(np.float32),
        val_matrix.toarray().astype(np.float32),
        test_matrix.toarray().astype(np.float32),
    )


def _build_word2vec_features(
    train_texts: list[str],
    val_texts: list[str],
    test_texts: list[str],
    vector_size: int,
    min_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_tokens = [_tokenize(text) for text in train_texts]
    model = Word2Vec(
        sentences=train_tokens,
        vector_size=vector_size,
        window=5,
        min_count=min_count,
        workers=4,
        sg=1,
        epochs=10,
    )

    def vectorize(texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            tokens = [tok for tok in _tokenize(text) if tok in model.wv]
            if tokens:
                vec = np.mean(model.wv[tokens], axis=0)
            else:
                vec = np.zeros(vector_size, dtype=np.float32)
            vectors.append(vec)
        return np.vstack(vectors).astype(np.float32)

    return vectorize(train_texts), vectorize(val_texts), vectorize(test_texts)


def _to_loader(features: np.ndarray, labels: list[int], batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = torch.from_numpy(features)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _load_ag_news(data_root: Path) -> tuple[list[str], list[int], list[str], list[int]]:
    train_data = list(AG_NEWS(root=str(data_root)))
    test_data = list(AG_NEWS(root=str(data_root), split="test"))

    train_texts = [text for label, text in train_data]
    train_labels = [label - 1 for label, _ in train_data]
    test_texts = [text for label, text in test_data]
    test_labels = [label - 1 for label, _ in test_data]

    return train_texts, train_labels, test_texts, test_labels


def _load_20newsgroups(data_root: Path) -> tuple[list[str], list[int], list[str], list[int]]:
    train = fetch_20newsgroups(subset="train", data_home=str(data_root))
    test = fetch_20newsgroups(subset="test", data_home=str(data_root))
    return train.data, train.target.tolist(), test.data, test.target.tolist()


def _cache_path(
    cache_dir: Path,
    dataset: str,
    feature: str,
    seed: int,
    val_split: float,
    max_features: int,
    word2vec_dim: int,
    word2vec_min_count: int,
) -> Path:
    key = f"{dataset}|{feature}|{seed}|{val_split}|{max_features}|{word2vec_dim}|{word2vec_min_count}"
    digest = md5(key.encode("utf-8")).hexdigest()
    return cache_dir / f"{dataset}_{feature}_{digest}.npz"


def _load_cached(cache_path: Path) -> TextDataLoaders | None:
    if not cache_path.exists():
        return None
    data = np.load(cache_path)
    return data


def _save_cache(
    cache_path: Path,
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    train_y: list[int],
    val_y: list[int],
    test_y: list[int],
    num_classes: int,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        train_x=train_x,
        val_x=val_x,
        test_x=test_x,
        train_y=np.array(train_y, dtype=np.int64),
        val_y=np.array(val_y, dtype=np.int64),
        test_y=np.array(test_y, dtype=np.int64),
        num_classes=np.array([num_classes], dtype=np.int64),
    )


def get_text_loaders(
    dataset: str,
    feature: str,
    data_root: Path,
    batch_size: int,
    val_split: float = 0.1,
    seed: int = 42,
    max_features: int = 20000,
    word2vec_dim: int = 300,
    word2vec_min_count: int = 2,
    cache_dir: Path | None = None,
) -> TextDataLoaders:
    dataset = dataset.lower()
    feature = feature.lower()

    cache_path = None
    if cache_dir is not None:
        cache_path = _cache_path(
            cache_dir,
            dataset,
            feature,
            seed,
            val_split,
            max_features,
            word2vec_dim,
            word2vec_min_count,
        )
        cached = _load_cached(cache_path)
        if cached is not None:
            train_x = cached["train_x"]
            val_x = cached["val_x"]
            test_x = cached["test_x"]
            train_y = cached["train_y"].tolist()
            val_y = cached["val_y"].tolist()
            test_y = cached["test_y"].tolist()
            num_classes = int(cached["num_classes"][0])

            train_loader = _to_loader(train_x, train_y, batch_size, shuffle=True)
            val_loader = _to_loader(val_x, val_y, batch_size, shuffle=False)
            test_loader = _to_loader(test_x, test_y, batch_size, shuffle=False)

            return TextDataLoaders(
                train=train_loader,
                val=val_loader,
                test=test_loader,
                input_dim=train_x.shape[1],
                num_classes=num_classes,
            )

    if dataset == "ag_news":
        train_texts, train_labels, test_texts, test_labels = _load_ag_news(data_root)
        num_classes = 4
    elif dataset == "20newsgroups":
        train_texts, train_labels, test_texts, test_labels = _load_20newsgroups(data_root)
        num_classes = 20
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_texts, val_texts, train_labels, val_labels = _split_train_val(
        train_texts, train_labels, val_split, seed
    )

    if feature == "tfidf":
        train_x, val_x, test_x = _build_tfidf_features(
            train_texts, val_texts, test_texts, max_features=max_features
        )
    elif feature == "word2vec":
        train_x, val_x, test_x = _build_word2vec_features(
            train_texts,
            val_texts,
            test_texts,
            vector_size=word2vec_dim,
            min_count=word2vec_min_count,
        )
    else:
        raise ValueError(f"Unknown feature: {feature}")

    if cache_path is not None:
        _save_cache(
            cache_path,
            train_x,
            val_x,
            test_x,
            train_labels,
            val_labels,
            test_labels,
            num_classes,
        )

    train_loader = _to_loader(train_x, train_labels, batch_size, shuffle=True)
    val_loader = _to_loader(val_x, val_labels, batch_size, shuffle=False)
    test_loader = _to_loader(test_x, test_labels, batch_size, shuffle=False)

    return TextDataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        input_dim=train_x.shape[1],
        num_classes=num_classes,
    )
