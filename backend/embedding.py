"""文本向量化服务 - 支持密集向量和稀疏向量（BM25），词表与 df 持久化 + 增量更新"""
import json
import logging
import math
import os
import re
import threading
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

_DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "bm25_state.json"
_DEFAULT_CPU_EMBEDDING_BATCH_SIZE = 8
_DEFAULT_CUDA_EMBEDDING_BATCH_SIZE = 16
_DEFAULT_CUDA_EMBEDDING_MAX_TOKENS = 8192
logger = logging.getLogger(__name__)

# The local BGE-M3 cache contains the PyTorch weights but not a processor
# config. Newer transformers versions try AutoProcessor first and fail before
# the tokenizer can be used. Disable their background conversion request when
# the application is configured for local/offline model use.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _resolve_embedding_device() -> str:
    requested = os.getenv("EMBEDDING_DEVICE", "auto").strip().lower()
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False
    if requested in {"", "auto"}:
        return "cuda" if cuda_available else "cpu"
    if requested.startswith("cuda") and not cuda_available:
        logger.warning("EMBEDDING_DEVICE=%s requested but CUDA is unavailable; falling back to CPU", requested)
        return "cpu"
    return requested


def _should_use_fp16(device: str) -> bool:
    return device.startswith("cuda") and _parse_bool(os.getenv("EMBEDDING_USE_FP16"), True)


class _LocalTransformerEmbeddings:
    """Minimal local fallback for old text-only SentenceTransformer caches."""

    def __init__(self, model_name: str, device: str, local_only: bool, revision: str, use_fp16: bool):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        load_kwargs = {"local_files_only": local_only}
        if revision:
            load_kwargs["revision"] = revision

        # Passing the repository id can make transformers 5.x call model_info
        # even with local_files_only=True. Resolve the cached snapshot first.
        model_path = model_name
        if local_only and "/" in model_name:
            model_path = snapshot_download(
                repo_id=model_name,
                revision=revision or None,
                local_files_only=True,
            )

        self._torch = torch
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, **load_kwargs)
        self._model = AutoModel.from_pretrained(model_path, **load_kwargs)
        self._model.to(device)
        if use_fp16:
            self._model.half()
        self._model.eval()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=min(getattr(self._tokenizer, "model_max_length", 8192), 8192),
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.no_grad():
            output = self._model(**encoded).last_hidden_state[:, 0]
            output = self._torch.nn.functional.normalize(output, p=2, dim=1)
        return output.detach().cpu().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# def _create_dense_embedder() -> HuggingFaceEmbeddings:
#     model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
#     device = os.getenv("EMBEDDING_DEVICE", "cpu")
#     return HuggingFaceEmbeddings(
#         model_name=model_name,
#         model_kwargs={"device": device},
#         encode_kwargs={"normalize_embeddings": True},
#     )


def _create_dense_embedder(device: str | None = None) -> HuggingFaceEmbeddings:
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    device = device or _resolve_embedding_device()
    local_only = os.getenv("EMBEDDING_LOCAL_ONLY", "1") == "1"
    revision = os.getenv("EMBEDDING_REVISION", "").strip()
    use_fp16 = _should_use_fp16(device)

    model_kwargs = {
        "device": device,
        "local_files_only": local_only,
    }
    if revision:
        model_kwargs["revision"] = revision

    # BGE-M3's cached sentence-transformers files are text-only. Avoid the
    # transformers 5.x AutoProcessor path and use the same local weights.
    try:
        import transformers

        major_version = int(str(transformers.__version__).split(".", 1)[0])
    except (ImportError, TypeError, ValueError):
        major_version = 0
    if "bge-m3" in model_name.lower() and major_version >= 5:
        logger.warning("Using local BGE-M3 tokenizer/model fallback for transformers %s", transformers.__version__)
        return _LocalTransformerEmbeddings(model_name, device, local_only, revision, use_fp16)

    embedder = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )
    if use_fp16:
        half = getattr(getattr(embedder, "client", None), "half", None)
        if callable(half):
            half()
    return embedder

class EmbeddingService:
    """文本向量化服务 - 密集向量本地模型 + BM25 稀疏向量（持久化统计）"""

    def __init__(self, state_path: Path | str | None = None):
        self._device = _resolve_embedding_device()
        self._embedder = _create_dense_embedder(device=self._device)
        self._state_path = Path(state_path or os.getenv("BM25_STATE_PATH", _DEFAULT_STATE_PATH))
        self._lock = threading.Lock()

        # BM25 参数
        self.k1 = 1.5
        self.b = 0.75

        self._vocab: dict[str, int] = {}
        self._vocab_counter = 0
        self._doc_freq: Counter[str] = Counter()
        self._total_docs = 0
        self._sum_token_len = 0
        self._avg_doc_len = 1.0

        self._load_state()

    def _recompute_avg_len(self) -> None:
        self._avg_doc_len = (
            self._sum_token_len / self._total_docs if self._total_docs > 0 else 1.0
        )

    def _load_state(self) -> None:
        path = self._state_path
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if raw.get("version") != 1:
            return
        self._vocab = {str(k): int(v) for k, v in raw.get("vocab", {}).items()}
        self._doc_freq = Counter({str(k): int(v) for k, v in raw.get("doc_freq", {}).items()})
        self._total_docs = int(raw.get("total_docs", 0))
        self._sum_token_len = int(raw.get("sum_token_len", 0))
        if self._vocab:
            self._vocab_counter = max(self._vocab.values()) + 1
        else:
            self._vocab_counter = 0
        self._recompute_avg_len()

    def _persist_unlocked(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "total_docs": self._total_docs,
            "sum_token_len": self._sum_token_len,
            "vocab": self._vocab,
            "doc_freq": dict(self._doc_freq),
        }
        tmp = self._state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._state_path)

    def _persist(self) -> None:
        with self._lock:
            self._persist_unlocked()

    def increment_add_documents(self, texts: list[str]) -> None:
        """
        将每个 text 视为 BM25 中的一篇文档（与当前 chunk 写入粒度一致），增量更新 N / df / 长度和。
        """
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._sum_token_len += doc_len
                self._total_docs += 1
                for token in set(tokens):
                    if token not in self._vocab:
                        self._vocab[token] = self._vocab_counter
                        self._vocab_counter += 1
                    self._doc_freq[token] += 1
            self._recompute_avg_len()
            self._persist_unlocked()

    def increment_remove_documents(self, texts: list[str]) -> None:
        """
        从语料统计中移除与 increment_add_documents 对称的文档集合（如删除某文件的全部 chunk 文本）。
        词表索引不回收，避免与 Milvus 中仍可能存在的旧稀疏向量维度冲突。
        """
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._sum_token_len = max(0, self._sum_token_len - doc_len)
                self._total_docs = max(0, self._total_docs - 1)
                for token in set(tokens):
                    if token not in self._doc_freq:
                        continue
                    self._doc_freq[token] -= 1
                    if self._doc_freq[token] <= 0:
                        del self._doc_freq[token]
            self._recompute_avg_len()
            self._persist_unlocked()

    def _get_embedding_batch_size(self) -> int:
        raw_value = os.getenv("EMBEDDING_BATCH_SIZE")
        default = (
            _DEFAULT_CUDA_EMBEDDING_BATCH_SIZE
            if getattr(self, "_device", "cpu").startswith("cuda")
            else _DEFAULT_CPU_EMBEDDING_BATCH_SIZE
        )
        try:
            batch_size = int(raw_value) if raw_value is not None else default
        except (TypeError, ValueError):
            batch_size = default
        return batch_size if batch_size > 0 else default

    def _get_embedding_max_tokens(self) -> int:
        """Bound padded tokens per GPU micro-batch, not merely document count."""
        if not getattr(self, "_device", "cpu").startswith("cuda"):
            return 0
        raw_value = os.getenv("EMBEDDING_MAX_BATCH_TOKENS")
        try:
            value = int(raw_value) if raw_value is not None else _DEFAULT_CUDA_EMBEDDING_MAX_TOKENS
        except (TypeError, ValueError):
            value = _DEFAULT_CUDA_EMBEDDING_MAX_TOKENS
        return value if value > 0 else _DEFAULT_CUDA_EMBEDDING_MAX_TOKENS

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Conservative local estimate used only for length bucketing.

        Transformer batching pads every item to the longest item in a batch.  A
        lightweight estimate is enough to prevent a long PDF page from turning
        a batch of short financial chunks into an expensive padded forward pass.
        """
        return max(1, len(re.findall(r"[$€£¥]?\d[\d,]*(?:\.\d+)?%?|[A-Za-z]+(?:[-'][A-Za-z]+)*|[\u4e00-\u9fff]|\S", text or "")))

    def _build_embedding_batches(self, texts: list[str], max_items: int) -> list[list[int]]:
        """Return original indexes grouped by similar length and token budget."""
        ordered = sorted(range(len(texts)), key=lambda index: self._estimate_tokens(texts[index]))
        token_budget = self._get_embedding_max_tokens()
        batches: list[list[int]] = []
        batch: list[int] = []
        longest = 0
        for index in ordered:
            estimate = self._estimate_tokens(texts[index])
            exceeds_budget = bool(batch and token_budget and max(longest, estimate) * (len(batch) + 1) > token_budget)
            if len(batch) >= max_items or exceeds_budget:
                batches.append(batch)
                batch, longest = [], 0
            batch.append(index)
            longest = max(longest, estimate)
        if batch:
            batches.append(batch)
        return batches

    def _maybe_empty_cuda_cache(self) -> None:
        try:
            import torch  # type: ignore
        except Exception:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        return "out of memory" in str(exc).lower()

    def get_embeddings(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        if not texts:
            return []
        batch_size = max(1, int(batch_size or self._get_embedding_batch_size()))
        try:
            embeddings: list[list[float] | None] = [None] * len(texts)
            batches = self._build_embedding_batches(texts, batch_size)
            batch_index = 0
            active_batch_size = batch_size
            while batch_index < len(batches):
                indexes = batches[batch_index]
                if len(indexes) > active_batch_size:
                    batches[batch_index:batch_index + 1] = [
                        indexes[offset:offset + active_batch_size]
                        for offset in range(0, len(indexes), active_batch_size)
                    ]
                    continue
                batch = [texts[index] for index in indexes]
                try:
                    values = self._embedder.embed_documents(batch)
                    for index, value in zip(indexes, values):
                        embeddings[index] = value
                except Exception as e:
                    if (
                        getattr(self, "_device", "cpu").startswith("cuda")
                        and self._is_cuda_oom(e)
                        and active_batch_size > 1
                    ):
                        next_batch_size = max(1, active_batch_size // 2)
                        logger.warning(
                            "CUDA embedding OOM for batch=%s; retrying with batch=%s",
                            active_batch_size,
                            next_batch_size,
                        )
                        active_batch_size = next_batch_size
                        self._maybe_empty_cuda_cache()
                        continue
                    first_index = min(indexes) + 1
                    last_index = max(indexes) + 1
                    raise Exception(
                        f"本地嵌入模型批处理失败: batch={first_index}-{last_index}: {str(e)}"
                    ) from e
                batch_index += 1
            if any(item is None for item in embeddings):
                raise RuntimeError("embedding batch completed with missing vectors")
            return [item for item in embeddings if item is not None]
        except Exception as e:
            raise Exception(f"本地嵌入模型调用失败: {str(e)}") from e

    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = []
        chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
        english_pattern = re.compile(r"[a-zA-Z]+")
        number_pattern = re.compile(r"(?:[$€£¥]\s*)?\d[\d,]*(?:\.\d+)?%?")
        i = 0
        while i < len(text):
            char = text[i]
            if chinese_pattern.match(char):
                tokens.append(char)
                i += 1
            elif english_pattern.match(char):
                match = english_pattern.match(text[i:])
                if match:
                    tokens.append(match.group())
                    i += len(match.group())
            elif number_pattern.match(text[i:]):
                match = number_pattern.match(text[i:])
                if match:
                    tokens.append(re.sub(r"\s+", "", match.group()))
                    i += len(match.group())
            else:
                i += 1
        return tokens

    def _sparse_vector_for_text_unlocked(self, text: str) -> tuple[dict, bool]:
        tokens = self.tokenize(text)
        doc_len = len(tokens)
        tf = Counter(tokens)
        sparse_vector: dict[int, float] = {}
        vocab_changed = False
        n = max(self._total_docs, 0)
        avg = max(self._avg_doc_len, 1.0)

        for token, freq in tf.items():
            if token not in self._vocab:
                self._vocab[token] = self._vocab_counter
                self._vocab_counter += 1
                vocab_changed = True

            idx = self._vocab[token]
            df = self._doc_freq.get(token, 0)
            if df == 0:
                idf = math.log((n + 1) / 1)
            else:
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / avg)
            score = idf * numerator / denominator
            if score > 0:
                sparse_vector[idx] = float(score)

        return sparse_vector, vocab_changed

    def get_sparse_embedding(self, text: str) -> dict:
        with self._lock:
            sparse_vector, vocab_changed = self._sparse_vector_for_text_unlocked(text)
            if vocab_changed:
                self._persist_unlocked()
        return sparse_vector

    def get_sparse_embeddings(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []
        with self._lock:
            out: list[dict] = []
            any_new_vocab = False
            for text in texts:
                sparse_vector, vocab_changed = self._sparse_vector_for_text_unlocked(text)
                out.append(sparse_vector)
                any_new_vocab = any_new_vocab or vocab_changed
            if any_new_vocab:
                self._persist_unlocked()
        return out

    def get_all_embeddings(self, texts: list[str]) -> tuple[list[list[float]], list[dict]]:
        dense_embeddings = self.get_embeddings(texts)
        sparse_embeddings = self.get_sparse_embeddings(texts)
        return dense_embeddings, sparse_embeddings


# 全进程唯一实例：写入与检索共用同一份 BM25 持久化状态
embedding_service = EmbeddingService()
