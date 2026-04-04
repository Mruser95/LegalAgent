"""
Pipeline:
  Query → 实体提取(元信息过滤) → BM25 + Vector 多路检索 → RRF 融合截断 → Rerank 重排截断 → 返回结果

用法:
  from retriver import LegalRetriever

  retriever = LegalRetriever()
  results = retriever.retrieve("刑法第二百六十四条关于盗窃罪的规定")
  for r in results:
      print(r.score, r.text[:80])
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jieba
from rank_bm25 import BM25Okapi

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

_RAG_DIR = str(Path(__file__).resolve().parent)
if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

from load_store import (
    DEFAULT_CHUNKS_DIR,
    DEFAULT_DB_URL,
    DEFAULT_EMBED_DIM,
    DEFAULT_EMBED_MODEL,
    DEFAULT_TABLE,
    create_vector_store,
    load_all_chunks,
)

logging.getLogger("jieba").setLevel(logging.WARNING)

__all__ = [
    "LegalRetriever",
    "RetrievalResult",
    "ExtractedEntities",
    "EntityExtractor",
]

# ═══════════════════════ Chinese numeral helpers ══════════════════════

_CN_DIGIT = {
    "零": 0, "〇": 0,
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}


def _cn_num_to_int(s: str) -> int | None:
    if not s:
        return None
    if s.isdigit():
        return int(s)
    try:
        result = 0
        current = 0
        for ch in s:
            if ch in _CN_DIGIT:
                current = _CN_DIGIT[ch]
            elif ch in _CN_UNIT:
                if current == 0:
                    current = 1
                result += current * _CN_UNIT[ch]
                current = 0
            else:
                return None
        return (result + current) or None
    except Exception:
        return None


# ═══════════════════════ Entity Extraction ════════════════════════════

_CN_RE = r"[一二三四五六七八九十百千零〇]+"

RE_LAW_BRACKET = re.compile(r"[《〈](.+?)[》〉]")
RE_ARTICLE_REF = re.compile(rf"第({_CN_RE}|\d+)条")
RE_CHAPTER_REF = re.compile(rf"第({_CN_RE}|\d+)章")
RE_SECTION_REF = re.compile(rf"第({_CN_RE}|\d+)节")
RE_PART_REF = re.compile(rf"第({_CN_RE}|\d+)编")


@dataclass
class ExtractedEntities:
    """查询中提取出的法律元信息"""

    law_names: List[str] = field(default_factory=list)
    article_numbers: List[int] = field(default_factory=list)
    article_labels: List[str] = field(default_factory=list)
    chapter_refs: List[str] = field(default_factory=list)
    section_refs: List[str] = field(default_factory=list)
    part_refs: List[str] = field(default_factory=list)
    is_amendment: bool = False

    def has_filters(self) -> bool:
        return bool(self.law_names or self.article_numbers)


class EntityExtractor:
    """从查询文本中提取法律名称、条款号等元信息，用于过滤和规范检索字段"""

    def __init__(self, known_titles: List[str]):
        self._known_titles = sorted(set(known_titles))
        self._short_to_full: Dict[str, List[str]] = {}
        for t in self._known_titles:
            self._short_to_full.setdefault(t, []).append(t)
            short = t.replace("中华人民共和国", "")
            if short and short != t:
                self._short_to_full.setdefault(short, []).append(t)

    def extract(self, query: str) -> ExtractedEntities:
        entities = ExtractedEntities()

        # — 法律名称: 优先从书名号《》提取 —
        for m in RE_LAW_BRACKET.finditer(query):
            entities.law_names.extend(self._resolve_title(m.group(1)))

        # — 无书名号时进行子串匹配 —
        if not entities.law_names:
            entities.law_names = self._substring_match(query)

        entities.law_names = list(dict.fromkeys(entities.law_names))

        # — 条款号 —
        for m in RE_ARTICLE_REF.finditer(query):
            raw = m.group(1)
            num = _cn_num_to_int(raw)
            if num is not None:
                entities.article_numbers.append(num)
                entities.article_labels.append(f"第{raw}条")
        entities.article_numbers = list(dict.fromkeys(entities.article_numbers))

        # — 编/章/节 —
        for m in RE_CHAPTER_REF.finditer(query):
            entities.chapter_refs.append(m.group(0))
        for m in RE_SECTION_REF.finditer(query):
            entities.section_refs.append(m.group(0))
        for m in RE_PART_REF.finditer(query):
            entities.part_refs.append(m.group(0))

        # — 修正案检测 —
        if any(kw in query for kw in ("修正案", "修正", "修订")):
            entities.is_amendment = True

        return entities

    def _resolve_title(self, name: str) -> List[str]:
        """将书名号内的名称解析为已知的法律全称"""
        if name in self._short_to_full:
            return list(self._short_to_full[name])
        full = f"中华人民共和国{name}"
        if full in self._short_to_full:
            return list(self._short_to_full[full])
        return [t for t in self._known_titles if name in t]

    def _substring_match(self, query: str) -> List[str]:
        """在查询文本中以子串方式匹配已知法律名称（优先长匹配）"""
        candidates = sorted(
            self._short_to_full.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )
        matched_titles: List[str] = []
        consumed: List[Tuple[int, int]] = []

        for short, fulls in candidates:
            if len(short) < 2:
                continue
            idx = query.find(short)
            if idx == -1:
                continue
            span = (idx, idx + len(short))
            if any(s <= span[0] < e or s < span[1] <= e for s, e in consumed):
                continue
            consumed.append(span)
            matched_titles.extend(fulls)

        return list(dict.fromkeys(matched_titles))


# ═══════════════════════ Metadata Filter Builders ════════════════════

def _build_vector_filters(entities: ExtractedEntities) -> Optional[MetadataFilters]:
    """构建 LlamaIndex MetadataFilters 用于向量检索的元数据过滤"""
    filters: List[MetadataFilter] = []

    if entities.law_names:
        if len(entities.law_names) == 1:
            filters.append(MetadataFilter(
                key="title", value=entities.law_names[0],
                operator=FilterOperator.EQ,
            ))
        else:
            filters.append(MetadataFilter(
                key="title", value=entities.law_names,
                operator=FilterOperator.IN,
            ))

    if entities.article_numbers:
        if len(entities.article_numbers) == 1:
            filters.append(MetadataFilter(
                key="article_number", value=entities.article_numbers[0],
                operator=FilterOperator.EQ,
            ))
        else:
            filters.append(MetadataFilter(
                key="article_number", value=entities.article_numbers,
                operator=FilterOperator.IN,
            ))

    if entities.is_amendment:
        filters.append(MetadataFilter(
            key="document_type", value="amendment",
            operator=FilterOperator.EQ,
        ))

    return MetadataFilters(filters=filters, condition=FilterCondition.AND) if filters else None


def _filter_node_indices(
    nodes: List[TextNode],
    entities: ExtractedEntities,
) -> Optional[List[int]]:
    """根据提取的实体过滤 BM25 候选节点索引"""
    if not entities.has_filters():
        return None

    indices: List[int] = []
    for i, node in enumerate(nodes):
        meta = node.metadata
        if entities.law_names and meta.get("title") not in entities.law_names:
            continue
        if entities.article_numbers and meta.get("article_number") not in entities.article_numbers:
            continue
        if entities.is_amendment and meta.get("document_type") != "amendment":
            continue
        indices.append(i)

    return indices or None


# ═══════════════════════ Chinese BM25 Retriever ══════════════════════

class ChineseBM25Retriever:
    """基于 jieba 分词的中文 BM25 检索器"""

    def __init__(self, nodes: List[TextNode]):
        self._nodes = nodes
        corpus_tokens = [self._tokenize(n.text) for n in nodes]
        self._bm25 = BM25Okapi(corpus_tokens)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in jieba.lcut_for_search(text) if t.strip()]

    def retrieve(
        self,
        query: str,
        top_k: int = 30,
        filter_indices: Optional[List[int]] = None,
    ) -> List[Tuple[str, float]]:
        """
        返回 [(node_id, bm25_score), ...] 按分数降序排列。
        filter_indices 不为 None 时仅在指定索引范围内检索。
        """
        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        if filter_indices is not None:
            candidates = [(i, float(scores[i])) for i in filter_indices if scores[i] > 0]
        else:
            candidates = [(i, float(s)) for i, s in enumerate(scores) if s > 0]

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [
            (self._nodes[i].node_id, score)
            for i, score in candidates[:top_k]
        ]


# ═══════════════════════ RRF Fusion ══════════════════════════════════

def reciprocal_rank_fusion(
    *ranked_lists: List[Tuple[str, float]],
    k: int = 60,
    top_n: int = 30,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion: 将多路检索结果合并为统一排序。
    score(d) = Σ 1 / (k + rank_i)
    """
    rrf_scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (node_id, _) in enumerate(ranked, start=1):
            rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_n]


# ═══════════════════════ Cross-Encoder Reranker ══════════════════════

class CrossEncoderReranker:
    """使用 FlagEmbedding BGE-Reranker 进行交叉编码重排序"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        try:
            from FlagEmbedding import FlagReranker
        except ImportError:
            raise ImportError(
                "Reranker 需要 FlagEmbedding: pip install FlagEmbedding"
            )
        self._reranker = FlagReranker(model_name, use_fp16=True)
        self._model_name = model_name

    def rerank(
        self,
        query: str,
        node_map: Dict[str, TextNode],
        candidates: List[Tuple[str, float]],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """对 RRF 融合后的候选文档进行精排，返回 top_k 结果。"""
        if not candidates:
            return []

        pairs: List[List[str]] = []
        valid_ids: List[str] = []
        for node_id, _ in candidates:
            node = node_map.get(node_id)
            if node:
                pairs.append([query, node.text])
                valid_ids.append(node_id)

        if not pairs:
            return []

        scores = self._reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, (int, float)):
            scores = [scores]

        scored = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ═══════════════════════ Result Container ════════════════════════════

@dataclass
class RetrievalResult:
    """单条检索结果"""
    node_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    source: str


# ═══════════════════════ Main Retriever ══════════════════════════════

class LegalRetriever:
    """
    法律文档多路检索器

    初始化流程:
      1. 加载所有 chunk 到内存 (用于 BM25 + 实体库)
      2. 构建 jieba BM25 索引
      3. 连接 PGVectorStore 向量数据库
      4. 加载 CrossEncoder Reranker

    检索流程:
      Query → 实体提取 → 元信息过滤 → BM25 + Vector 并行检索
            → RRF 融合截断 → Rerank 重排截断 → 返回结果
    """

    def __init__(
        self,
        chunks_dir: str = DEFAULT_CHUNKS_DIR,
        db_url: Optional[str] = None,
        table_name: str = DEFAULT_TABLE,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        embed_dim: int = DEFAULT_EMBED_DIM,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        bm25_top_k: int = 30,
        vector_top_k: int = 30,
        rrf_top_n: int = 30,
        rrf_k: int = 60,
        final_top_k: int = 10,
        rerank_enabled: bool = True,
    ):
        self.bm25_top_k = bm25_top_k
        self.vector_top_k = vector_top_k
        self.rrf_top_n = rrf_top_n
        self.rrf_k = rrf_k
        self.final_top_k = final_top_k
        self.rerank_enabled = rerank_enabled

        db_url = db_url or os.environ.get("PG_URL", DEFAULT_DB_URL)

        # ── 1. 加载 chunks → 内存 ──
        print("[init] 加载文档 chunks ...")
        self._nodes: List[TextNode] = load_all_chunks(chunks_dir)
        self._node_map: Dict[str, TextNode] = {n.node_id: n for n in self._nodes}
        print(f"[init]   共 {len(self._nodes)} 个节点")

        # ── 2. 实体提取器 ──
        known_titles = list({
            n.metadata["title"]
            for n in self._nodes
            if n.metadata.get("title")
        })
        self._entity_extractor = EntityExtractor(known_titles)

        # ── 3. BM25 索引 ──
        print("[init] 构建中文 BM25 索引 (jieba 分词) ...")
        self._bm25 = ChineseBM25Retriever(self._nodes)

        # ── 4. 向量检索 ──
        print("[init] 加载 embedding 模型 & 连接向量数据库 ...")
        embed_model = HuggingFaceEmbedding(
            model_name=embed_model_name,
            embed_batch_size=64,
        )
        Settings.embed_model = embed_model

        self._vector_store = create_vector_store(
            db_url=db_url,
            table_name=table_name,
            embed_dim=embed_dim,
        )
        self._vector_index = VectorStoreIndex.from_vector_store(
            self._vector_store,
        )

        # ── 5. Reranker ──
        self._reranker: Optional[CrossEncoderReranker] = None
        if rerank_enabled:
            print(f"[init] 加载 reranker: {reranker_model} ...")
            self._reranker = CrossEncoderReranker(reranker_model)

        print("[init] 检索器初始化完成\n")

    # ── 公开 API ──

    def retrieve(
        self,
        query: str,
        bm25_top_k: Optional[int] = None,
        vector_top_k: Optional[int] = None,
        rrf_top_n: Optional[int] = None,
        final_top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        执行完整的多路检索 pipeline，返回排序后的检索结果列表。

        Args:
            query: 用户查询文本
            bm25_top_k: BM25 召回数量 (覆盖默认值)
            vector_top_k: 向量召回数量 (覆盖默认值)
            rrf_top_n: RRF 融合后保留数量 (覆盖默认值)
            final_top_k: 最终返回数量 (覆盖默认值)
        """
        bm25_k = bm25_top_k or self.bm25_top_k
        vec_k = vector_top_k or self.vector_top_k
        rrf_n = rrf_top_n or self.rrf_top_n
        final_k = final_top_k or self.final_top_k

        # ── Step 1: 实体提取 ──
        entities = self._entity_extractor.extract(query)
        print(
            f"[实体提取] 法律: {entities.law_names or '—'}, "
            f"条款: {entities.article_labels or '—'}, "
            f"修正案: {entities.is_amendment}"
        )

        # ── Step 2a: BM25 检索 ──
        bm25_filter = _filter_node_indices(self._nodes, entities)
        if bm25_filter is not None:
            print(f"[BM25] 元信息过滤后候选 {len(bm25_filter)} 个节点")
        bm25_results = self._bm25.retrieve(
            query, top_k=bm25_k, filter_indices=bm25_filter,
        )
        print(f"[BM25] 召回 {len(bm25_results)} 条")

        # ── Step 2b: Vector 检索 ──
        vec_filters = _build_vector_filters(entities)
        vec_retriever = self._vector_index.as_retriever(
            similarity_top_k=vec_k,
            filters=vec_filters,
        )
        vec_nodes = vec_retriever.retrieve(query)

        if not vec_nodes and vec_filters:
            vec_retriever_fallback = self._vector_index.as_retriever(
                similarity_top_k=vec_k,
            )
            vec_nodes = vec_retriever_fallback.retrieve(query)
            print(f"[Vector] 过滤无结果，回退无过滤召回 {len(vec_nodes)} 条")
        else:
            print(f"[Vector] 召回 {len(vec_nodes)} 条")

        vector_results = [
            (n.node.node_id, float(n.score) if n.score is not None else 0.0)
            for n in vec_nodes
        ]

        # ── Step 3: RRF 融合 ──
        fused = reciprocal_rank_fusion(
            bm25_results, vector_results,
            k=self.rrf_k, top_n=rrf_n,
        )
        print(f"[RRF] 融合后 {len(fused)} 条 (k={self.rrf_k})")

        # ── Step 4: Rerank ──
        if self.rerank_enabled and self._reranker and fused:
            reranked = self._reranker.rerank(
                query, self._node_map, fused, top_k=final_k,
            )
            print(f"[Rerank] 精排后 {len(reranked)} 条")
            source_label = "reranked"
        else:
            reranked = fused[:final_k]
            source_label = "fused"

        # ── 组装结果 ──
        results: List[RetrievalResult] = []
        for node_id, score in reranked:
            node = self._node_map.get(node_id)
            if node:
                results.append(RetrievalResult(
                    node_id=node_id,
                    text=node.text,
                    score=float(score),
                    metadata=dict(node.metadata),
                    source=source_label,
                ))
        return results

    def extract_entities(self, query: str) -> ExtractedEntities:
        """仅执行实体提取，不进行检索（调试用）。"""
        return self._entity_extractor.extract(query)


# ═══════════════════════ CLI Demo ════════════════════════════════════

def main():
    """交互式检索演示"""
    import argparse

    parser = argparse.ArgumentParser(description="法律文档多路检索器")
    parser.add_argument("--chunks-dir", default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--db-url", default=os.environ.get("PG_URL", DEFAULT_DB_URL))
    parser.add_argument("--table-name", default=DEFAULT_TABLE)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    retriever = LegalRetriever(
        chunks_dir=args.chunks_dir,
        db_url=args.db_url,
        table_name=args.table_name,
        embed_model_name=args.embed_model,
        reranker_model=args.reranker_model,
        rerank_enabled=not args.no_rerank,
        final_top_k=args.top_k,
    )

    print("=" * 60)
    print("法律文档检索器 — 输入查询，空行退出")
    print("=" * 60)

    while True:
        try:
            query = input("\n查询> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            break

        results = retriever.retrieve(query)
        print(f"\n{'─' * 60}")
        print(f"共返回 {len(results)} 条结果:\n")
        for i, r in enumerate(results, 1):
            title = r.metadata.get("title", "")
            article = r.metadata.get("article_label", "") or ""
            preview = r.text.replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:120] + "..."
            print(f"  [{i}] score={r.score:.4f}  {title} {article}")
            print(f"      {preview}")
            print()


if __name__ == "__main__":
    main()
