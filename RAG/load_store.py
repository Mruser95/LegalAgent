"""
  python RAG/load_store.py
  python RAG/load_store.py --chunks-dir RAG/chunks_json
  python RAG/load_store.py --db-url "postgresql://user:pass@localhost:5432/legal_db"
  python RAG/load_store.py --embed-model BAAI/bge-large-zh-v1.5 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

DEFAULT_CHUNKS_DIR = "RAG/chunks_json"
DEFAULT_TABLE = "legal_chunks"
DEFAULT_EMBED_MODEL = "BAAI/bge-large-zh-v1.5"
DEFAULT_EMBED_DIM = 1024
DEFAULT_DB_URL = "postgresql://postgres:postgres@localhost:5432/legal_db"


def load_nodes_from_json(json_path: Path) -> List[TextNode]:
    """从单个 JSON 文件加载 TextNode 列表。"""
    with open(json_path, "r", encoding="utf-8") as f:
        raw_list = json.load(f)

    nodes: List[TextNode] = []
    for item in raw_list:
        node = TextNode(
            id_=item["id_"],
            text=item["text"],
            metadata=item.get("metadata", {}),
        )
        excluded = [
            "char_count", "token_estimate",
            "split_part", "split_total",
        ]
        node.excluded_embed_metadata_keys = excluded
        node.excluded_llm_metadata_keys = excluded
        nodes.append(node)
    return nodes


def build_relationships(nodes: List[TextNode]) -> None:
    """为同一文档内相邻 TextNode 设置 PREVIOUS / NEXT 关系。"""
    for i in range(len(nodes)):
        if i > 0:
            nodes[i].relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=nodes[i - 1].node_id,
            )
        if i < len(nodes) - 1:
            nodes[i].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=nodes[i + 1].node_id,
            )


def load_all_chunks(chunks_dir: str, verbose: bool = False) -> List[TextNode]:
    """加载目录下所有 JSON 文件并建立文档内节点关联。"""
    chunks_path = Path(chunks_dir)
    json_files = sorted(chunks_path.glob("*.json"))

    if not json_files:
        print(f"未找到 JSON 文件: {chunks_path}", file=sys.stderr)
        sys.exit(1)

    all_nodes: List[TextNode] = []
    for jf in json_files:
        doc_nodes = load_nodes_from_json(jf)
        build_relationships(doc_nodes)
        all_nodes.extend(doc_nodes)
        if verbose:
            print(f"  已加载 {jf.name}: {len(doc_nodes)} 个节点")

    return all_nodes


def create_vector_store(
    db_url: str,
    table_name: str = DEFAULT_TABLE,
    embed_dim: int = DEFAULT_EMBED_DIM,
    hnsw_m: int = 16,
    hnsw_ef_construction: int = 64,
) -> PGVectorStore:
    """创建 PGVectorStore 实例，配置 HNSW 索引参数。"""
    vector_store = PGVectorStore.from_params(
        database=_extract_dbname(db_url),
        host=_extract_host(db_url),
        port=str(_extract_port(db_url)),
        user=_extract_user(db_url),
        password=_extract_password(db_url),
        table_name=table_name,
        embed_dim=embed_dim,
        hnsw_kwargs={
            "hnsw_m": hnsw_m,
            "hnsw_ef_construction": hnsw_ef_construction,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )
    return vector_store


def _parse_db_url(url: str) -> dict:
    """解析 postgresql://user:pass@host:port/dbname 格式 URL。"""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return {
        "user": parsed.username or "postgres",
        "password": parsed.password or "postgres",
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "dbname": parsed.path.lstrip("/") or "legal_db",
    }


def _extract_user(url: str) -> str:
    return _parse_db_url(url)["user"]


def _extract_password(url: str) -> str:
    return _parse_db_url(url)["password"]


def _extract_host(url: str) -> str:
    return _parse_db_url(url)["host"]


def _extract_port(url: str) -> int:
    return _parse_db_url(url)["port"]


def _extract_dbname(url: str) -> str:
    return _parse_db_url(url)["dbname"]


def main():
    parser = argparse.ArgumentParser(
        description="将法律文档 chunks 加载到 PostgreSQL pgvector 向量数据库"
    )
    parser.add_argument(
        "--chunks-dir", default=DEFAULT_CHUNKS_DIR,
        help=f"chunks JSON 目录 (默认: {DEFAULT_CHUNKS_DIR})",
    )
    parser.add_argument(
        "--db-url", default=os.environ.get("PG_URL", DEFAULT_DB_URL),
        help="PostgreSQL 连接 URL",
    )
    parser.add_argument(
        "--table-name", default=DEFAULT_TABLE,
        help=f"向量表名称 (默认: {DEFAULT_TABLE})",
    )
    parser.add_argument(
        "--embed-model", default=DEFAULT_EMBED_MODEL,
        help=f"HuggingFace embedding 模型 (默认: {DEFAULT_EMBED_MODEL})",
    )
    parser.add_argument(
        "--embed-dim", type=int, default=DEFAULT_EMBED_DIM,
        help=f"embedding 维度 (默认: {DEFAULT_EMBED_DIM})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="embedding 批处理大小 (默认: 128)",
    )
    parser.add_argument(
        "--hnsw-m", type=int, default=16,
        help="HNSW 索引 M 参数 (默认: 16)",
    )
    parser.add_argument(
        "--hnsw-ef-construction", type=int, default=64,
        help="HNSW 索引 ef_construction 参数 (默认: 64)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="打印详细日志",
    )
    args = parser.parse_args()

    # 1. 配置 embedding 模型 ==================================================
    print(f"加载 embedding 模型: {args.embed_model} ...")
    embed_model = HuggingFaceEmbedding(
        model_name=args.embed_model,
        embed_batch_size=args.batch_size,
    )
    Settings.embed_model = embed_model

    # 2. 加载所有 chunk 并建立前后关联 ==========================================
    print(f"加载 chunks: {args.chunks_dir} ...")    
    all_nodes = load_all_chunks(args.chunks_dir, verbose=args.verbose)
    print(f"共加载 {len(all_nodes)} 个节点")

    # 3. 创建 PGVectorStore (HNSW 索引) =======================================
    print(f"连接 PostgreSQL: {args.db_url}")
    vector_store = create_vector_store(
        db_url=args.db_url,
        table_name=args.table_name,
        embed_dim=args.embed_dim,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
    )

    # 4. 构建索引并写入数据库 ==================================================
    print("正在生成 embedding 并写入数据库 ...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=all_nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"\n写入完成:")
    print(f"  节点总数: {len(all_nodes)}")
    print(f"  向量表: {args.table_name}")
    print(f"  HNSW M: {args.hnsw_m}, ef_construction: {args.hnsw_ef_construction}")
    print(f"  embedding 模型: {args.embed_model}")
    print(f"  embedding 维度: {args.embed_dim}")


if __name__ == "__main__":
    main()
