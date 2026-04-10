"""
Chunking Strategy Comparison Experiment
----------------------------------------
Chạy: python experiment.py

So sánh 3 chiến lược chunking trên các file trong data/.
Kết quả in ra màn hình — copy vào REPORT.md Section 3 (Baseline Analysis).
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path
from src.chunking import ChunkingStrategyComparator, FixedSizeChunker, SentenceChunker, RecursiveChunker
from src.store import EmbeddingStore
from src.models import Document

DATA_DIR = Path("data")

# ─── Cấu hình experiment ───────────────────────────────────────────────────
# Thay đổi các tham số này để test strategy của bạn
FIXED_SIZE_PARAMS   = dict(chunk_size=500, overlap=100)
SENTENCE_PARAMS     = dict(max_sentences_per_chunk=4)
RECURSIVE_PARAMS    = dict(chunk_size=500)

# Chỉ chạy các file luật (bỏ trống = tất cả .txt/.md trong data/)
TARGET_FILES: list[str] = [
    "134_TTNT.md",
    "133_CNC.md",
    "125_NDKH.md",
    "65_CNTT.md",
    "71_CNCNS.md",
    "91_BVDLCN.md",
]

# ─── Benchmark queries (thay bằng 5 queries nhóm thống nhất) ──────────────
BENCHMARK_QUERIES = [
    "Trách nhiệm của nhà cung cấp hệ thống trí tuệ nhân tạo là gì?",
    "Hệ thống AI có rủi ro cao gồm những lĩnh vực nào?",
    "Quyền của người dùng khi tương tác với hệ thống AI là gì?",
    "Cơ quan nào quản lý nhà nước về trí tuệ nhân tạo tại Việt Nam?",
    "Điều kiện để được cấp phép cung cấp dịch vụ AI là gì?",
]


# ──────────────────────────────────────────────────────────────────────────

def load_files(target: list[str] = None) -> list[tuple[str, str]]:
    """Trả về list (filename, content) từ data/."""
    exts = {".txt", ".md"}
    files = sorted(DATA_DIR.glob("*"))
    results = []
    for f in files:
        if f.suffix not in exts:
            continue
        if target and f.name not in target:
            continue
        results.append((f.name, f.read_text(encoding="utf-8")))
    return results


def print_separator(title: str = ""):
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print("-" * pad + f" {title} " + "-" * pad)
    else:
        print("-" * width)


def run_baseline(docs: list[tuple[str, str]]):
    """Chạy ChunkingStrategyComparator trên tất cả doc, in bảng so sánh."""
    comparator = ChunkingStrategyComparator()

    print_separator("BASELINE: So Sánh 3 Chiến Lược Chunking")
    print(f"{'File':<30} {'Strategy':<14} {'Count':>6} {'Avg Len':>9} {'Min':>6} {'Max':>6}")
    print_separator()

    for filename, content in docs:
        result = comparator.compare(content, chunk_size=FIXED_SIZE_PARAMS["chunk_size"])
        label_map = {
            "fixed_size":  "FixedSize",
            "by_sentences": "Sentence",
            "recursive":   "Recursive",
        }
        for key, label in label_map.items():
            stats = result[key]
            chunks = stats["chunks"]
            lengths = [len(c) for c in chunks]
            min_l = min(lengths) if lengths else 0
            max_l = max(lengths) if lengths else 0
            print(
                f"{filename:<30} {label:<14} {stats['count']:>6} "
                f"{stats['avg_length']:>9.1f} {min_l:>6} {max_l:>6}"
            )
        print_separator()


def run_retrieval_comparison(docs: list[tuple[str, str]]):
    """Xây 3 store (mỗi store dùng 1 chunker), chạy queries, so sánh top-1."""
    print_separator("RETRIEVAL: So Sánh Top-1 Chunk Theo Query")

    strategies = {
        "FixedSize":  FixedSizeChunker(**FIXED_SIZE_PARAMS),
        "Sentence":   SentenceChunker(**SENTENCE_PARAMS),
        "Recursive":  RecursiveChunker(**RECURSIVE_PARAMS),
    }

    # Build một store cho mỗi strategy
    stores: dict[str, EmbeddingStore] = {}
    for name, chunker in strategies.items():
        store = EmbeddingStore(collection_name=name)
        for filename, content in docs:
            chunks = chunker.chunk(content)
            store_docs = [
                Document(
                    id=f"{filename}_{i}",
                    content=chunk,
                    metadata={"source": filename, "strategy": name, "chunk_index": i},
                )
                for i, chunk in enumerate(chunks)
            ]
            store.add_documents(store_docs)
        stores[name] = store
        print(f"[{name}] loaded {store.get_collection_size()} chunks")

    print_separator()

    for q_idx, query in enumerate(BENCHMARK_QUERIES, 1):
        print(f"\nQuery {q_idx}: {query}")
        print(f"  {'Strategy':<12} {'Score':>7}  Top-1 Chunk (80 chars)")
        print(f"  {'-'*12} {'-'*7}  {'-'*40}")
        for name, store in stores.items():
            results = store.search(query, top_k=1)
            if results:
                r = results[0]
                snippet = r["content"].replace("\n", " ")[:80]
                print(f"  {name:<12} {r['score']:>7.4f}  {snippet}")
            else:
                print(f"  {name:<12}   (no results)")


def run_search_with_filter_demo(docs: list[tuple[str, str]]):
    """Demo search_with_filter — lọc theo nguồn file."""
    print_separator("BONUS: search_with_filter theo source")

    store = EmbeddingStore(collection_name="filter_demo")
    for filename, content in docs:
        chunks = RecursiveChunker(**RECURSIVE_PARAMS).chunk(content)
        store.add_documents([
            Document(
                id=f"{filename}_{i}",
                content=chunk,
                metadata={"source": filename},
            )
            for i, chunk in enumerate(chunks)
        ])

    query = BENCHMARK_QUERIES[0]
    print(f"Query: {query}")
    print(f"\nKhông filter (top 3):")
    for r in store.search(query, top_k=3):
        print(f"  [{r['metadata']['source']}] {r['score']:.4f}  {r['content'][:60].replace(chr(10),' ')}")

    if docs:
        first_file = docs[0][0]
        print(f"\nFilter source='{first_file}' (top 3):")
        for r in store.search_with_filter(query, top_k=3, metadata_filter={"source": first_file}):
            print(f"  [{r['metadata']['source']}] {r['score']:.4f}  {r['content'][:60].replace(chr(10),' ')}")


if __name__ == "__main__":
    docs = load_files(TARGET_FILES or None)

    if not docs:
        print("Không tìm thấy file trong data/. Thêm file .txt hoặc .md vào thư mục data/.")
        exit(1)

    print(f"\nLoaded {len(docs)} file(s): {[d[0] for d in docs]}\n")

    run_baseline(docs)
    print()
    run_retrieval_comparison(docs)
    print()
    run_search_with_filter_demo(docs)
    print()
    print_separator("XONG")
    print("Copy kết quả baseline vào REPORT.md → Section 3 (Baseline Analysis)")
    print("Thay BENCHMARK_QUERIES bằng 5 queries nhóm đã thống nhất khi có data thật.")
