"""
Chiến lược cá nhân: FixedSizeChunker
Sinh viên: Nguyen Tuan Khai — 2A202600231
Chạy: python -X utf8 my_strategy.py
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import os
from dotenv import load_dotenv
from pathlib import Path
from src.chunking import FixedSizeChunker, ChunkingStrategyComparator
from src.store import EmbeddingStore
from src.models import Document
from src.embeddings import _mock_embed, LocalEmbedder, OpenAIEmbedder

load_dotenv(override=False)


# ─── OpenRouter embedder ──────────────────────────────────────────────────────
class OpenRouterEmbedder:
    """Embedding qua OpenRouter API (tương thích OpenAI SDK)."""

    def __init__(self) -> None:
        from openai import OpenAI
        self._backend_name = os.getenv("OPENROUTER_EMBEDDING_MODEL", "mistral/mistral-embed")
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )

    def __call__(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(model=self._backend_name, input=text)
        return [float(v) for v in resp.data[0].embedding]


# ─── OpenRouter LLM function cho agent.answer ─────────────────────────────────
def openrouter_llm(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    model = os.getenv("OPENROUTER_LLM_MODEL", "google/gemini-2.0-flash-001")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return resp.choices[0].message.content or ""


def mock_llm(prompt: str) -> str:
    return "[mock LLM] " + prompt[:200].replace("\n", " ") + "..."


# ─── Chọn embedder và LLM theo .env ──────────────────────────────────────────
def _get_embedder():
    provider = os.getenv("EMBEDDING_PROVIDER", "mock").strip().lower()
    if provider == "openrouter":
        try:
            emb = OpenRouterEmbedder()
            print(f"[embedder] OpenRouter / {emb._backend_name}")
            return emb
        except Exception as e:
            print(f"[embedder] OpenRouterEmbedder failed ({e}), fallback to mock")
    elif provider == "local":
        try:
            emb = LocalEmbedder()
            print(f"[embedder] {emb._backend_name}")
            return emb
        except Exception as e:
            print(f"[embedder] LocalEmbedder failed ({e}), fallback to mock")
    elif provider == "openai":
        try:
            emb = OpenAIEmbedder()
            print(f"[embedder] {emb._backend_name}")
            return emb
        except Exception as e:
            print(f"[embedder] OpenAIEmbedder failed ({e}), fallback to mock")
    print("[embedder] mock (64-dim, khong hieu ngu nghia)")
    return _mock_embed


def _get_llm():
    provider = os.getenv("EMBEDDING_PROVIDER", "mock").strip().lower()
    if provider in ("openrouter", "openai"):
        print(f"[llm] OpenRouter / {os.getenv('OPENROUTER_LLM_MODEL', 'gemini-2.0-flash')}")
        return openrouter_llm
    print("[llm] mock")
    return mock_llm


EMBEDDER = _get_embedder()
LLM_FN   = _get_llm()

# ─── Strategy cá nhân ────────────────────────────────────────────────────────
# Lý do chọn tham số:
#   - Mỗi Điều luật trung bình 800-1400 ký tự → chunk_size=600 bắt được
#     phần lớn nội dung 1 khoản mà không quá lớn cho embedding
#   - overlap=150 (~25%) giúp không mất ngữ cảnh khi Điều bị cắt ngang
#   - Tham số này cho phép tìm kiếm theo khoản con (mục 1., 2., 3.) của Điều

CHUNK_SIZE = 600
OVERLAP    = 150

MY_CHUNKER = FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)

# Metadata schema — thay đổi theo tài liệu thật của nhóm
FILE_METADATA = {
    "134_TTNT.md":   {"ten_luat": "Luật Trí tuệ nhân tạo",           "so_hieu": "134/2025/QH15", "nam": 2025, "loai": "luat"},
    "133_CNC.md":    {"ten_luat": "Luật Công nghệ cao",               "so_hieu": "133/2025/QH15", "nam": 2025, "loai": "luat"},
    "125_NDKH.md":   {"ten_luat": "Luật Khoa học và Công nghệ",       "so_hieu": "125/2025/QH15", "nam": 2025, "loai": "luat"},
    "65_CNTT.md":    {"ten_luat": "Luật Công nghệ thông tin",         "so_hieu": "65/2006/QH11",  "nam": 2006, "loai": "luat"},
    "71_CNCNS.md":   {"ten_luat": "Luật Công nghệ số",                "so_hieu": "71/2024/QH15",  "nam": 2024, "loai": "luat"},
    "91_BVDLCN.md":  {"ten_luat": "Luật Bảo vệ dữ liệu cá nhân",     "so_hieu": "91/2024/QH15",  "nam": 2024, "loai": "luat"},
}

# ─── 5 Benchmark queries nhóm thống nhất ─────────────────────────────────────
# TODO: thay bằng 5 queries thật của nhóm
BENCHMARK_QUERIES = [
    "Việc xử lý dữ liệu cá nhân của trẻ em từ đủ 07 tuổi trở lên cần lưu ý gì?",
    "Tỷ lệ trích tối thiểu từ nguồn thu học phí của các cơ sở giáo dục đại học để phục vụ hoạt động khoa học, công nghệ và đổi mới sáng tạo được quy định như thế nào?",
    "Công ty tôi đang có kế hoạch đầu tư một nhà máy sản xuất sản phẩm nằm trong Danh mục sản phẩm công nghệ chiến lược. Theo quy định mới nhất, dự án này của chúng tôi sẽ được hưởng những ưu đãi đặc biệt gì về đầu tư?",
    "Doanh nghiệp khởi nghiệp sáng tạo trong lĩnh vực trí tuệ nhân tạo được Nhà nước hỗ trợ những gì?",
    "Khi tiến hành hoạt động phát thanh và truyền hình trên môi trường mạng, các tổ chức, cá nhân bắt buộc phải tuân thủ những quy định của các loại pháp luật nào?",
    "Một công ty công nghệ nước ngoài cung cấp dịch vụ quản lý dữ liệu số phục vụ riêng cho hoạt động cơ yếu để bảo vệ bí mật nhà nước tại Việt Nam. Công ty này có thuộc đối tượng áp dụng và phạm vi điều chỉnh của Luật này không?"
]


# ─── Load data ────────────────────────────────────────────────────────────────
def load_store() -> EmbeddingStore:
    store = EmbeddingStore(collection_name="fixedsize_personal", embedding_fn=EMBEDDER)
    data_dir = Path("data")

    for fname, meta in FILE_METADATA.items():
        path = data_dir / fname
        if not path.exists():
            print(f"  [skip] {fname} not found")
            continue
        text = path.read_text(encoding="utf-8")
        chunks = MY_CHUNKER.chunk(text)
        docs = [
            Document(
                id=f"{fname}_{i}",
                content=chunk,
                metadata={**meta, "source": fname, "chunk_index": i},
            )
            for i, chunk in enumerate(chunks)
        ]
        store.add_documents(docs)
        print(f"  [ok] {fname}: {len(chunks)} chunks")

    return store


# ─── Baseline comparison ──────────────────────────────────────────────────────
def run_baseline():
    print("\n" + "=" * 70)
    print("BASELINE: So sanh 3 strategy (chunk_size=600)")
    print("=" * 70)
    print(f"{'File':<22} {'Strategy':<12} {'Count':>6} {'Avg':>7} {'Min':>6} {'Max':>6}")
    print("-" * 70)

    comparator = ChunkingStrategyComparator()
    for fname in FILE_METADATA:
        path = Path("data") / fname
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        result = comparator.compare(text, chunk_size=CHUNK_SIZE)
        label_map = {"fixed_size": "FixedSize", "by_sentences": "Sentence", "recursive": "Recursive"}
        for key, label in label_map.items():
            s = result[key]
            lengths = [len(c) for c in s["chunks"]]
            mn, mx = (min(lengths), max(lengths)) if lengths else (0, 0)
            print(f"{fname:<22} {label:<12} {s['count']:>6} {s['avg_length']:>7.0f} {mn:>6} {mx:>6}")
        print("-" * 70)


# ─── Benchmark ───────────────────────────────────────────────────────────────
def run_benchmark(store: EmbeddingStore):
    print("\n" + "=" * 70)
    print(f"BENCHMARK: FixedSize (size={CHUNK_SIZE}, overlap={OVERLAP})")
    print(f"Store: {store.get_collection_size()} chunks tong cong")
    print("=" * 70)

    from src.agent import KnowledgeBaseAgent
    agent = KnowledgeBaseAgent(store=store, llm_fn=LLM_FN)

    rows = []
    for i, query in enumerate(BENCHMARK_QUERIES, 1):
        results = store.search(query, top_k=3)

        print(f"\nQuery {i}: {query}")
        print(f"  {'#':<3} {'Score':>7}  {'Source':<18}  Top chunk (100 chars)")
        print(f"  {'-'*3} {'-'*7}  {'-'*18}  {'-'*40}")
        for rank, r in enumerate(results, 1):
            snippet = r["content"].replace("\n", " ")[:100]
            src = r["metadata"].get("source", "?")
            print(f"  {rank:<3} {r['score']:>7.4f}  {src:<18}  {snippet}")

        print(f"\n  [Agent answer]")
        try:
            answer = agent.answer(query, top_k=3)
            for line in answer.strip().splitlines():
                print(f"  {line}")
        except Exception as e:
            answer = f"(loi: {e})"
            print(f"  {answer}")

        top1 = results[0] if results else {}
        rows.append({
            "query": query,
            "top1_score": top1.get("score", 0),
            "top1_source": top1.get("metadata", {}).get("source", "?"),
            "agent_answer": answer[:120].replace("\n", " "),
            "relevant": "?",
        })

    # In bảng tóm tắt để copy vào REPORT.md
    print("\n" + "=" * 70)
    print("BANG TOM TAT (copy vao REPORT.md Section 6)")
    print("=" * 70)
    print(f"| # | Query (tóm tắt) | Top-1 Source | Score | Agent Answer (tóm tắt) | Relevant? |")
    print(f"|---|-----------------|--------------|-------|------------------------|-----------|")
    for i, r in enumerate(rows, 1):
        q_short = r["query"][:35] + "..." if len(r["query"]) > 35 else r["query"]
        a_short = r["agent_answer"][:60] + "..." if len(r["agent_answer"]) > 60 else r["agent_answer"]
        print(f"| {i} | {q_short} | {r['top1_source']} | {r['top1_score']:.4f} | {a_short} | {r['relevant']} |")

    return rows


# ─── Filter demo ──────────────────────────────────────────────────────────────
def run_filter_demo(store: EmbeddingStore):
    print("\n" + "=" * 70)
    print("FILTER DEMO: search_with_filter theo loai + nam")
    print("=" * 70)

    query = BENCHMARK_QUERIES[0]
    print(f"Query: {query}\n")

    print("Khong filter (top 3):")
    for r in store.search(query, top_k=3):
        src = r["metadata"].get("source", "?")
        nam = r["metadata"].get("nam", "?")
        print(f"  {r['score']:.4f}  [{src} | {nam}]  {r['content'][:70].replace(chr(10),' ')}")

    print("\nFilter: loai='luat', nam >= 2024 (chi luat moi nhat):")
    recent = [r for r in store.search(query, top_k=50)
              if r["metadata"].get("loai") == "luat" and r["metadata"].get("nam", 0) >= 2024]
    for r in recent[:3]:
        src = r["metadata"].get("source", "?")
        nam = r["metadata"].get("nam", "?")
        print(f"  {r['score']:.4f}  [{src} | {nam}]  {r['content'][:70].replace(chr(10),' ')}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("CHIEN LUOC CA NHAN: FixedSizeChunker")
    print(f"Tham so: chunk_size={CHUNK_SIZE}, overlap={OVERLAP}")
    print("Ly do: Dieu luat trung binh 800-1400 ky tu; overlap 25% giu ngu canh")
    print("=" * 70)

    run_baseline()

    print("\nLoading store...")
    store = load_store()

    run_benchmark(store)
    run_filter_demo(store)

    print("\nDone. Dien ket qua vao REPORT.md:")
    print("  Section 3 -> copy bang Baseline")
    print("  Section 6 -> copy bang Benchmark")
