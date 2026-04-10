# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Tuấn Khải
**MSSV:** 2A202600231
**Nhóm:** Nhóm Luật pháp Việt Nam (2025-2026)
**Ngày:** 10/04/2026

---

# PHẦN CÁ NHÂN (60 điểm)

---

## 1. Warm-up — Cá nhân (5 điểm)

### Ex 1.1 — Cosine Similarity

**High cosine similarity nghĩa là gì?**

> Cosine similarity cao (gần 1.0) có nghĩa là hai vector embedding chỉ về cùng một hướng trong không gian nhiều chiều — tức hai đoạn văn bản có ngữ nghĩa tương đồng dù có thể dùng từ ngữ khác nhau. Giá trị này đo góc giữa hai vector, không bị ảnh hưởng bởi độ dài văn bản.

**Ví dụ HIGH similarity (từ domain luật của nhóm):**

- Sentence A: *"Doanh nghiệp khởi nghiệp sáng tạo trong lĩnh vực AI được ưu tiên hỗ trợ từ Quỹ Phát triển trí tuệ nhân tạo quốc gia."*
- Sentence B: *"Startup AI được Nhà nước cấp kinh phí ưu tiên từ quỹ phát triển trí tuệ nhân tạo."*
- Lý do tương đồng: Cùng chủ thể (startup/doanh nghiệp AI), cùng hành động (nhận hỗ trợ tài chính từ quỹ nhà nước), chỉ khác cách diễn đạt bề mặt.

**Ví dụ LOW similarity:**

- Sentence A: *"Cơ sở giáo dục đại học trích tối thiểu 8% học phí để phục vụ hoạt động khoa học công nghệ."*
- Sentence B: *"Người đại diện theo pháp luật phải đồng ý khi xử lý dữ liệu cá nhân của trẻ em."*
- Lý do khác: Hai câu hoàn toàn khác chủ đề — tài chính giáo dục đại học vs. bảo vệ dữ liệu trẻ em.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**

> Cosine similarity chỉ quan tâm đến hướng của vector, không bị ảnh hưởng bởi magnitude. Hai văn bản có cùng nội dung nhưng khác độ dài sẽ có embedding vector có magnitude khác nhau — Euclidean distance sẽ lớn dù nội dung giống nhau, trong khi cosine similarity vẫn gần 1.0. Với text, ý nghĩa nằm ở hướng của vector, không phải độ lớn.

---

### Ex 1.2 — Chunking Math

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

> - Mỗi bước tiến: `step = 500 - 50 = 450` ký tự
> - Số chunk: `ceil((10,000 - 50) / 450) = ceil(9,950 / 450) = ceil(22.11) = **23 chunks**`

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**

> Khi overlap = 100: `step = 400`, số chunk = `ceil(9,900 / 400) = ceil(24.75) = **25 chunks**` — tăng thêm 2 chunks. Overlap lớn hơn giúp mỗi chunk lặp lại phần đuôi của chunk trước, tránh mất thông tin tại ranh giới — đặc biệt quan trọng với văn bản pháp luật khi một điều khoản quan trọng có thể nằm đúng chỗ bị cắt.

---

## 2. Core Implementation — Cá nhân (30 điểm)

### Kết Quả Tests

```
pytest tests/ -v
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-9.0.2, pluggy-1.6.0
rootdir: E:\a\codeLabAI\2A202600231_NguyenTuanKhai_Day07
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.19s ==============================
```

**Kết quả: 42 / 42 tests PASSED**

---

## 3. My Approach — Cá nhân (10 điểm)

### `SentenceChunker.chunk`

> Dùng `re.split(r'(?<=[\.\!\?])\s+|(?<=\.)\n', text)` để tách câu tại các dấu kết thúc câu (lookbehind), tránh split nhầm tại dấu chấm trong số thứ tự kiểu "Điều 1." hay "khoản 2.". Sau đó strip và lọc câu rỗng. Gộp mỗi `max_sentences_per_chunk` câu liên tiếp bằng khoảng trắng thành một chunk. Text rỗng trả về list rỗng.

### `RecursiveChunker.chunk` / `_split`

> `chunk` gọi `_split` với toàn bộ danh sách separators `["\n\n", "\n", ". ", " ", ""]`. `_split` là hàm đệ quy: nếu `len(text) <= chunk_size` thì giữ nguyên (base case); nếu không thì split bằng separator đầu danh sách, ghép lại từng phần nhỏ theo `chunk_size`, phần nào vẫn quá lớn thì đệ quy với phần còn lại của danh sách separator. Khi hết separator thì force-split theo ký tự.

### `compute_similarity`

> Áp dụng công thức cosine: `dot(a, b) / (||a|| * ||b||)`. Tính `dot` và `magnitude` bằng vòng lặp Python thuần không dùng numpy. Guard case: nếu bất kỳ vector nào có magnitude = 0.0 thì trả về 0.0 ngay để tránh ZeroDivisionError.

### `EmbeddingStore.add_documents` + `search`

> `_make_record` nhận một `Document`, gọi `self._embedding_fn(doc.content)` để embed, tạo dict `{"content": ..., "embedding": ..., "metadata": {..., "doc_id": doc.id}}` — `doc_id` được gắn vào metadata để dùng cho `delete_document`. `add_documents` lặp qua list và append từng record vào `self._store`. `search` embed query rồi gọi `_search_records` tính dot product với toàn bộ store, sort descending, trả top-k với keys `content`, `score`, `metadata`.

### `EmbeddingStore.search_with_filter` + `delete_document`

> `search_with_filter` lọc `self._store` bằng list comprehension **trước** — giữ lại record có metadata khớp tất cả key-value trong `metadata_filter` — rồi mới chạy `_search_records` trên tập đã lọc. Filter trước để tránh tính embedding cho record không cần thiết. `delete_document` xây list mới loại bỏ mọi record có `metadata["doc_id"] == doc_id`, trả `True` nếu danh sách ngắn hơn trước, `False` nếu không đổi.

### `KnowledgeBaseAgent.answer`

> `__init__` lưu `self.store` và `self.llm_fn`. `answer` gọi `self.store.search(question, top_k=top_k)` lấy top-k chunks, ghép thành `context` bằng `"\n\n".join(r["content"])`, build prompt dạng `"Context:\n{context}\n\nQuestion: {question}\nAnswer:"` rồi truyền vào `self.llm_fn(prompt)` và trả về kết quả. Prompt đơn giản nhưng đủ để LLM biết phải dựa vào context được cung cấp.

---

## 4. Similarity Predictions — Cá nhân (5 điểm)

Chạy `compute_similarity` với `text-embedding-3-small` (qua OpenRouter) trên 5 cặp câu từ domain luật của nhóm. Dự đoán trước khi chạy:

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Doanh nghiệp khởi nghiệp sáng tạo được hỗ trợ chi phí đánh giá sự phù hợp." | "Startup được nhà nước cung cấp miễn phí hồ sơ mẫu và công cụ tự đánh giá." | high | **0.5421** | ❌ (medium) |
| 2 | "Tỷ lệ trích tối thiểu từ nguồn thu học phí là 8% đối với đại học." | "Mức phạt tối đa đối với tổ chức vi phạm bảo vệ dữ liệu cá nhân." | low | **0.2995** | ✅ |
| 3 | "Việc xử lý dữ liệu cá nhân của trẻ em phải có sự đồng ý của người đại diện." | "Trẻ em từ đủ 07 tuổi cần có sự chấp thuận của cha mẹ khi chia sẻ thông tin cá nhân." | high | **0.6424** | ✅ |
| 4 | "Luật này không điều chỉnh hoạt động cơ yếu để bảo vệ bí mật nhà nước." | "Các quy định về an toàn thông tin mạng áp dụng cho mọi tổ chức tại Việt Nam." | low | **0.4501** | ❌ (medium) |
| 5 | "Cơ sở giáo dục đại học trích 8% học phí cho hoạt động khoa học công nghệ." | "Tỷ lệ trích lập quỹ KH&CN từ doanh thu của cơ sở giáo dục là 8 phần trăm." | high | **0.5861** | ✅ |

**Dự đoán đúng: 3 / 5**

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**

> **Pair 1** (dự đoán high → 0.54): Cả hai câu nói về "hỗ trợ startup", nhưng mô tả hai loại hỗ trợ khác nhau ("chi phí đánh giá" vs "hồ sơ mẫu/công cụ"), embedding phân biệt được sự khác nhau đó nên chỉ đạt mức trung bình. **Pair 4** (dự đoán low → 0.45): Hai câu mang nghĩa đối lập (loại trừ vs. bao quát) nhưng cùng domain "pháp luật thông tin tại Việt Nam" — embedding bắt được context chung của domain, làm score cao hơn kỳ vọng. Bài học: `text-embedding-3-small` với tiếng Việt học theo chủ đề/domain, không chỉ so sánh nghĩa literal — cần đặt threshold retrieval thấp hơn (~0.5) thay vì ~0.7 khi làm việc với corpus pháp luật tiếng Việt.

---

## 5. Competition Results — Cá nhân (10 điểm)

Chạy 6 benchmark queries của nhóm với **FixedSizeChunker(chunk_size=600, overlap=150)** + `text-embedding-3-small` (qua OpenRouter) + Gemini 2.0 Flash làm LLM. Tổng 826 chunks từ 6 file luật.

> **Câu hỏi tôi phụ trách (Q4):** *"Doanh nghiệp khởi nghiệp sáng tạo trong lĩnh vực trí tuệ nhân tạo được Nhà nước hỗ trợ những gì?"*

### Benchmark Queries & Gold Answers

| # | Người phụ trách | Query | Gold Answer |
|---|----------------|-------|-------------|
| 1 | Nguyễn Quốc Khánh | Việc xử lý dữ liệu cá nhân của trẻ em từ đủ 07 tuổi trở lên cần lưu ý gì? | Việc xử lý dữ liệu cá nhân của trẻ em nhằm công bố, tiết lộ thông tin về đời sống riêng tư, bí mật cá nhân của trẻ em từ đủ 07 tuổi trở lên thì phải có sự đồng ý của trẻ em và người đại diện theo pháp luật. |
| 2 | Lê Huy Hồng Nhật | Tỷ lệ trích tối thiểu từ nguồn thu học phí của các cơ sở giáo dục đại học để phục vụ hoạt động khoa học, công nghệ và đổi mới sáng tạo được quy định như thế nào? | Cơ sở giáo dục đại học trích tối thiểu 8% đối với đại học và 5% đối với cơ sở giáo dục đại học khác. |
| 3 | Nguyễn Quế Sơn | Công ty tôi đang có kế hoạch đầu tư một nhà máy sản xuất sản phẩm nằm trong Danh mục sản phẩm công nghệ chiến lược. Theo quy định mới nhất, dự án này sẽ được hưởng những ưu đãi đặc biệt gì về đầu tư? | Được hưởng chính sách ưu đãi, hỗ trợ đầu tư đặc biệt theo pháp luật về đầu tư; ưu đãi cao nhất về thuế, đất đai và các chính sách liên quan. (điểm a khoản 3 Điều 16) |
| **4** | **Nguyễn Tuấn Khải** | **Doanh nghiệp khởi nghiệp sáng tạo trong lĩnh vực trí tuệ nhân tạo được Nhà nước hỗ trợ những gì?** | Được hỗ trợ chi phí đánh giá sự phù hợp và cung cấp miễn phí hồ sơ mẫu, công cụ tự đánh giá, đào tạo và tư vấn. Được ưu tiên hỗ trợ từ Quỹ Phát triển trí tuệ nhân tạo quốc gia. Được hỗ trợ phiếu sử dụng hạ tầng tính toán, dữ liệu dùng chung, mô hình ngôn ngữ lớn. Được hỗ trợ khi tham gia thử nghiệm AI. |
| 5 | Phan Văn Tấn | Khi tiến hành hoạt động phát thanh và truyền hình trên môi trường mạng, các tổ chức, cá nhân bắt buộc phải tuân thủ những quy định của các loại pháp luật nào? | Pháp luật về viễn thông; pháp luật về báo chí; các quy định của Luật Công nghệ thông tin. (Khoản 3, Điều 13) |
| 6 | Lê Công Thành | Một công ty công nghệ nước ngoài cung cấp dịch vụ quản lý dữ liệu số phục vụ riêng cho hoạt động cơ yếu để bảo vệ bí mật nhà nước tại Việt Nam. Công ty này có thuộc phạm vi điều chỉnh của Luật này không? | Không thuộc phạm vi điều chỉnh vì Luật không điều chỉnh hoạt động công nghiệp công nghệ số chỉ phục vụ mục đích cơ yếu bảo vệ bí mật nhà nước. |

### Kết Quả Chạy (FixedSizeChunker 600/150 + text-embedding-3-small)

| # | Query (tóm tắt) | Top-1 Source | Score | Relevant? | Agent Answer khớp Gold? | Điểm |
|---|----------------|--------------|-------|-----------|------------------------|------|
| 1 | Xử lý dữ liệu trẻ em từ 07 tuổi | 91_BVDLCN.md | 0.7352 | ✅ | ✅ Đồng ý của trẻ em + người đại diện | **2/2** |
| 2 | Tỷ lệ trích học phí KH&CN | 125_NDKH.md | 0.6496 | ✅ | ⚠️ Trả lời "8% và 5%" nhưng thiếu phân biệt rõ "đại học" vs "cơ sở GD ĐH khác" | **1/2** |
| 3 | Ưu đãi dự án công nghệ chiến lược | 71_CNCNS.md | 0.6078 | ✅ | ✅ Ưu đãi đầu tư đặc biệt, thuế, đất đai | **2/2** |
| **4** | **Hỗ trợ doanh nghiệp khởi nghiệp AI** | **134_TTNT.md** | **0.6987** | ✅ | ✅ **Chi phí đánh giá, hồ sơ mẫu, Quỹ TTNT, hạ tầng tính toán** | **2/2** |
| 5 | Phát thanh truyền hình — tuân thủ pháp luật nào? | 65_CNTT.md | 0.5642 | ✅ | ⚠️ Đúng "viễn thông, báo chí, CNTT" nhưng chunk bị cắt không có số điều khoản | **1/2** |
| 6 | Công ty nước ngoài dịch vụ cơ yếu | 71_CNCNS.md | 0.6535 | ✅ | ✅ Không thuộc phạm vi — loại trừ hoạt động cơ yếu | **2/2** |

**Tổng: 10 / 12 | Chunk relevant trong top-3: 6 / 6**

> **Phân tích Q2:** Chunk chứa thông tin tỷ lệ 8%/5% bị cắt giữa, phần giải thích "đối với đại học" vs "cơ sở GD ĐH khác" nằm ở 2 chunk khác nhau. Overlap 150 ký tự chưa đủ để giữ cả hai phần trong cùng một chunk → agent trả lời thiếu chi tiết.
>
> **Phân tích Q5:** Câu khoản về phát thanh/truyền hình (Khoản 3, Điều 13) bị gộp chung với các khoản về thương mại điện tử trong cùng chunk — agent lấy được nội dung đúng nhưng không trích được số điều khoản cụ thể do chunk không bao đầy đủ header.

---

# PHẦN NHÓM (40 điểm)

---

## 6. Document Set Quality — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Luật pháp Việt Nam — văn bản pháp luật về công nghệ số, CNTT, trí tuệ nhân tạo, bảo vệ dữ liệu cá nhân (ban hành 2025-2026)

**Tại sao nhóm chọn domain này?**

> Các văn bản pháp luật công nghệ số Việt Nam mới ban hành (2025-2026) chưa được indexing tốt trên các công cụ tìm kiếm thông thường, nên RAG mang lại giá trị thực tế cao. Domain này có đáp án rõ ràng, có thể đối chiếu trực tiếp với điều khoản luật — phù hợp để đánh giá retrieval precision một cách khách quan. Cấu trúc phân cấp rõ ràng (Chương → Điều → Khoản) cũng tạo điều kiện so sánh các chunking strategy một cách công bằng.

### Data Inventory (6 tài liệu)

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|-------------|-------|----------|-----------------|
| 1 | 65_CNTT.md (Luật Công nghệ thông tin hợp nhất) | Cổng thông tin pháp điển | 67,598 | category: luat, year: 2006 |
| 2 | 71_CNCNS.md (Luật Công nghiệp công nghệ số) | Quochoi.vn | ~55,000 | category: luat, year: 2025 |
| 3 | 91_BVDLCN.md (Luật Bảo vệ dữ liệu cá nhân) | Quochoi.vn | 52,907 | category: luat, year: 2025 |
| 4 | 125_NDKH.md (NĐ 125/2026 về KH&CN đại học) | Chinhphu.vn | ~42,000 | category: nghi_dinh, year: 2026 |
| 5 | 133_CNC.md (Luật Công nghệ cao sửa đổi) | Quochoi.vn | ~38,000 | category: luat, year: 2025 |
| 6 | 134_TTNT.md (Luật Trí tuệ nhân tạo) | Quochoi.vn | 49,054 | category: luat, year: 2025 |

### Metadata Schema

| Trường | Kiểu | Ví dụ | Tại sao hữu ích? |
|--------|------|-------|-----------------|
| `ten_luat` | string | `"Luật Trí tuệ nhân tạo"` | Hiển thị tên đầy đủ trong kết quả |
| `so_hieu` | string | `"134/2025/QH15"` | Trích dẫn chính xác trong agent answer |
| `nam` | int | `2025`, `2026` | Filter ưu tiên văn bản mới, loại văn bản hết hiệu lực |
| `loai` | string | `"luat"`, `"nghi_dinh"` | Filter theo loại văn bản khi cần độ chính xác pháp lý |
| `source` | string | `"91_BVDLCN.md"` | Trace về file gốc để kiểm chứng |

---

## 7. Strategy Design — Nhóm (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare(chunk_size=600)` trên 3 tài liệu chính:

| Tài liệu | Strategy | Count | Avg Length | Min | Max | Nhận xét |
|----------|----------|-------|------------|-----|-----|----------|
| 134_TTNT.md | FixedSize | 90 | 597 | 363 | 600 | Rất đều, nhưng ~98% cắt giữa khoản |
| 134_TTNT.md | Sentence | 121 | 406 | 33 | 1663 | Giữ câu nguyên, size không đều |
| 134_TTNT.md | Recursive | 138 | 357 | 34 | 600 | Cân bằng nhất |
| 91_BVDLCN.md | FixedSize | 97 | 599 | 538 | 600 | Rất đều, nhưng ~98% cắt giữa khoản |
| 91_BVDLCN.md | Sentence | 132 | 403 | 32 | 1517 | Giữ câu nguyên, size không đều |
| 91_BVDLCN.md | Recursive | 155 | 344 | 3 | 600 | Cân bằng nhất |
| 65_CNTT.md | FixedSize | 124 | 599 | 487 | 600 | Rất đều, nhưng ~98% cắt giữa khoản |
| 65_CNTT.md | Sentence | 205 | 331 | 47 | 1519 | Giữ câu nguyên, size không đều |
| 65_CNTT.md | Recursive | 177 | 385 | 3 | 598 | Cân bằng nhất |

### Strategy Của Tôi

**Loại:** `FixedSizeChunker(chunk_size=600, overlap=150)`

**Mô tả cách hoạt động:**

> Cắt văn bản thành các đoạn 600 ký tự, mỗi chunk kế tiếp lặp lại 150 ký tự cuối của chunk trước (overlap ~25%). Đây là strategy đơn giản nhất: bước tiến = 600 - 150 = 450 ký tự. Kết quả là 826 chunks từ 6 file luật, mỗi chunk có size rất đều (min=125, max=600 ký tự).

**Tại sao chọn tham số này cho domain luật?**

> Phân tích baseline cho thấy mỗi Điều luật trung bình dài 800-1400 ký tự. chunk_size=600 bắt được phần lớn nội dung của 1 khoản con (mục 1., 2., 3.) mà không quá lớn để embedding bị pha loãng. overlap=150 (~25%) giúp bảo tồn ngữ cảnh tại ranh giới — khi một quy định quan trọng bị cắt ngang, phần đầu của nó sẽ xuất hiện lại ở chunk tiếp theo.

**Code snippet:**

```python
MY_CHUNKER = FixedSizeChunker(chunk_size=600, overlap=150)

# Với metadata schema đầy đủ
for fname, meta in FILE_METADATA.items():
    text = Path("data/" + fname).read_text(encoding="utf-8")
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
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Count | Avg Length | Retrieval Score |
|----------|----------|-------|------------|-----------------|
| 134_TTNT.md | Recursive (best baseline) | 138 | 357 | 7/10 |
| 134_TTNT.md | **FixedSize 600/150 (của tôi)** | **110** | **597** | **8/10** |
| 91_BVDLCN.md | Recursive (best baseline) | 155 | 344 | 7/10 |
| 91_BVDLCN.md | **FixedSize 600/150 (của tôi)** | **119** | **599** | **8/10** |

> FixedSize với overlap lớn (150) vượt Recursive baseline vì overlap bù đắp được phần cắt, trong khi Recursive tạo nhiều chunk quá nhỏ (min=3) làm loãng signal embedding.

### So Sánh Với Toàn Nhóm

| Thành viên | Strategy | Retrieval Score | Điểm mạnh | Điểm yếu |
|-----------|----------|-----------------|-----------|----------|
| **Nguyễn Tuấn Khải (tôi)** | FixedSize(600, overlap=150) + text-embedding-3-small | **7/10** | Đơn giản, ổn định, dễ tune | Cắt giữa khoản, thiếu số điều khoản |
| Lê Huy Hồng Nhật | SentenceChunker(max=2) + text-embedding-3-small | 9/10 | Chunk ngắn, embedding tập trung | Quá ngắn, thiếu ngữ cảnh đa khoản |
| Nguyễn Quốc Khánh | LawRecursiveChunker(1000) + Gemini Embeddings | 9/10 | Separator domain-specific theo cấu trúc luật | Phụ thuộc format văn bản Việt Nam |
| Nguyễn Quế Sơn | SentenceChunker(max=6) + text-embedding-3-large | 8/10 | Chunk dài, đủ ngữ cảnh | Pha loãng semantic signal |
| Phan Văn Tấn | FixedSize(500, overlap=100) + all-MiniLM-L6-v2 | 6/10 | Model local, không cần API | Model nhỏ, similarity thấp hơn |
| Lê Công Thành | FixedSize(1000) + all-MiniLM-L6-v2 | 7/10 | Chunk dài, bao quát tốt | 2265 chunks, tốc độ chậm |

**Strategy nào tốt nhất cho domain này? Tại sao?**

> `LawRecursiveChunker` (Nguyễn Quốc Khánh) và `SentenceChunker(max=2)` (Lê Huy Hồng Nhật) đều đạt 9/10. Yếu tố quyết định không phải thuật toán phức tạp mà là **domain fit**: separator tùy chỉnh theo cấu trúc pháp lý (Điều, Khoản) và embedder mạnh tiếng Việt. FixedSize của tôi đạt 7/10 — tốt cho production vì đơn giản, nhưng kém hơn khi domain có cấu trúc rõ ràng cần khai thác.

---

## 8. Retrieval Quality — Nhóm (10 điểm)

So sánh precision của từng thành viên trên 6 benchmark queries:

| Thành viên | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Tổng |
|-----------|----|----|----|----|----|----|------|
| Lê Huy Hồng Nhật (SentenceChunker max=2) | 2 | 2 | 1 | 2 | 2 | 2 | **11/12** |
| Nguyễn Quốc Khánh (LawRecursiveChunker) | 2 | 2 | 2 | 2 | 2 | 2 | **12/12** |
| **Nguyễn Tuấn Khải (FixedSize 600)** | **2** | **1** | **2** | **2** | **1** | **2** | **10/12** |
| Nguyễn Quế Sơn (SentenceChunker max=6) | 2 | 2 | 2 | 2 | 2 | 2 | **12/12** |
| Phan Văn Tấn (FixedSize 500) | 2 | 1 | 1 | 2 | 2 | 1 | **9/12** |
| Lê Công Thành (FixedSize 1000) | 2 | 2 | 1 | 2 | 1 | 2 | **10/12** |

*Thang điểm: 2 = top-3 relevant + answer đúng; 1 = relevant nhưng answer thiếu chi tiết; 0 = không relevant*

**Observation:** Q5 (phát thanh/truyền hình) và Q2 (tỷ lệ học phí) là hai query phân hóa nhất đối với strategy FixedSize. Q5 thất bại vì câu khoản cần tìm bị gộp với khoản khác trong cùng chunk — `SentenceChunker(max=2)` và `LawRecursiveChunker` xử lý tốt hơn nhờ chunk nhỏ và separator phù hợp. Q2 thất bại vì con số 8%/5% nằm ở hai phần của câu bị cắt đôi; `RecursiveChunker` tránh được vì ưu tiên split tại `\n` trước.

---

## 9. Demo & What I Learned — Nhóm (5 điểm)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**

> Nguyễn Quốc Khánh thiết kế `LawRecursiveChunker` với separators tùy chỉnh theo cấu trúc luật Việt Nam (`\n#### Điều`, `\n\n`, `\n`). Cách tiếp cận domain-specific này đạt 12/12 và cho thấy: hiểu cấu trúc dữ liệu của mình quan trọng hơn chọn thuật toán phức tạp. Tôi đã bỏ qua cấu trúc Điều/Khoản của Markdown khi thiết kế FixedSize — đây là điểm yếu lớn nhất của strategy.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**

> Một nhóm khác dùng `search_with_filter` để lọc theo `year >= 2024` trước mỗi query — tăng precision vì loại bỏ văn bản cũ hết hiệu lực. Cách dùng metadata filter như "pre-filtering layer" trước similarity search là insight tôi chưa khai thác triệt để trong strategy của mình.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**

> (1) Thêm metadata `dieu_so` và `khoan_so` bằng cách parse heading `#### Điều X` trong Markdown, cho phép filter đến từng điều khoản cụ thể; (2) Chuyển sang `RecursiveChunker` với separators `["\n#### Điều", "\n\n", "\n", ". "]` để tận dụng cấu trúc pháp lý thay vì dùng FixedSize; (3) Giảm chunk_size xuống 400 và overlap xuống 80 để embedding tập trung hơn vào từng khoản riêng lẻ.

---

## Tự Đánh Giá

### Điểm Cá Nhân (60 điểm)

| Hạng mục | Mô tả | Điểm tối đa | Tự đánh giá |
|----------|-------|-------------|-------------|
| Core Implementation | 42/42 tests passed | 30 | **30 / 30** |
| My Approach | Giải thích implement từng phần src | 10 | **9 / 10** |
| Competition Results | 6/6 queries relevant, 10/12 điểm | 10 | **9 / 10** |
| Warm-up | Cosine similarity + chunking math | 5 | **5 / 5** |
| Similarity Predictions | 3/5 đúng, reflection rõ | 5 | **4 / 5** |
| **Tổng cá nhân** | | **60** | **57 / 60** |

### Điểm Nhóm (40 điểm)

| Hạng mục | Mô tả | Điểm tối đa | Tự đánh giá |
|----------|-------|-------------|-------------|
| Strategy Design | Strategy cá nhân + rationale + so sánh nhóm | 15 | **13 / 15** |
| Document Set Quality | 6 tài liệu, metadata rõ ràng, nguồn minh bạch | 10 | **10 / 10** |
| Retrieval Quality | Precision 10/12 trên 6 benchmark queries | 10 | **8 / 10** |
| Demo | Insights, so sánh, bài học rút ra | 5 | **5 / 5** |
| **Tổng nhóm** | | **40** | **36 / 40** |

### **Tổng: 93 / 100**
