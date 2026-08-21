# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Mạnh Thắng  **MSSV**: 2A202601944  **Ngày**: 21/08/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `T4 16 GB`

Các số liệu dưới đây lấy từ lượt full eval: 50 mẫu target và 15 mẫu regression.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 steps |

**Template có giữ khối `<think>` không?** Có — `template_check.json` xác nhận reasoning được giữ và an toàn để train trace.

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Tỷ lệ supervised 41.49% cho thấy prompt hệ thống và câu hỏi người dùng đã được mask; JSON trả lời được giữ lại để tính loss.

## 3. Ba baseline (NB2 — đo trước khi train)

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.000 | 0.7578 | 0.000 | 3649.5 |
| (b) base + optimized prompt | 0.765 | 0.7578 | 1.000 | 1132.2 |
| (c) LoRA fine-tune | 0.965 | 0.7222 | 1.000 | 1606.6 |

**(b) có thật sự mạnh hơn (a) không?** Có. Prompt tối ưu tăng target từ 0.000 lên 0.765, đưa format từ 0 lên 1.000 và giảm latency. Tôi không sửa `OPTIMIZED_PROMPT`; gatekeeper xác nhận checksum prompt không đổi. Vì vậy LoRA được so với một baseline prompt mạnh.

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss | **target** | s | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6268 | 0.965 | 1031.6 | 12.01 |
| `attn_only` | q,v | 283 (matched) | 32,456,704 | 0.0001 | 0.5390 | 0.960 | 874.2 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5704 | 0.000 | 1011.0 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.940 | 1099.6 | 7.09 |

**4.1 — Vị trí adapter so với rank.** `attn_only` có 32,456,704 tham số trainable, gần bằng `correct` (sai lệch dưới 5%), nên đây là đối chứng công bằng về vị trí gắn adapter. Nó đạt target 0.960 và thua sát `correct` ở 0.965. Theo train loss thì thứ tự lại đảo: `attn_only` thấp hơn, 0.5390 so với 0.6268. Vì vậy rank tăng để khớp ngân sách không thay thế hoàn toàn lợi ích của việc đặt adapter trên text-linear layers.

**4.2 — Sai learning rate.** `wrong_lr` chỉ khác `correct` ở LR 0.00001 thay vì 0.0001. Final loss tăng lên 1.5704 và target, format đều rơi về 0. Nếu chỉ nhìn loss mà không biết LR, tôi có thể nhầm đây là lỗi dữ liệu hoặc placement; thực nghiệm cho thấy thang LR full fine-tuning quá nhỏ cho LoRA ở cấu hình này.

**4.3 — QLoRA.** QLoRA giảm peak VRAM từ 12.01 GB xuống 7.09 GB, tiết kiệm 4.92 GB (khoảng 41%). Đổi lại, thời gian train tăng từ 1031.6 s lên 1099.6 s và target giảm từ 0.965 xuống 0.940. Vì T4 16 GB đã chứa được bản 16-bit, số đo này ủng hộ việc không ưu tiên QLoRA khi VRAM không phải nút thắt.

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.200` · `regression Δ = -0.036` · `valid_trace_rate = 0.00`

Fine-tune thắng baseline prompt tối ưu trên target, 0.965 so với 0.765, nhưng regression giảm từ 0.7578 xuống 0.7222. Mức giảm -0.0356 vượt ngưỡng cho phép -0.020 nên không thể kết luận adapter an toàn để deploy, dù cả hai đều tạo JSON hợp lệ hoàn toàn. Fine-tune cũng chậm hơn baseline prompt tối ưu, 1606.6 ms so với 1132.2 ms. Đây phù hợp với catastrophic forgetting khi tập train chuyên biệt 225 ticket không có dữ liệu replay cho kiến thức/chỉ dẫn phổ thông. `valid_trace_rate` bằng 0.00 cũng phù hợp với corpus trả lời JSON không có reasoning trace. Nếu cải tiến, tôi sẽ trộn 1–5% replay data, giữ eval set đóng băng và đo lại đủ bốn nhóm.

## 6. Định tính — có cả ca thua

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Đổi trả chuột không dây, cần gấp | doi_tra; cao; chuột không dây; tích cực | Điểm thấp hơn FT | 1.00 | ✅ FT thắng |
| 2 | Hoàn tiền đèn bàn LED, quá hạn | hoan_tien; cao; đèn bàn LED; tích cực | Điểm thấp hơn FT | 1.00 | ✅ FT thắng |
| 3 | Chưa thấy tiền bình giữ nhiệt | hoan_tien; thấp; bình giữ nhiệt; tích cực | Mạnh hơn FT ở mẫu này | 0.75 | ❌ FT thua: sai một trường |
| 4 | Nồi chiên không dầu thiếu phụ kiện | san_pham_loi; thấp; nồi chiên không dầu; trung tính | Mạnh hơn FT ở mẫu này | 0.75 | ❌ FT thua: sai một trường |
| 5 | Đèn bàn LED giao chậm | van_chuyen; thấp; đèn bàn LED; tích cực | Điểm thấp hơn FT | 1.00 | ✅ FT thắng |

Các ca FT thua đều đạt 0.75: model đã nhận đúng cấu trúc và phần lớn trường, nhưng sai một nhãn tinh tế như urgency hoặc sentiment. Những lỗi này dễ bị che khuất nếu chỉ nhìn accuracy tổng, nên tôi giữ cả các ca thua trong report.

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi không nên deploy bản fine-tune này ngay ở trạng thái hiện tại. Nó cải thiện mạnh target từ 0.765 của base model với optimized prompt lên 0.965 và vẫn duy trì JSON format hoàn toàn hợp lệ. Tuy nhiên, mức cải thiện đó không đủ để biện minh cho deployment khi regression giảm 0.0356, vượt ngưỡng 0.020 của cổng an toàn. Bản fine-tune cũng chậm hơn baseline prompt tối ưu khoảng 474 ms/mẫu. Kết quả quan trọng nhất của lab là một phép so sánh công bằng có thể phát hiện chi phí ẩn của cải thiện target. Mask đúng là điều kiện nền tảng: chỉ 41.49% token được supervised và câu hỏi không bị tính loss. Trong các đòn bẩy đã thử, learning rate nhạy nhất: LR 1e-5 làm target và format cùng rơi về 0. Placement text-linear nhỉnh hơn attention-only dù đã khớp ngân sách tham số; QLoRA tiết kiệm VRAM nhưng không cần thiết trên T4. Bước tiếp theo hợp lý là bổ sung replay data và lặp lại đánh giá bốn nhóm trước khi quyết định triển khai.

**Ba điều tôi học được:**

1. Không dùng train loss để xếp hạng cấu hình: `attn_only` loss 0.5390 thấp hơn `correct` 0.6268 nhưng target vẫn thấp hơn, 0.960 so với 0.965.
2. `COMPUTE_TIER=T4` chỉ chọn cấu hình; notebook phải chạy trong runtime T4 Colab. VS Code local báo `GPU: NONE`, không phù hợp cho NB2–NB5.
3. Baseline prompt tối ưu là bắt buộc: riêng prompt đã nâng target từ 0.000 lên 0.765, nên không được quy toàn bộ mức tăng cho LoRA.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 1–5% replay data, giữ nguyên eval set, train lại `correct` và đo lại target, regression, format, latency để kiểm tra cổng regression có chuyển từ FAILED sang PASSED hay không.

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
