# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Điều làm tôi ngạc nhiên nhất là hiệu quả của prompt tối ưu trước khi fine-tune. Ở lượt chạy thử với 8 mẫu eval, baseline prompt thường có target bằng 0.000, trong khi prompt tối ưu đạt 0.688. Điều này cho thấy không thể chỉ thấy LoRA đạt điểm cao rồi kết luận toàn bộ cải thiện đến từ fine-tuning; một baseline prompt được thiết kế cẩn thận đã giải quyết được phần đáng kể của bài toán.
**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở phần huấn luyện và các run đối chứng NB4, đặc biệt là khi tải lại model và chạy bốn cấu hình. Tôi cũng mất thời gian vì ban đầu chạy notebook trong VS Code: cấu hình ghi T4 nhưng kernel local chỉ có CPU, nên NB2 không thể chạy đúng như Colab. Tôi đã dự đoán train sẽ lâu, nhưng không dự đoán việc kiểm tra đúng runtime và chạy lại eval đủ 50 mẫu cũng quan trọng đến vậy.
**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước lab, tôi nghĩ chỉ cần loss train thấp hơn là mô hình fine-tune tốt hơn. Sau lab, tôi không còn tin điều đó. Bốn run có thể có train loss khác nhau, nhưng cần xếp hạng bằng target score trên tập eval cố định, đồng thời kiểm tra regression, định dạng JSON và latency. Một cấu hình có loss thấp chưa chắc là cấu hình tốt để dùng thực tế.
**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi dùng AI assistant để đọc yêu cầu, kiểm tra artefact đã sinh ra, giải thích lỗi CPU/GPU và lập thứ tự chạy lại các notebook. AI giúp tôi nhận ra `COMPUTE_TIER=T4` không thể tự biến máy local thành GPU T4. Tuy nhiên, ban đầu tôi định nén kết quả quá sớm; gatekeeper cho thấy run đó vẫn đang dùng `EVAL_LIMIT=8` nên chưa thể nộp. Tôi cần luôn đối chiếu lời gợi ý với output của `make verify` thay vì chỉ dựa vào hướng dẫn.
**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên của tôi là xác định rõ thước đo thành công và đóng băng một tập eval đại diện trước khi train. Tôi sẽ tạo baseline mạnh bằng prompt, kiểm tra định dạng đầu ra và các tình huống regression quan trọng cho khách hàng. Chỉ sau khi biết baseline hiện có đang yếu ở đâu, tôi mới quyết định có cần fine-tune hay không và chuẩn bị dữ liệu phù hợp.
