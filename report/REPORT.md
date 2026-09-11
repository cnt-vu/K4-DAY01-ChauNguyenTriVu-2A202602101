# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 2026-09-11

**Runtime Colab:** GPU (T4)

**Python / PyTorch / Ultralytics:** Python 3.10.12 / PyTorch 2.5.1+cu124 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): `class_id`: 468, `class_name`: "cab", `rank`: 1, `score`: 0.510915, `taxonomy_name`: "ImageNet-1K".
- Record này mô tả toàn ảnh như thế nào? Mô hình gán một nhãn phân loại toàn cục duy nhất cho toàn bộ bức ảnh; nó chỉ ra thực thể nổi bật nhất theo suy luận của mô hình (xe taxi - "cab" với độ tự tin ~51.09%) mà không cung cấp vị trí không gian (tọa độ pixel), kích thước hộp bao hay số lượng các thực thể khác cùng xuất hiện trong cảnh.
- Ai định nghĩa class list mà checkpoint có thể dự đoán? Nhóm tác giả và tổ chức xây dựng bộ dữ liệu ImageNet-1K đã thiết lập danh mục cố định gồm 1.000 lớp ngữ nghĩa mà mô hình `yolo11n-cls.pt` được huấn luyện từ trước.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  * `class_id`: Tối ưu hóa việc lập chỉ mục, lưu trữ dữ liệu và xử lý logic số học trong pipeline máy học.
  * `class_name`: Giúp annotator, reviewer và kỹ sư đọc hiểu ngữ nghĩa trực quan.
  * `taxonomy_name`: Định danh không gian phân loại (schema/ontology), ngăn ngừa xung đột ngữ nghĩa khi hợp nhất dữ liệu từ nhiều chuẩn khác nhau (ví dụ: lớp xe cộ trong ImageNet-1K khác với COCO-80).
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? Guideline cần nêu rõ thứ tự ưu tiên: (1) Gán nhãn cho chủ thể chiếm diện tích trung tâm lớn nhất; (2) Chuyển đổi bài toán sang phân loại đa nhãn (multi-label classification); hoặc (3) Gán nhãn bối cảnh bao quát (scene-level label, ví dụ: "street scene") thay vì gán nhãn vật thể đơn lẻ.
- Vì sao model score không phải ground truth? Model score chỉ là xác suất thống kê (softmax confidence) thể hiện độ tự tin mang tính phỏng đoán của mạng nơ-ron trên phân phối dữ liệu đã học, có thể sai lệch hoặc thiên kiến; ground truth là nhãn thực tế khách quan do con người thẩm định và xác nhận theo đúng thực tế.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): `class_name`: "person", `score`: 0.912625, `bbox_xyxy`: [385.33, 69.24, 498.92, 348.92], `bbox_width`: 113.58, `bbox_height`: 279.68.
- Diễn giải vị trí box bằng lời: Hộp bao quanh người ("person") nằm ở khu vực phía bên phải của căn bếp, kéo dài từ tọa độ góc trên bên trái `(x=385.33, y=69.24)` xuống góc dưới bên phải `(x=498.92, y=348.92)`.
- So sánh số prediction ở hai threshold: Ở ngưỡng mặc định của bài thực hành (`conf = 0.35`), mô hình phát hiện 11 đối tượng (2 person, 5 bowl, 2 oven, 2 cup). Nếu tăng ngưỡng lọc lên `conf = 0.50`, số prediction giảm còn 6 đối tượng (loại bỏ 3 bowl có score 0.499, 0.465, 0.381 và 2 cup có score 0.451, 0.382).
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? Ngưỡng thấp tăng độ bao phủ (recall cao, ít bỏ sót các vật thể nhỏ như bát, cốc) nhưng kéo theo dự đoán nhiễu (false positives), làm tăng khối lượng kiểm duyệt và sửa lỗi của reviewer; ngưỡng cao lọc sạch nhiễu nhưng dễ bỏ lọt các đối tượng thực tế (tăng false negatives).
- Đề xuất một quy tắc box chặt: Bounding box phải bao trọn các điểm biên ngoài cùng của phần nhìn thấy được của vật thể, khoảng cách viền thừa không quá 2 pixel và tuyệt đối không cắt lẹm vào chi tiết của đối tượng.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? Cần quy định tỷ lệ diện tích nhìn thấy tối thiểu để đóng box (ví dụ: > 15-20%), quy định đóng box theo phần nhìn thấy thực tế (visible box) hay ước lượng cả phần bị khuất (amodal box); nếu vật thể bị chia cắt làm hai mảng do chướng ngại vật chắn ngang, cần quy chuẩn đóng 1 box chung hay 2 box rời rạc. Khi vượt ngoài guideline, annotator phải gắn cờ escalation để reviewer thống nhất.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): `instance_id`: "kitchen-001", `class_name`: "person", `score`: 0.899318, số điểm: 348 điểm, `polygon_xy`: [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0], ...].
- Polygon bổ sung chi tiết gì so với box? Polygon bám sát đường bao hình học thực tế của đối tượng (contour), loại bỏ hoàn toàn các pixel nền thừa (background) mà bounding box hình chữ nhật bắt buộc phải bao gồm.
- `instance_id` dùng để làm gì và không phải loại ID nào? Dùng để phân biệt các cá thể riêng biệt trong cùng một bức ảnh (ví dụ: tách biệt `kitchen-001` và `kitchen-009` đều là "person"); đây không phải là `class_id` (mã lớp ngữ nghĩa) và không phải là ID định danh toàn cục cố định xuyên suốt dataset.
- Đề xuất một quy tắc biên mask: Đường biên đa giác phải ôm khít đường viền thực của đối tượng với sai lệch không quá 1-2 pixel; các đoạn cong phải có mật độ điểm neo vừa đủ mịn, không tạo góc nhọn gãy khúc giả tạo và không để lọt pixel nền vào bên trong mask của vật thể liền khối.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định? Guideline cần quy định ranh giới chia cắt khi hai vật thể cùng màu chạm nhau (như các bát xếp cạnh nhau), quy chuẩn mask tại vùng bóng đổ hoặc nhòe mờ chuyển động; trường hợp mắt thường không thể phân biệt ranh giới thực tế, annotator cần gắn cờ escalation để hội đồng kiểm duyệt ra quyết định thống nhất.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn đơn cấp ảnh (`class_id` / `class_name`). | Ảnh chứa nhiều chủ thể ngang hàng (ví dụ: vừa có taxi, xe buýt, xe con); góc chụp xa gây nhiễu. | Đối chiếu quy tắc trọng tâm/diện tích trong guideline để chọn 1 nhãn bao quát nhất. | Kiểm tra tính nhất quán ngữ nghĩa toàn cục và phát hiện trường hợp ép nhãn sai bối cảnh. |
| Phát hiện vật thể | Tọa độ Bounding Box (`[x1, y1, x2, y2]`) kèm nhãn lớp. | Box quá rộng dính nền, cắt phạm biên vật thể, bỏ sót đồ vật nhỏ ở hậu cảnh (như cup, bowl). | Kéo box ôm sát mép biên thực tế của từng đối tượng; loại bỏ box trùng lặp. | Đo độ chặt viền (IoU), rà soát nhầm nhãn lớp, kiểm tra tỷ lệ sót đối tượng (false negatives). |
| Instance segmentation | Tập hợp đỉnh Polygon (`[[x, y], ...]`) kèm `instance_id` và nhãn lớp. | Đường viền polygon gãy khúc thô, lẹm sang nền hoặc dính liền hai vật thể đặt cạnh nhau. | Chấm các điểm polygon bám sát đường cong thực tế, tách biệt ranh giới từng cá thể riêng biệt. | Phóng to (zoom-in) kiểm tra độ mịn của biên, hiện tượng rò rỉ mask ra ngoài nền và tính chuẩn xác của từng instance. |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Không sao chép, tải về máy cá nhân hoặc phát tán dữ liệu hình ảnh nội bộ và file nhãn ra ngoài môi trường thực hành được quy định; không lưu trữ thông tin định danh cá nhân (PII) trong các báo cáo kỹ thuật.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Giảng viên phụ trách môn học / Quản trị viên hệ thống (Data Lead).

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.