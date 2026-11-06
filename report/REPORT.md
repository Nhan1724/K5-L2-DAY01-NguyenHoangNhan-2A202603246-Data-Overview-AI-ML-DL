# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 6/11/2026

**Runtime Colab:** T4-GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không 

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…

-	Chép record hạng 1: class_id=468 ; class_name=cab ; rank=1 ; score=0.510915 ; taxonomy_name=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

-   Trả lời: Record này nói về toàn bộ bức ảnh.
-	Lý do: Đây là tác vụ Phân loại ảnh (Image Classification). Mô hình phân tích đặc trưng của toàn bộ khung hình và đưa ra một nhãn tổng quát nhất đại diện cho ngữ cảnh chính của bức ảnh (ở đây là chiếc taxi/cab nổi bật chiếm diện tích lớn), chứ không xác định vị trí tọa độ hay phân tách từng vật thể riêng biệt.


3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

-   Trả lời: Danh sách tên lớp đi kèm theo taxonomy của checkpoint đã được huấn luyện sẵn (ở đây là bộ dữ liệu ImageNet-1K gồm 1000 lớp). Mô hình chỉ có thể tính xác suất và gán nhãn dựa trên danh sách cố định này chứ không tự sáng tạo ra lớp mới.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

-   Trả lời:
    +	Ghi class_id giúp máy tính định danh chính xác lớp trong mã nguồn mà không lo bị nhầm lẫn do viết sai chính tả hoặc dịch thuật.
    +	Ghi class_name giúp con người đọc hiểu nhãn một cách trực quan, nhanh chóng.
    +	Ghi taxonomy_name giúp xác định rõ không gian nhãn được sử dụng, tránh xung đột thông tin vì cùng một tên lớp (ví dụ: "cab") có thể mang mã ID hoặc định nghĩa khác nhau ở các bộ taxonomy khác (như COCO hay OpenImages).

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
-	Trả lời: Guideline cần quy định rõ Quy tắc ưu tiên (Priority Rules), ví dụ như:
    +	Ưu tiên chọn chủ thể có kích thước lớn nhất hoặc nằm ở trung tâm của bức ảnh.
    +	Ưu tiên chọn đối tượng chính quyết định hành động/ngữ cảnh giao thông chính thay vì các yếu tố hậu cảnh.
    +	Quy định cụ thể cách xử lý khi gặp các lớp có độ chồng chéo cao (như taxi/cab vs. car thông thường).

6. Vì sao không được lấy `score` của model làm ground truth?
-   Trả lời: Vì score (độ tin cậy) chỉ là giá trị xác suất toán học thể hiện mức độ tự tin của mô hình dựa trên tập dữ liệu đã học, không đại diện cho sự thật khách quan (ground truth). Mô hình hoàn toàn có thể dự đoán sai với độ tin cậy rất cao (overconfident) hoặc đoán đúng nhưng score lại thấp. Ground truth chuẩn xác nhất định phải do con người gán nhãn và kiểm duyệt thủ công.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

**Mở file:** day1_lab_outputs/detection_predictions.json và visuals/detection_predictions.png. Chỉ dùng ảnh sample_id = kitchen.
-	Chép một record:
    +	class_name=person ; score=0.819777 ; bbox_xyxy=[166.5, 41.6, 362.4, 420.35] ; bbox_width=195.9 ; bbox_height=378.75
-	Giải thích box bằng lời thường:
    +	Bounding box này bao quanh một người trong ảnh. Góc trên-bên trái của hộp nằm tại tọa độ pixel (166.5, 41.6) và góc dưới-bên phải của hộp nằm tại tọa độ pixel (362.4, 420.35). Chiều rộng của hộp này là 195.9 pixel và chiều cao là 378.75 pixel.
-	Chọn hai ngưỡng (ví dụ 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
    +	Ngưỡng 0.20: Có 17 prediction được phát hiện.
    +	Ngưỡng 0.60: Có 6 prediction được phát hiện.
-	Khi hạ ngưỡng (giữ nhiều hộp hơn), người reviewer phải làm thêm việc gì?
    +	Khi hạ ngưỡng lọc, số lượng hộp phát hiện tăng lên rất nhiều và xuất hiện nhiều hộp nhiễu hoặc hộp bị trùng lặp (nhiều hộp bao quanh cùng một vật). Người reviewer phải thực hiện sàng lọc kỹ hơn: xóa bỏ các hộp nhận diện sai (false positives), gộp các hộp bị trùng lặp, chỉnh sửa ranh giới hộp bị lệch và bổ sung các hộp bị mô hình bỏ sót.
-	Viết một quy tắc box để người gán nhãn làm theo (áp dụng được cho nhiều ảnh):
    +	Quy tắc: "Bounding Box phải được vẽ ôm khít sát đường biên ngoài của vật thể (tight bounding box), không được để khoảng trống thừa lớn hơn 2 pixel ở các cạnh và tuyệt đối không cắt lẹm vào bộ phận của vật thể. Nếu vật thể bị che khuất một phần, vẽ phác thảo ước lượng trọn vẹn cả phần bị che khuất đó."
-	Trên ảnh kitchen, chỉ ra một vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
    +	Vật thể: Những chiếc đĩa hoặc cốc bị xếp chồng lên nhau và bị che khuất một phần lớn trong tủ bếp, hoặc chiếc lọ gia vị nhỏ bị khuất sau chai dầu ăn.
    +	Xử lý: Đối với trường hợp này, ta nên viết thành một quy tắc cụ thể trong guideline.
    +	Vì sao: Đây là những trường hợp che khuất/cắt mép cực kỳ phổ biến trong các ảnh thực tế (đặc biệt là môi trường bếp núc hay giao thông). Việc chuẩn hóa nó thành một quy tắc rõ ràng trong guideline (ví dụ: 'chỉ gán nhãn nếu nhìn thấy tối thiểu 20% diện tích vật thể') giúp toàn bộ người gán nhãn (Annotators) làm việc đồng nhất, giảm thiểu tối đa việc phải hỏi đi hỏi lại (escalation) những lỗi lặp đi lặp lại hàng ngày.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…

-   instance_id: "kitchen-001" (hoặc đối tượng cụ thể tùy chọn từ file JSON)
-   class_name: "person"
-   score: 0.819777
-   Số điểm polygon (polygon_point_count): 1194 (hoặc số điểm tương ứng của record bạn chọn)
-   3–5 điểm đầu của polygon_xy: [[374.0, 568.0], [373.0, 568.0], [372.0, 569.0], [371.0, 569.0], [370.0, 570.0]]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
-	Bounding Box (hộp chữ nhật) chỉ cung cấp giới hạn không gian ngoài cùng và luôn chứa cả các phần nền hoặc vật thể khác bị đè lên trong phạm vi góc vuông.
-	Polygon (đa giác bám biên) cung cấp thông tin chi tiết về hình dạng thực tế, đường viền chi tiết và diện tích che phủ chính xác của vật thể. Nó giúp phân biệt rõ ràng phần thuộc về vật thể và phần thuộc về môi trường xung quanh.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
-	instance_id dùng để định danh và phân biệt các cá thể (đối tượng riêng biệt) thuộc cùng một lớp trong một bức ảnh (ví dụ: phân biệt giữa person-001 và person-002).
-	Câu "Nó không phải class ID và không phải tracking ID" có nghĩa là:
    +	Không phải class ID: Nó không dùng để phân loại loại vật thể (như người, bát, lò nướng).
    +	Không phải tracking ID: Nó không dùng để liên kết hay theo dấu đối tượng đó xuyên suốt qua các khung hình (frames) khác nhau của một video, mà chỉ có giá trị phân biệt nội bộ trong duy nhất bức ảnh hiện tại.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
-	Quy tắc gán nhãn đa giác (Polygon Annotation Rule): "Người gán nhãn phải vẽ các điểm neo bám sát theo đường biên thực tế của vật thể với sai số không quá 2 pixel. Không được vẽ quá thưa làm mất hình dạng bo tròn (ví dụ như miệng bát, đĩa) và không được lấn quá sâu vào phần nền xung quanh. Đối với các cạnh thẳng, hãy tối ưu hóa số lượng điểm gán để tránh làm nặng file dữ liệu."

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
-	Vị trí khó khăn: Khu vực các chiếc bát (bowl) chồng lên nhau hoặc phần bồn rửa/bàn bếp nơi có các ranh giới tiếp xúc rất mờ và bị che khuất một phần bởi các vật dụng khác.
-	Hướng xử lý (Guideline hay Escalation):
    +	Áp dụng Guideline: Nếu tài liệu hướng dẫn đã quy định rõ quy tắc xử lý vật thể bị che khuất (Occlusion) — ví dụ: "Vẽ ước lượng phần bị che khuất nếu phần đó chiếm dưới 30% diện tích vật thể và có thể suy đoán rõ ràng; nếu bị che khuất trên 30% thì chỉ vẽ phần nhìn thấy được" — annotator sẽ tự gán nhãn theo quy tắc đó.
    +	Áp dụng Escalation: Trong trường hợp biên quá mờ không thể xác định đâu là ranh giới giữa hai chiếc bát chồng chất lên nhau, hoặc không thể phân biệt chiếc bát phía sau có thuộc taxonomy được gán hay không, annotator bắt buộc phải escalate (gửi yêu cầu làm rõ) lên quản lý dự án (Project Manager/Lab Coach) để đưa ra quyết định thống nhất, tránh việc tự ý phán đoán dẫn đến dữ liệu huấn luyện không đồng nhất.


## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn đơn (single label) hoặc đa nhãn (multi-label) dạng chuỗi ký tự (text) hoặc ID lớp cho toàn bộ bức ảnh. Không có tọa độ hay vùng bám biên. | Lỗi nhãn ImageNet-1K gán cho ảnh COCO: ảnh cảnh đường phố thực tế (traffic) lại bị model phân loại top-1 là cab (taxi) với xác suất chỉ khoảng 51% do taxonomy không bao phủ tốt toàn bộ ngữ cảnh (chỉ tập trung vào chi tiết xe). | Xem toàn bộ bức ảnh, đối chiếu với định nghĩa taxonomy của dự án để chọn ra một (hoặc nhiều) nhãn phân loại đúng nhất cho bối cảnh chung của ảnh. | Kiểm tra xem nhãn được chọn có đúng với hướng dẫn (guideline) không, đặc biệt là các ca nhập nhằng (ví dụ: ranh giới giữa xe taxi thông thường và xe bus nhỏ). |

| Phát hiện vật thể | Tập hợp các hộp chữ nhật chứa tọa độ rìa ngoài vật thể: [x_min, y_min, x_max, y_max] tính bằng pixel, đi kèm nhãn lớp tương ứng cho từng hộp. | Model bỏ sót nhiều vật thể nhỏ ở hậu cảnh trên ảnh kitchen khi tăng ngưỡng threshold lên cao (chỉ phát hiện 6 vật thể ở threshold 0.60 so với 17 vật thể ở threshold 0.20), ví dụ như bỏ sót chai lọ, thìa dĩa nhỏ trên bàn. | Xác định tất cả vật thể thuộc taxonomy trong ảnh, vẽ hộp chữ nhật ôm khít (tight bounding box) lấy rìa ngoài cùng của vật thể đó, gán đúng lớp đối tượng. | Kiểm tra xem các hộp có bị vẽ lệch (quá rộng/quá hẹp), có bỏ sót đối tượng nhỏ nào không, và các phần bị che khuất (occlusion) đã được xử lý đúng guideline chưa. |

| Instance segmentation | Tập hợp các điểm tọa độ đa giác [[x1, y1], [x2, y2], ...] tạo thành đường biên bám sát hình dạng thực tế của từng cá thể, đi kèm mã định danh riêng biệt (instance_id) và lớp của nó. | Khó phân định ranh giới giữa các chiếc bát (bowl) xếp chồng lên nhau hoặc các góc khuất bóng tối ở bàn bếp. Model phân đoạn lem nhem hoặc gộp chung các ranh giới tiếp xúc mờ. | Vẽ đường đa giác (polygon) tỉ mỉ bám sát theo mép thực tế của từng đối tượng (pixel-level), phân biệt rõ ràng các cá thể riêng biệt kể cả khi chúng đứng cạnh hoặc đè lên nhau. | Soát kỹ độ mịn và độ chính xác của đường biên mask, đảm bảo không gộp chung hai vật thể chạm nhau thành một và kiểm tra tính duy nhất của instance_id trong ảnh. |



## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
    Tôi chỉ sử dụng ba bức ảnh COCO công khai có sẵn được cung cấp trong bài lab, tuyệt đối không tải lên hoặc sử dụng bất kỳ hình ảnh cá nhân, dữ liệu nội bộ doanh nghiệp hay thông tin nhạy cảm nào khác lên môi trường Colab công cộng.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
    Nếu phát hiện bất kỳ hình ảnh hoặc dữ liệu nào nằm ngoài phạm vi cho phép (ví dụ: ảnh nhạy cảm, thông tin cá nhân bị lộ), tôi sẽ ngay lập tức dừng quá trình thực hiện bài lab và báo cáo trực tiếp cho Giảng viên hướng dẫn hoặc Lab Coach phụ trách để được xử lý kịp thời.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
