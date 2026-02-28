# 🛍️ Phân Tích Cảm Xúc Đa Khía Cạnh Tiếng Việt (Vietnamese ABSA)

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=flat&logo=streamlit&logoColor=white)

Đồ án Xử lý Ngôn ngữ Tự nhiên (NLP) tập trung giải quyết bài toán **Aspect-Based Sentiment Analysis (ABSA)** trên tập dữ liệu đánh giá sản phẩm thương mại điện tử (Shopee) bằng tiếng Việt.

Dự án đối sánh hiệu năng giữa mô hình học sâu tuần tự cơ sở (**BiLSTM-Attention**) và kiến trúc tự chú ý tiên tiến (**PhoBERT**), đồng thời triển khai một giao diện Web Demo thực tế.

## 🌟 Tính Năng Nổi Bật

* **Tiền xử lý văn bản tiếng Việt:** Tích hợp bộ từ điển chuẩn hóa Teencode tự xây dựng và bộ tách từ `underthesea`.
* **Phân loại 8 Khía cạnh (Aspects):** `Price` (Giá cả), `Shipping` (Giao hàng), `Outlook` (Hình thức), `Quality` (Chất lượng), `Size` (Kích cỡ), `Shop_Service` (Dịch vụ), `General` (Đánh giá chung), `Others` (Khác).
* **Phân cực 4 Sắc thái (Polarities):** Tích cực (Positive), Tiêu cực (Negative), Trung tính (Neutral), Không đề cập (None).
* **Xử lý Mất cân bằng lớp (Class Imbalance):** Áp dụng kỹ thuật Class Weights vào hàm Weighted Cross-Entropy Loss.
* **Giao diện Web Tương tác:** Tích hợp Streamlit cho phép dự đoán cảm xúc thời gian thực (Real-time Inference).

## 📊 Kết Quả Đánh Giá (Tập Test)

Sử dụng độ đo **Macro F1-Score** để đánh giá độ chính xác trên tập dữ liệu mất cân bằng. Mô hình PhoBERT thể hiện sự vượt trội trong việc nắm bắt ngữ cảnh tiếng Việt phức tạp.

| Mô hình | Average Training Loss | Macro F1-Score (Tổng thể) |
| :--- | :---: | :---: |
| **BiLSTM-Attention** | 0.0161 | 0.6444 |
| **PhoBERT (Fine-tuned)** | 0.1085 | **0.7013** |

*(Đồ thị so sánh Loss và bảng F1-Score chi tiết từng khía cạnh xem trong file báo cáo hoặc Notebook).*

## 📁 Cấu Trúc Thư Mục

```text
├── datasets_cleaned/       # Chứa dữ liệu train/val/test đã qua tiền xử lý
├── images/                 # Chứa các biểu đồ trực quan hóa (Learning curves, Data distribution)
├── model/                  # Chứa file kiến trúc mạng nơ-ron
│   ├── bi_lstm.py          # Khởi tạo class BiLSTMAttentionABSA
│   └── pho_bert.py         # Khởi tạo class PhoBERT_ABSA
├── saved_models/           # (Ignored) Nơi lưu trữ trọng số mô hình tốt nhất (.pth)
├── app.py                  # Mã nguồn giao diện Web Demo bằng Streamlit
├── eda.py                  # Mã nguồn phân tích và trực quan hóa dữ liệu (EDA)
├── text_preprocessing.py   # Các hàm làm sạch văn bản, chuẩn hóa teencode
├── train.ipynb             # Notebook huấn luyện mô hình
├── test.ipynb              # Notebook chạy đối sánh và in kết quả dự đoán
├── requirements.txt        # Danh sách các thư viện cần cài đặt
└── README.md               # Tài liệu mô tả dự án

🚀 Hướng Dẫn Cài Đặt và Sử Dụng
Bước 1: Clone Repository
```bash
git clone [https://github.com/nhg011244/ABSA_NLP.git](https://github.com/nhg011244/ABSA_NLP.git)

Bước 2: Cài đặt thư viện
Khuyến nghị sử dụng môi trường ảo (Virtual Environment) để tránh xung đột thư viện.

pip install -r requirements.txt

Bước 3: Chạy giao diện Web Demo
Đảm bảo bạn đã có file trọng số phobert_absa_weights.pth nằm trong thư mục saved_models/. Sau đó chạy lệnh:

streamlit run app.py
Trình duyệt sẽ tự động mở trang Web Demo tại địa chỉ: http://localhost:8501