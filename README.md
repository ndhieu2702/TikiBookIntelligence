# Tiki Book Intelligence

**Hệ thống dự đoán và phân tích hiệu quả sản phẩm sách trên sàn thương mại điện tử Tiki**

## Giới thiệu

Tiki Book Intelligence là một hệ thống thông minh dùng **học có giám sát** để dự đoán và phân tích hiệu quả của các sản phẩm sách bán trên Tiki. 

Hệ thống sử dụng **XGBoost Classifier** làm mô hình chính, với **Decision Tree** và **Random Forest** để so sánh.

### 5 Nhãn hiệu quả sản phẩm

1. **Best Seller** 🌟: Sản phẩm bán chạy, doanh thu cao, đánh giá tốt
2. **High Potential** 📈: Sản phẩm có tiềm năng phát triển
3. **Premium / Niche Quality** 💎: Sản phẩm giá cao/ngách, chất lượng tốt
4. **Normal** 📊: Sản phẩm bình thường
5. **Needs Improvement** ⚠️: Sản phẩm cần cải thiện

## Cấu trúc thư mục

```
TikiBookIntelligence/
├── data/
│   ├── tiki_books_reviews.csv          # Dữ liệu crawl (dòng là review)
│   ├── tiki_books_reviews_backup.csv   # Backup dữ liệu cũ
│   ├── tiki_books_labeled.csv          # Dữ liệu đã gán nhãn (dòng là sản phẩm)
│   ├── model_comparison.csv            # So sánh các mô hình
│   ├── classification_report.csv       # Classification report chi tiết
│   └── crawl_state.json                # Trạng thái crawl (trang cuối)
├── images/
│   ├── confusion_matrix.png            # Confusion matrix của mô hình tốt nhất
│   └── feature_importance.png          # Feature importance
├── models/
│   ├── product_performance_model.pkl   # Mô hình huấn luyện (XGBoost/Random Forest/Decision Tree)
│   └── label_encoder.pkl               # Label encoder
├── crawl_tiki_books.py                 # Crawler crawl dữ liệu từ Tiki
├── label_products.py                   # Gán nhãn sản phẩm
├── train_product_classifier.py         # Huấn luyện mô hình
├── app.py                              # Dashboard Streamlit
├── requirements.txt                    # Thư viện cần thiết
└── README.md                           # File này
```

## Cài đặt

### 1. Clone hoặc tạo thư mục dự án

```bash
cd TikiBookIntelligence
```

### 2. Cài đặt thư viện

```bash
python -m pip install -r requirements.txt
```

## Cách sử dụng

### Bước 1: Crawl dữ liệu

#### Crawl lần đầu (khoảng 3000 sản phẩm)

```bash
python crawl_tiki_books.py
```

Lần đầu sẽ crawl **60 trang** (mỗi trang 50 sản phẩm = ~3000 sản phẩm).

Kết quả:
- `data/tiki_books_reviews.csv`: Dữ liệu crawl
- `data/crawl_state.json`: Trạng thái crawl (last_crawled_page = 60)

#### Crawl tiếp (lần sau)

```bash
python crawl_tiki_books.py
```

Lần này sẽ crawl từ trang 61 đến trang 120, dữ liệu mới được cộng dồn vào file cũ (tự động chống trùng).

#### Crawl số trang tùy chọn

```bash
python crawl_tiki_books.py --pages 100
```

Crawl 100 trang tiếp theo.

#### Crawl với số sản phẩm mỗi trang tùy chọn

```bash
python crawl_tiki_books.py --pages 100 --limit 50
```

#### Reset tiến độ crawl

```bash
python crawl_tiki_books.py --reset-state
```

Đặt lại `last_crawled_page = 0`, lần sau crawl lại từ trang 1, **nhưng không xóa file CSV**.

### Bước 2: Gán nhãn sản phẩm

```bash
python label_products.py
```

Kết quả:
- `data/tiki_books_labeled.csv`: Dữ liệu gán nhãn (mỗi dòng là 1 sản phẩm)

Đặc trưng tạo ra:
- `comment_count`: Số bình luận
- `avg_comment_rating`: Đánh giá trung bình
- `positive_ratio`: Tỷ lệ bình luận tích cực
- `neutral_ratio`: Tỷ lệ bình luận trung lập
- `negative_ratio`: Tỷ lệ bình luận tiêu cực
- `estimated_revenue`: Doanh thu ước tính (price × sold_count)
- `product_label`: Nhãn sản phẩm (5 lớp)

### Bước 3: Huấn luyện mô hình

```bash
python train_product_classifier.py
```

Kết quả:
- `data/model_comparison.csv`: So sánh 3 mô hình (Decision Tree, Random Forest, XGBoost)
- `data/classification_report.csv`: Chi tiết precision, recall, f1-score cho mỗi lớp
- `images/confusion_matrix.png`: Confusion matrix của mô hình tốt nhất
- `images/feature_importance.png`: Tầm quan trọng của các đặc trưng
- `models/product_performance_model.pkl`: Mô hình tốt nhất (XGBoost nếu có thư viện)
- `models/label_encoder.pkl`: Label encoder

### Bước 4: Chạy dashboard

```bash
python -m streamlit run app.py
```

Mở trình duyệt, truy cập `http://localhost:8501`

#### Các trang trong dashboard:

- **🏠 Tổng quan**: Thống kê cơ bản, biểu đồ phân bố nhãn
- **📦 Dữ liệu sản phẩm**: Xem toàn bộ dữ liệu, tìm kiếm theo tên
- **🏷️ Gán nhãn sản phẩm**: Giải thích 5 nhãn, hiển thị biểu đồ và bảng
- **🤖 Mô hình dự đoán**: Bảng so sánh 3 mô hình, mô hình tốt nhất
- **📊 Đánh giá mô hình**: Confusion matrix, feature importance, classification report
- **🔍 Dự đoán sản phẩm**: Form dự đoán nhãn cho sản phẩm mới
- **💡 Gợi ý cải thiện**: Danh sách sản phẩm cần cải thiện với gợi ý cụ thể
- **✅ Kết luận**: Tổng kết hệ thống

## Quy trình hoàn chỉnh

```bash
# 1. Cài thư viện
python -m pip install -r requirements.txt

# 2. Crawl lần đầu
python crawl_tiki_books.py

# 3. Gán nhãn
python label_products.py

# 4. Huấn luyện mô hình
python train_product_classifier.py

# 5. Chạy dashboard
python -m streamlit run app.py
```

## Ghi chú quan trọng

### Doanh thu ước tính

Doanh thu = **giá bán × số lượng đã bán**

Đây là ước tính, **không phải doanh thu thực tế** (chưa tính chi phí, hoàn hàng, v.v.)

### Nhãn sản phẩm

Nhãn được tạo bằng **rule-based labeling** dựa trên:
- Số lượng bán (`sold_count`)
- Doanh thu ước tính (`estimated_revenue`)
- Đánh giá (`rating`)
- Tỷ lệ phản hồi (`positive_ratio`, `negative_ratio`)

Nhãn chưa phải là nhãn chuyên gia, mà là ước tính dựa trên dữ liệu.

### Mô hình chính

Nếu cài **xgboost**, mô hình chính là **XGBoost Classifier** vì:
- Phù hợp với dữ liệu dạng bảng
- Cho phép đánh giá bằng các chỉ số định lượng
- Có feature importance rõ ràng

Nếu chưa cài, sẽ dùng **Random Forest** hoặc **Decision Tree**.

## Hướng phát triển tương lai

1. **Crawl định kỳ**: Tự động cập nhật dữ liệu hàng ngày
2. **Phân tích cảm xúc**: NLP để phân tích feedback tiếng Việt
3. **Dự đoán xu hướng**: Time series để dự đoán doanh thu tương lai
4. **Triển khai online**: Đưa dashboard lên cloud (Heroku, AWS, v.v.)
5. **Tích hợp API**: Kết nối với hệ thống quản lý sản phẩm

## Liên hệ

- **Dự án**: Tiki Book Intelligence
- **Ngôn ngữ**: Python 3.8+
- **Framework**: Streamlit, Scikit-learn, XGBoost

---

**Cập nhật lần cuối**: 2026-05-11
