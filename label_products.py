"""
Gán nhãn sản phẩm dựa trên dữ liệu crawl.
- Tạo các đặc trưng cấp sản phẩm.
- Gán nhãn 5 lớp: Best Seller, High Potential, Premium, Normal, Needs Improvement.
"""

import os
import re
import pandas as pd
import numpy as np
from datetime import datetime

DATA_PATH = "data/tiki_books_reviews.csv"
OUTPUT_PATH = "data/tiki_books_labeled.csv"

# ==================== UTILITY FUNCTIONS ====================
def load_data():
    """Đọc dữ liệu từ file."""
    if not os.path.exists(DATA_PATH):
        print(f"❌ File {DATA_PATH} không tồn tại")
        print(f"   Hãy chạy: python crawl_tiki_books.py")
        exit(1)
    
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Đã đọc {len(df)} dòng từ {DATA_PATH}")
    return df

def parse_numeric_value(value):
    """Chuyển các giá trị price/sold_count sang số nguyên hợp lệ."""
    if pd.isna(value):
        return 0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    text = str(value).strip()
    if not text:
        return 0

    # Nếu là chuỗi dict hoặc JSON-like, cố gắng lấy số đầu tiên
    match = re.search(r"([0-9]+(?:[\.,][0-9]+)?)([kKmM]?)", text)
    if not match:
        return 0

    number_str, suffix = match.groups()
    number_str = number_str.replace(',', '.')
    try:
        number = float(number_str)
    except ValueError:
        return 0

    if suffix.lower() == 'k':
        number *= 1_000
    elif suffix.lower() == 'm':
        number *= 1_000_000

    return int(round(number))


def clean_data(df):
    """Làm sạch dữ liệu."""
    print("\n🧹 Làm sạch dữ liệu...")
    
    # Xử lý giá và số lượng bán trước khi numeric
    df['price'] = df['price'].apply(parse_numeric_value)
    df['sold_count'] = df['sold_count'].apply(parse_numeric_value)

    # Chuyển các cột số về numeric
    numeric_cols = ['price', 'rating', 'review_count', 'sold_count', 'comment_rating']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['sold_count'] = df['sold_count'].astype(float).round().astype(int)
    df['price'] = df['price'].astype(float).fillna(0)

    # Xử lý cột text
    df['comment_text'] = df['comment_text'].fillna('')
    df['product_url'] = df['product_url'].fillna('')
    
    print(f"   ✓ Chuyển đổi cột số")
    print(f"   ✓ Làm sạch cột text")
    
    return df

def create_product_features(df):
    """Tạo đặc trưng cấp sản phẩm."""
    print("\n📊 Tạo đặc trưng cấp sản phẩm...")
    
    # Gom theo product_id và product_name
    grouped = df.groupby(['product_id', 'product_name'], as_index=False).agg({
        'price': 'first',
        'rating': 'first',
        'review_count': 'max',
        'sold_count': 'max',
        'comment_text': lambda x: sum(1 for t in x if t.strip()),  # comment_count
        'comment_rating': 'mean',  # avg_comment_rating
        'product_url': 'first'
    })
    
    # Đổi tên cột
    grouped.rename(columns={'comment_text': 'comment_count', 'comment_rating': 'avg_comment_rating'}, inplace=True)
    
    # Tính toán tỷ lệ
    def calculate_ratios(df_group):
        """Tính tỷ lệ review theo rating."""
        comment_ratings = df_group['comment_rating'].values
        
        if len(comment_ratings) == 0:
            return 0, 0, 0
        
        total = len(comment_ratings)
        positive = sum(1 for r in comment_ratings if r >= 4)
        neutral = sum(1 for r in comment_ratings if r == 3)
        negative = sum(1 for r in comment_ratings if 0 < r < 3)
        
        return (positive / total if total > 0 else 0,
                neutral / total if total > 0 else 0,
                negative / total if total > 0 else 0)
    
    # Tính tỷ lệ cho mỗi sản phẩm
    ratios = []
    for product_id in grouped['product_id']:
        product_reviews = df[df['product_id'] == product_id]
        pos_ratio, neu_ratio, neg_ratio = calculate_ratios(product_reviews)
        ratios.append((pos_ratio, neu_ratio, neg_ratio))
    
    grouped[['positive_ratio', 'neutral_ratio', 'negative_ratio']] = pd.DataFrame(ratios, index=grouped.index)
    
    # Tính doanh thu ước tính
    grouped['estimated_revenue'] = grouped['price'] * grouped['sold_count']
    
    # Xử lý avg_comment_rating
    grouped['avg_comment_rating'] = grouped['avg_comment_rating'].fillna(0)

    print(f"   ✓ Gom theo sản phẩm: {len(grouped)} sản phẩm duy nhất")
    print(f"   - price: min={grouped['price'].min():,.0f}, max={grouped['price'].max():,.0f}, mean={grouped['price'].mean():,.0f}")
    print(f"   - sold_count: min={grouped['sold_count'].min():,.0f}, max={grouped['sold_count'].max():,.0f}, mean={grouped['sold_count'].mean():,.0f}")
    print(f"   - estimated_revenue: min={grouped['estimated_revenue'].min():,.0f}, max={grouped['estimated_revenue'].max():,.0f}, mean={grouped['estimated_revenue'].mean():,.0f}, sum={grouped['estimated_revenue'].sum():,.0f}")
    if grouped['estimated_revenue'].sum() == 0:
        print("   ⚠️ estimated_revenue toàn bộ bằng 0, kiểm tra lại price và sold_count")
    
    return grouped

def assign_labels(df):
    """Gán nhãn sản phẩm dựa trên score-based labeling."""
    print("\n🏷️  Gán nhãn sản phẩm...")
    
    df = df.copy()
    
    def minmax_scale(series):
        min_val = series.min()
        max_val = series.max()
        if max_val > min_val:
            return (series - min_val) / (max_val - min_val)
        return pd.Series(0.5, index=series.index)
    
    df['sold_score'] = minmax_scale(df['sold_count'])
    df['revenue_score'] = minmax_scale(df['estimated_revenue'])
    df['rating_score'] = minmax_scale(df['rating'])
    df['review_score'] = minmax_scale(df['review_count'])
    df['positive_score'] = minmax_scale(df['positive_ratio'])
    df['negative_penalty'] = minmax_scale(df['negative_ratio'])
    df['price_score'] = minmax_scale(df['price'])
    
    df['performance_score'] = (
        0.25 * df['sold_score'] +
        0.25 * df['revenue_score'] +
        0.20 * df['rating_score'] +
        0.10 * df['review_score'] +
        0.10 * df['positive_score'] +
        0.10 * df['price_score'] -
        0.15 * df['negative_penalty']
    ).clip(0, 1)
    
    median_positive_ratio = df['positive_ratio'].median()
    effective_positive_threshold = max(median_positive_ratio, 0.1)
    best_threshold = df['performance_score'].quantile(0.90)
    premium_threshold = df['performance_score'].quantile(0.75)
    high_threshold = df['performance_score'].quantile(0.55)
    needs_threshold = df['performance_score'].quantile(0.20)
    
    print(f"   Thống kê score:")
    print(f"   - median_positive_ratio: {median_positive_ratio:.2f}")
    print(f"   - effective_positive_threshold: {effective_positive_threshold:.2f}")
    print(f"   - best_threshold: {best_threshold:.2f}")
    print(f"   - premium_threshold: {premium_threshold:.2f}")
    print(f"   - high_threshold: {high_threshold:.2f}")
    print(f"   - needs_threshold: {needs_threshold:.2f}")
    
    labels = []
    
    for idx, row in df.iterrows():
        label = None
        
        if (row['performance_score'] >= best_threshold and
            row['rating'] >= 4.0 and
            row['negative_ratio'] <= 0.25):
            label = "Best Seller"
        elif (row['performance_score'] >= premium_threshold and
              row['rating'] >= 4.0 and
              row['negative_ratio'] <= 0.30):
            label = "Premium / Niche Quality"
        elif (row['performance_score'] >= high_threshold and
              row['rating'] >= 3.8 and
              row['positive_ratio'] >= effective_positive_threshold and
              row['negative_ratio'] <= 0.35):
            label = "High Potential"
        elif ((row['rating'] > 0 and row['rating'] < 3.5) or
              row['negative_ratio'] >= 0.35 or
              row['performance_score'] <= needs_threshold):
            label = "Needs Improvement"
        else:
            label = "Normal"
        
        labels.append(label)
    
    df['product_label'] = labels
    
    return df

def print_statistics(df):
    """In thống kê."""
    print("\n" + "="*60)
    print("📊 THỐNG KÊ GÁN NHÃN")
    print("="*60)
    print(f"Tổng số sản phẩm: {len(df)}")
    
    label_order = [
        "Best Seller",
        "High Potential",
        "Premium / Niche Quality",
        "Normal",
        "Needs Improvement"
    ]
    label_counts = df['product_label'].value_counts()
    present_labels = 0
    for label in label_order:
        count = int(label_counts.get(label, 0))
        print(f"  {label}: {count}")
        if count > 0:
            present_labels += 1
    
    total_revenue = df['estimated_revenue'].sum()
    print(f"\nTổng doanh thu ước tính: {total_revenue:,.0f} VND")
    
    has_comment = len(df[df['comment_count'] > 0])
    no_comment = len(df[df['comment_count'] == 0])
    print(f"\nCó bình luận: {has_comment} sản phẩm")
    print(f"Không bình luận: {no_comment} sản phẩm")
    
    if present_labels < 3:
        print("\n⚠️ Cảnh báo: phân bố nhãn chưa đa dạng, cần kiểm tra lại ngưỡng.")
    
    print(f"\n✅ Lưu: {OUTPUT_PATH}")
    print("="*60)

def main():
    print(f"\n🚀 Tiki Book Intelligence - Gán nhãn sản phẩm")
    print(f"   Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Đọc và làm sạch dữ liệu
    df = load_data()
    df = clean_data(df)
    
    # Tạo đặc trưng
    df_products = create_product_features(df)
    
    # Gán nhãn
    df_products = assign_labels(df_products)
    
    # In thống kê
    print_statistics(df_products)
    
    # Lưu dữ liệu
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    df_products.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n✅ Hoàn thành!")

if __name__ == "__main__":
    main()
