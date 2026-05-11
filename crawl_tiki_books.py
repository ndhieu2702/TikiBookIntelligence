"""
Crawler để crawl dữ liệu sách từ Tiki API.
- Crawl mà không cần đăng nhập.
- Nhớ trang cuối để crawl tiếp lần sau.
- Chống trùng và gộp dữ liệu.
"""

import os
import json
import random
import time
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# ==================== CONFIGURATION ====================
PRODUCTS_PER_PAGE = 50
DEFAULT_PAGES_PER_RUN = 60
REVIEWS_PER_PAGE = 10
MAX_REVIEW_PAGES = 3
DATA_PATH = "data/tiki_books_reviews.csv"
BACKUP_PATH = "data/tiki_books_reviews_backup.csv"
STATE_PATH = "data/crawl_state.json"
EMPTY_PAGE_LIMIT = 3

# Tiki API endpoints
SEARCH_URL = "https://tiki.vn/api/v2/products"
REVIEW_URL = "https://tiki.vn/api/v2/reviews"
PRODUCT_DETAIL_API = "https://tiki.vn/api/v2/products/{product_id}"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://tiki.vn/",
}

# ==================== UTILITY FUNCTIONS ====================
def load_crawl_state():
    """Đọc trạng thái crawl từ file."""
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, 'r') as f:
                state = json.load(f)
                return state.get("last_crawled_page", 0)
        except:
            return 0
    return 0

def save_crawl_state(last_page):
    """Lưu trạng thái crawl vào file."""
    os.makedirs(os.path.dirname(STATE_PATH) or ".", exist_ok=True)
    state = {"last_crawled_page": last_page}
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f)

def make_request(url, params=None, timeout=10):
    """Gửi HTTP request với xử lý lỗi."""
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
        if response.status_code == 404:
            print(f"  ⚠️  Endpoint sản phẩm không hợp lệ, kiểm tra SEARCH_URL: {url}")
            print(f"      Status code: {response.status_code}")
            return None
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print(f"  ⚠️  Timeout: {url} params={params}")
        return None
    except requests.exceptions.RequestException as e:
        status = getattr(e.response, 'status_code', None)
        if status == 404:
            print(f"  ⚠️  Endpoint sản phẩm không hợp lệ, kiểm tra SEARCH_URL: {url}")
        print(f"  ⚠️  Request lỗi: {e} url={url} params={params}")
        return None
    except json.JSONDecodeError:
        print(f"  ⚠️  JSON lỗi: {url} params={params}")
        return None

def get_reviews(product_id):
    """Crawl review của một sản phẩm."""
    reviews = []
    for review_page in range(1, MAX_REVIEW_PAGES + 1):
        params = {
            "limit": REVIEWS_PER_PAGE,
            "include": "comments,contribute_info,attribute_vote_summary",
            "sort": "score|desc,id|desc,stars|all",
            "page": review_page,
            "product_id": product_id,
        }
        data = make_request(REVIEW_URL, params=params)
        if not data:
            print(f"Không lấy được review product_id={product_id}")
            return []
        
        review_list = data.get("data", [])
        if not review_list:
            break
        
        for review in review_list:
            reviews.append({
                "comment_text": review.get("content", "").strip(),
                "comment_rating": review.get("rating", 0)
            })
        
        time.sleep(random.uniform(0.3, 0.6))
    
    return reviews

def crawl_products(start_page, end_page, products_per_page=PRODUCTS_PER_PAGE):
    """Crawl dữ liệu sản phẩm từ Tiki API."""
    all_products = []
    
    print(f"\n📖 Bắt đầu crawl từ trang {start_page} đến trang {end_page}")
    print(f"   Mỗi trang: {products_per_page} sản phẩm")
    
    empty_count = 0
    
    for page in tqdm(range(start_page, end_page + 1), desc="Crawling"):
        params = {
            "limit": products_per_page,
            "page": page,
            "q": "sách",
            "include": "advertisement",
            "aggregations": 2,
            "version": "home-persionalized",
        }
        data = make_request(SEARCH_URL, params=params)
        
        if not data:
            empty_count += 1
            print(f"  Trang {page}: không lấy được dữ liệu (API lỗi)")
            if empty_count >= EMPTY_PAGE_LIMIT:
                print(f"  ❌ Gặp {EMPTY_PAGE_LIMIT} trang rỗng, dừng crawl sớm")
                return all_products, page
            time.sleep(random.uniform(1.0, 2.0))
            continue
        
        products = data.get("data", [])
        
        if not products:
            empty_count += 1
            print(f"  Trang {page}: không có sản phẩm")
            if empty_count >= EMPTY_PAGE_LIMIT:
                print(f"  ❌ Gặp {EMPTY_PAGE_LIMIT} trang rỗng, dừng crawl sớm")
                return all_products, page
            time.sleep(random.uniform(1.0, 2.0))
            continue
        
        empty_count = 0
        print(f"  Trang {page}: lấy được {len(products)} sản phẩm")
        
        for idx, product in enumerate(products, start=1):
            product_id = product.get("id")
            product_name = product.get("name", "").strip()
            print(f"  Đang xử lý sản phẩm {idx}/{len(products)}: {product_name[:50]}...")
            price = product.get("price", 0)
            rating = product.get("rating_average", 0)
            review_count = product.get("review_count", 0)
            sold_count = product.get("sold", 0)
            
            # Nếu không có review thì tạo 1 dòng không có review
            reviews = get_reviews(product_id)
            if not reviews:
                reviews = [{"comment_text": "", "comment_rating": 0}]
            
            product_url = f"https://tiki.vn/p/{product_id}"
            
            for review in reviews:
                all_products.append({
                    "product_id": product_id,
                    "product_name": product_name,
                    "price": price,
                    "rating": rating,
                    "review_count": review_count,
                    "sold_count": sold_count,
                    "comment_text": review["comment_text"],
                    "comment_rating": review["comment_rating"],
                    "product_url": product_url
                })
            
            time.sleep(random.uniform(0.5, 1.0))
        
        time.sleep(random.uniform(1.0, 2.0))
    
    return all_products, end_page

def combine_and_deduplicate(old_df, new_df):
    """Gộp dữ liệu cũ và mới, chống trùng."""
    # Gộp
    combined_df = pd.concat([old_df, new_df], ignore_index=True)
    
    print(f"\n🔄 Gộp dữ liệu:")
    print(f"   Dữ liệu cũ: {len(old_df)} dòng")
    print(f"   Dữ liệu mới: {len(new_df)} dòng")
    print(f"   Trước chống trùng: {len(combined_df)} dòng")
    
    # Chống trùng
    combined_df['_dedupe_key'] = combined_df.apply(
        lambda row: f"{row['product_id']}_{row['comment_text']}"
        if row['comment_text'].strip()
        else f"{row['product_id']}_{row['product_name']}_{row['price']}_{row['product_url']}",
        axis=1
    )
    
    duplicates_count = len(combined_df) - len(combined_df.drop_duplicates(subset=['_dedupe_key']))
    combined_df = combined_df.drop_duplicates(subset=['_dedupe_key'], keep='first')
    combined_df = combined_df.drop(columns=['_dedupe_key'])
    
    print(f"   Sau chống trùng: {len(combined_df)} dòng")
    print(f"   Bị loại do trùng: {duplicates_count} dòng")
    
    return combined_df

def save_data(df, last_page):
    """Lưu dữ liệu và cập nhật trạng thái."""
    # Backup file cũ
    if os.path.exists(DATA_PATH):
        try:
            os.makedirs("data", exist_ok=True)
            df_backup = pd.read_csv(DATA_PATH)
            df_backup.to_csv(BACKUP_PATH, index=False)
            print(f"✅ Backup: {BACKUP_PATH}")
        except:
            pass
    
    # Lưu file mới
    os.makedirs(os.path.dirname(DATA_PATH) or ".", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Lưu: {DATA_PATH}")
    
    # Cập nhật trạng thái
    save_crawl_state(last_page)
    print(f"✅ Cập nhật crawl_state.json: last_crawled_page = {last_page}")

def print_statistics(old_len, new_len, combined_len, duplicates_count, df_final):
    """In thống kê."""
    print("\n" + "="*60)
    print("📊 THỐNG KÊ CRAWL")
    print("="*60)
    print(f"Dữ liệu cũ: {old_len} dòng")
    print(f"Dữ liệu mới crawl: {new_len} dòng")
    print(f"Trước chống trùng: {combined_len} dòng")
    print(f"Sau chống trùng: {len(df_final)} dòng")
    print(f"Loại do trùng: {duplicates_count} dòng")
    print(f"Số sản phẩm duy nhất: {df_final['product_id'].nunique()}")
    print(f"Có bình luận: {len(df_final[df_final['comment_text'].str.strip() != ''])} dòng")
    print(f"Không bình luận: {len(df_final[df_final['comment_text'].str.strip() == ''])} dòng")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Crawl sách từ Tiki")
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGES_PER_RUN, help="Số trang cần crawl")
    parser.add_argument("--limit", type=int, default=PRODUCTS_PER_PAGE, help="Số sản phẩm mỗi trang")
    parser.add_argument("--reset-state", action="store_true", help="Reset trạng thái crawl")
    args = parser.parse_args()
    
    print(f"\n🚀 Tiki Book Intelligence - Crawler")
    print(f"   Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Reset state nếu cần
    if args.reset_state:
        save_crawl_state(0)
        print(f"✅ Reset crawl_state.json")
    
    # Tính toán trang crawl
    last_page = load_crawl_state()
    start_page = last_page + 1
    end_page = start_page + args.pages - 1
    
    print(f"\n📄 Cấu hình:")
    print(f"   Sản phẩm/trang: {args.limit}")
    print(f"   Trang cần crawl: {args.pages}")
    print(f"   Trang: {start_page} → {end_page}")
    
    # Crawl dữ liệu
    new_products, final_page = crawl_products(start_page, end_page, args.limit)
    new_df = pd.DataFrame(new_products)
    
    if len(new_df) == 0:
        print("❌ Không crawl được dữ liệu nào")
        return
    
    # Đọc dữ liệu cũ
    if os.path.exists(DATA_PATH):
        old_df = pd.read_csv(DATA_PATH)
    else:
        old_df = pd.DataFrame()
    
    old_len = len(old_df)
    new_len = len(new_df)
    
    # Gộp và chống trùng
    combined_df = combine_and_deduplicate(old_df, new_df)
    combined_len = old_len + new_len
    duplicates_count = combined_len - len(combined_df)
    
    # Lưu dữ liệu
    save_data(combined_df, final_page)
    
    # In thống kê
    print_statistics(old_len, new_len, combined_len, duplicates_count, combined_df)
    
    print(f"\n✅ Hoàn thành! Crawl lần tới sẽ bắt đầu từ trang {final_page + 1}")

if __name__ == "__main__":
    main()
