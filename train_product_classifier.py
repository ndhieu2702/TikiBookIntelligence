"""
Huấn luyện mô hình dự đoán hiệu quả sản phẩm.
- So sánh Decision Tree, Random Forest, XGBoost.
- Chọn mô hình tốt nhất dựa trên F1-score weighted.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
DATA_PATH = "data/tiki_books_labeled.csv"
OUTPUT_CSV = "data/model_comparison.csv"
REPORT_CSV = "data/classification_report.csv"
CM_PATH = "images/confusion_matrix.png"
FI_PATH = "images/feature_importance.png"
MODEL_PATH = "models/product_performance_model.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

FEATURES = ['price', 'rating', 'review_count', 'sold_count', 'comment_count',
            'avg_comment_rating', 'positive_ratio', 'neutral_ratio', 
            'negative_ratio', 'estimated_revenue']
TARGET = 'product_label'

# ==================== UTILITY FUNCTIONS ====================
def load_data():
    """Đọc dữ liệu đã gán nhãn."""
    if not os.path.exists(DATA_PATH):
        print(f"❌ File {DATA_PATH} không tồn tại")
        print(f"   Vui lòng chạy: python label_products.py trước")
        exit(1)
    
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Đã đọc {len(df)} sản phẩm từ {DATA_PATH}")
    return df

def prepare_data(df):
    """Chuẩn bị dữ liệu cho huấn luyện."""
    print("\n📋 Chuẩn bị dữ liệu...")
    
    # Kiểm tra cột
    for col in FEATURES + [TARGET]:
        if col not in df.columns:
            print(f"❌ Cột {col} không tồn tại")
            exit(1)
    
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    
    # Xử lý NaN
    X = X.fillna(0)
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"   ✓ Features: {len(FEATURES)} cột")
    print(f"   ✓ Target: {len(le.classes_)} lớp")
    for i, label in enumerate(le.classes_):
        count = sum(y_encoded == i)
        print(f"      - {label}: {count}")
    
    return X, y_encoded, y, le

def train_test_split_data(X, y_encoded):
    """Chia dữ liệu train/test."""
    print("\n📊 Chia dữ liệu train/test...")
    
    # Kiểm tra stratify
    use_stratify = True
    for label in np.unique(y_encoded):
        if sum(y_encoded == label) < 2:
            use_stratify = False
            print(f"   ⚠️  Cảnh báo: Lớp {label} có quá ít mẫu, không dùng stratify")
            break
    
    if use_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
    
    print(f"   ✓ Train: {len(X_train)} mẫu")
    print(f"   ✓ Test: {len(X_test)} mẫu")
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, X_test, y_train, y_test, le):
    """Huấn luyện các mô hình."""
    print("\n🤖 Huấn luyện các mô hình...")
    
    models_data = []
    trained_models = {}
    
    # 1. Decision Tree
    print("\n   Decision Tree Classifier...")
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_metrics = evaluate_model(dt_pred, y_test, "Decision Tree Classifier")
    models_data.append(dt_metrics)
    trained_models["Decision Tree"] = dt_model
    
    # 2. Random Forest
    print("\n   Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_metrics = evaluate_model(rf_pred, y_test, "Random Forest Classifier")
    models_data.append(rf_metrics)
    trained_models["Random Forest"] = rf_model
    
    # 3. XGBoost
    try:
        import xgboost as xgb
        print("\n   XGBoost Classifier...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.1,
            random_state=42,
            tree_method='hist'
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_metrics = evaluate_model(xgb_pred, y_test, "XGBoost Classifier")
        models_data.append(xgb_metrics)
        trained_models["XGBoost"] = xgb_model
    except ImportError:
        print("\n   ⚠️  Chưa cài xgboost. Hãy chạy: python -m pip install xgboost")
    
    return models_data, trained_models, y_test

def evaluate_model(y_pred, y_test, model_name):
    """Đánh giá mô hình."""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"      Accuracy: {accuracy:.4f}")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall: {recall:.4f}")
    print(f"      F1-score: {f1:.4f}")
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }

def save_model_comparison(models_data):
    """Lưu bảng so sánh mô hình."""
    df_comparison = pd.DataFrame(models_data)
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    df_comparison.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Lưu: {OUTPUT_CSV}")
    return df_comparison

def save_classification_report(best_model, X_test, y_test, le):
    """Lưu classification report."""
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, 
                                    target_names=le.classes_,
                                    output_dict=True)
    
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(REPORT_CSV)
    print(f"✅ Lưu: {REPORT_CSV}")

def plot_confusion_matrix(best_model, X_test, y_test, le):
    """Vẽ confusion matrix."""
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues', aspect='auto')
    plt.title('Confusion Matrix - Mô hình tốt nhất', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(range(len(le.classes_)), le.classes_, rotation=45, ha='right')
    plt.yticks(range(len(le.classes_)), le.classes_)
    
    # Thêm giá trị vào từng ô
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', 
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    plt.colorbar(label='Count')
    plt.tight_layout()
    os.makedirs(os.path.dirname(CM_PATH) or ".", exist_ok=True)
    plt.savefig(CM_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Lưu: {CM_PATH}")

def plot_feature_importance(best_model, model_name):
    """Vẽ feature importance."""
    if not hasattr(best_model, 'feature_importances_'):
        print(f"⚠️  Mô hình {model_name} không có feature_importances_")
        return
    
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance - Mô hình tốt nhất', fontsize=14, fontweight='bold')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [FEATURES[i] for i in indices], rotation=45, ha='right')
    plt.ylabel('Importance', fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(FI_PATH) or ".", exist_ok=True)
    plt.savefig(FI_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Lưu: {FI_PATH}")

def save_best_model(best_model, le):
    """Lưu mô hình tốt nhất."""
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"✅ Lưu: {MODEL_PATH}")
    print(f"✅ Lưu: {LABEL_ENCODER_PATH}")

def main():
    print(f"\n🚀 Tiki Book Intelligence - Huấn luyện mô hình")
    print(f"   Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Đọc dữ liệu
    df = load_data()
    X, y_encoded, y, le = prepare_data(df)
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split_data(X, y_encoded)
    
    # Huấn luyện mô hình
    models_data, trained_models, y_test = train_models(X_train, X_test, y_train, y_test, le)
    
    # Lưu bảng so sánh
    df_comparison = save_model_comparison(models_data)
    
    # Chọn mô hình tốt nhất
    best_idx = df_comparison['F1-score'].idxmax()
    best_model_name = df_comparison.loc[best_idx, 'Model']
    
    if "Decision Tree" in best_model_name:
        best_model = trained_models["Decision Tree"]
    elif "Random Forest" in best_model_name:
        best_model = trained_models["Random Forest"]
    elif "XGBoost" in best_model_name:
        best_model = trained_models["XGBoost"]
    else:
        best_model = list(trained_models.values())[0]
        best_model_name = list(trained_models.keys())[0]
    
    best_metrics = df_comparison.loc[best_idx]
    
    print(f"\n🏆 Mô hình tốt nhất: {best_model_name}")
    print(f"   F1-score: {best_metrics['F1-score']:.4f}")
    
    # Lưu classification report
    save_classification_report(best_model, X_test, y_test, le)
    
    # Vẽ confusion matrix
    plot_confusion_matrix(best_model, X_test, y_test, le)
    
    # Vẽ feature importance
    plot_feature_importance(best_model, best_model_name)
    
    # Lưu mô hình
    save_best_model(best_model, le)
    
    print(f"\n" + "="*60)
    print("✅ Hoàn thành huấn luyện mô hình!")
    print("="*60)

if __name__ == "__main__":
    main()
