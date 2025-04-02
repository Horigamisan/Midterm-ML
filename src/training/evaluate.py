import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)

def evaluate_model(model, X_test, y_test, model_name):
    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # In kết quả
    print(f"\n===== Ket qua danh gia cho {model_name} =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nBao cao phan loai chi tiet:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    
    # Thêm ROC curve nếu mô hình hỗ trợ predict_proba
    if hasattr(model, "predict_proba"):
        try:
            # Kiểm tra xem dữ liệu có phải là binary classification
            if len(np.unique(y_test)) == 2:
                # Dự đoán xác suất của lớp dương tính
                y_prob = model.predict_proba(X_test)[:, 1]
                
                # Tính ROC curve
                # fpr, tpr, _ = roc_curve(y_test, y_prob)
                fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
                roc_auc = auc(fpr, tpr)
                
                # Vẽ ROC curve
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {model_name}')
                plt.legend(loc="lower right")
                plt.show()
            else:
                print("ROC curve chi ap dung cho bai toan phan lop. Bo qua...")
        except Exception as e:
            print(f"Khong the tinh ROC curve cho mo hinh nay: {str(e)}")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }