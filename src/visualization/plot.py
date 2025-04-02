import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_feature_importance(model, X, model_name):
    # Kiểm tra xem mô hình có hỗ trợ feature_importances_ không
    if hasattr(model, 'feature_importances_'):
        # Lấy trọng số của từng đặc trưng
        importances = model.feature_importances_
        
        # Tạo DataFrame để lưu trữ tên đặc trưng và trọng số tương ứng
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        })
        
        # Sắp xếp theo trọng số giảm dần
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # Vẽ biểu đồ cột
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        plt.show()
    elif hasattr(model, 'coef_'):
        # Đối với các mô hình tuyến tính
        coefficients = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        
        # Tạo DataFrame để lưu trữ tên đặc trưng và hệ số tương ứng
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'coefficient': np.abs(coefficients)  # Lấy giá trị tuyệt đối của hệ số
        })
        
        # Sắp xếp theo hệ số giảm dần
        feature_importance = feature_importance.sort_values('coefficient', ascending=False)
        
        # Vẽ biểu đồ cột
        plt.figure(figsize=(12, 8))
        sns.barplot(x='coefficient', y='feature', data=feature_importance)
        plt.title(f'Feature Coefficients - {model_name}')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Mo hinh {model_name} khong ho tro hien thi feature importance")



def plot_learning_curve(model, X, y, model_name, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Ve learning curve de danh gia hieu suat cua mo hinh theo kich thuoc tap huan luyen.
    
    Parameters:
    model: Mo hinh can danh gia
    X: Du lieu dac trung
    y: Nhan
    model_name: Ten mo hinh de hien thi
    cv: So luong fold trong cross-validation
    train_sizes: Kich thuoc tuong doi cua tap huan luyen
    """
    from sklearn.model_selection import learning_curve
    
    # Sampling neu du lieu qua lon de tang toc do
    if X.shape[0] > 10000:
        sample_indices = np.random.choice(X.shape[0], 10000, replace=False)
        X_sampled = X.iloc[sample_indices] if hasattr(X, 'iloc') else X[sample_indices]
        y_sampled = y[sample_indices]
    else:
        X_sampled, y_sampled = X, y
    
    try:
        # Tinh learning curve
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X_sampled, y_sampled, train_sizes=train_sizes, cv=cv, n_jobs=-1, scoring="accuracy"
        )
        
        # Tinh gia tri trung binh va do lech chuan
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Ve bieu do
        plt.figure(figsize=(10, 6))
        plt.title(f"Learning Curve - {model_name}")
        plt.xlabel("Training examples")
        plt.ylabel("Accuracy")
        plt.grid()
        
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
        plt.plot(train_sizes_abs, train_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes_abs, test_mean, 'o-', color="g", label="Cross-validation score")
        
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Loi khi ve learning curve cho {model_name}: {str(e)}")