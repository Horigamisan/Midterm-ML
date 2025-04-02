import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Hàm tải và trả về dữ liệu
def load_data():
    adult = fetch_ucirepo(id=2) 
    X = adult.data.features 
    y = adult.data.targets 
    return X, y

# Hàm làm sạch dữ liệu
def clean_data(X, y):
    # 1. Kiểm tra và xử lý giá trị thiếu
    missing_values = X.isnull().sum()
    print("Gia tri thieu trong cac cot:\n", missing_values[missing_values > 0])
    # Thay thế giá trị thiếu bằng giá trị trung bình (áp dụng cho các cột số)
    X.fillna(X.mean(numeric_only=True), inplace=True)
    
    # 2. Chuyển đổi kiểu dữ liệu của các biến phân loại (nếu có)
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print("Cac cot phan loai: ", list(categorical_cols))
        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col])
    
    # 3. Loại bỏ các bản ghi trùng lặp
    duplicate_rows = X.duplicated().sum()
    print(f"So luong ban ghi trung lap: {duplicate_rows}")
    X.drop_duplicates(inplace=True)
    
    # 4. Xử lý ngoại lai (Outliers) cho các cột số bằng phương pháp IQR
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        before = X.shape[0]
        X = X[(X[col] >= lower_bound) & (X[col] <= upper_bound)]
        after = X.shape[0]
        print(f"{col}: Loai bo {before - after} ngoai lai")
    
     # Cập nhật lại index của X và y
    X.reset_index(drop=True, inplace=True)
    y = y.loc[X.index].reset_index(drop=True)
    return X, y

def visualize_data(X):
    sns.set_theme(style="whitegrid")
    
    # Histogram cho các đặc trưng
    X.hist(bins=15, figsize=(15, 10))
    plt.tight_layout()
    plt.show()
    
    # Ma trận tương quan và heatmap
    correlation_matrix = X.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Ma tran tuong quan giua cac dac trung')
    plt.show()


# Hàm chuyển đổi và chuẩn hóa dữ liệu, sau đó chia Train-Test
def preprocess_data(X, y, test_size=0.2, random_state=42):
    # Nếu X có các biến phân loại dạng số chưa chuẩn, có thể chuyển đổi kiểu dữ liệu (đã làm ở clean_data)
    # Chuẩn hóa dữ liệu: Sử dụng StandardScaler cho các cột số
    scaler = StandardScaler()
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Tap huan luyen: {X_train.shape}, Tap kiem tra: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# Hàm chính để thực hiện các bước
def main():
    # 1. Tải dữ liệu
    X, y = load_data()
    print(f"So luong mau ban dau: {X.shape[0]}, so luong dac trung: {X.shape[1]}")

    # 2. Làm sạch dữ liệu
    X, y = clean_data(X, y)
    print(f"So luong sau khi lam sach : {X.shape[0]}, so luong dac trung: {X.shape[1]}")
    
    # 3. Trực quan hóa dữ liệu
    visualize_data(X)
    
    # 4. Chuyển đổi, chuẩn hóa và chia thành tập Train - Test
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
if __name__ == "__main__":
    main()
