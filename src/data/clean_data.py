import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.impute import SimpleImputer

def clean_data(X, y):
    print("So luong ban ghi ban dau:", X.shape[0])

    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"Cot so: {len(numeric_cols)}, Cot phan loai: {len(categorical_cols)}")

    X_processed = X.copy()

    imputer_num = SimpleImputer(strategy='median')
    if numeric_cols:
        X_processed[numeric_cols] = imputer_num.fit_transform(X_processed[numeric_cols])

    imputer_cat = SimpleImputer(strategy='most_frequent')

    if categorical_cols.any(): # Kiểm tra xem có cột phân loại không
        X_processed[categorical_cols] = X_processed[categorical_cols].replace('?', np.nan)
        X_processed[categorical_cols] = imputer_cat.fit_transform(X_processed[categorical_cols])

    y_aligned = y.loc[X_processed.index].copy()

    # --- Xử lý trùng lặp ---
    duplicate_mask = X_processed.duplicated()
    duplicate_rows = duplicate_mask.sum()
    print(f"So luong ban ghi trung lap: {duplicate_rows}")

    if duplicate_rows > 0:
        # Lấy index của các bản ghi KHÔNG trùng lặp từ X_processed
        non_duplicate_indices = X_processed[~duplicate_mask].index
        # Loại bỏ bản ghi trùng lặp trong X_processed
        X_processed = X_processed.loc[non_duplicate_indices]
        # Cập nhật y tương ứng bằng index đã lọc
        y_cleaned = y_aligned.loc[non_duplicate_indices]
        print(f"Da loai bo {duplicate_rows} ban ghi trung lap. Con lai: {X_processed.shape[0]}")
    else:
        y_cleaned = y_aligned # Không có trùng lặp thì y_cleaned là y_aligned


    for col in numeric_cols:
         if col in X_processed.columns: # Đảm bảo cột vẫn tồn tại
            Q1 = X_processed[col].quantile(0.25)
            Q3 = X_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            X_processed[col] = X_processed[col].clip(lower_bound, upper_bound)


    le = LabelEncoder()
    # Kiểm tra xem y_cleaned là Series hay DataFrame
    if isinstance(y_cleaned, pd.DataFrame):
        # Nếu là DataFrame, lấy cột đầu tiên (hoặc cột target cụ thể nếu biết tên)
        y_values = y_cleaned.iloc[:, 0]
    else: # Nếu là Series
        y_values = y_cleaned

    y_encoded = le.fit_transform(y_values)
    print(f"  Cac lop cua y: {le.classes_} -> duoc ma hoa thanh {np.unique(y_encoded)}")
    print(f"  So luong mau theo lop sau ma hoa: {np.unique(y_encoded, return_counts=True)}")
    # ------------------------------------

    # Reset index cho cả X và y cuối cùng (y_encoded bây giờ là numpy array)
    X_processed.reset_index(drop=True, inplace=True)

    # Kiem tra xac nhan khong con gia tri thieu trong X
    missing_after = X_processed.isnull().sum().sum()
    if missing_after > 0:
        print(f"Canh bao: Van con {missing_after} gia tri thieu trong X sau khi xu ly")
    else:
        print("Xac nhan: Khong con gia tri thieu trong X")

    print(f"So luong ban ghi sau khi xu ly: {X_processed.shape[0]}")

    return X_processed, y_encoded