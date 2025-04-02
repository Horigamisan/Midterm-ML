import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Import các thuật toán machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

# Import các metric đánh giá
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.model_selection import cross_val_score

# Hàm tải và trả về dữ liệu
def load_data():
    # adult = fetch_ucirepo(id=320)
    adult = fetch_ucirepo(id=27)
    X = adult.data.features
    y = adult.data.targets

    print(X.info())
    return X, y

# Hàm làm sạch dữ liệu
# Thêm import ở đầu file nếu chưa có
from sklearn.preprocessing import LabelEncoder
import numpy as np # Đảm bảo numpy được import

# Hàm clean_data đã được sửa đổi
def clean_data(X, y):
    print("So luong ban ghi ban dau:", X.shape[0])

    # Luu index truoc khi xu ly
    original_indices = X.index.copy() # Lưu index gốc của X ban đầu

    # --- Phần xử lý X ---
    # (Giữ nguyên phần xử lý missing values, phân loại cột, SimpleImputer cho X)
    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Cot so: {len(numeric_cols)}, Cot phan loai: {len(categorical_cols)}")
    X_processed = X.copy()
    for col in categorical_cols:
        X_processed[col] = X_processed[col].replace('?', np.nan)
        # Bỏ print ở đây để tránh rối
        # if X_processed[col].isnull().sum() > 0:
        #     print(f"Da chuyen doi dau '?' thanh NaN trong cot {col}: {X_processed[col].isnull().sum()} gia tri")

    imputer_num = SimpleImputer(strategy='median')
    if numeric_cols:
        X_processed[numeric_cols] = imputer_num.fit_transform(X_processed[numeric_cols])

    # Sửa lại cách dùng imputer cho cột phân loại
    imputer_cat = SimpleImputer(strategy='most_frequent')
    if categorical_cols.any(): # Kiểm tra xem có cột phân loại không
        X_processed[categorical_cols] = imputer_cat.fit_transform(X_processed[categorical_cols])


    # --- Cập nhật y theo X TRƯỚC KHI loại bỏ trùng lặp ---
    # Đảm bảo y có cùng index với X_processed tại thời điểm này
    # Quan trọng: Nếu y là DataFrame, cần đảm bảo chỉ lấy cột target
    y_target_column = y.columns[0] if isinstance(y, pd.DataFrame) else None # Lấy tên cột target nếu y là DataFrame
    y_aligned = y.loc[X_processed.index].copy() # Align y với X trước khi xử lý duplicate

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

    # --- Xử lý ngoại lệ cho X ---
    # (Giữ nguyên phần xử lý IQR clipping cho X)
    for col in numeric_cols:
         if col in X_processed.columns: # Đảm bảo cột vẫn tồn tại
            Q1 = X_processed[col].quantile(0.25)
            Q3 = X_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            X_processed[col] = X_processed[col].clip(lower_bound, upper_bound)


    # --- *** THÊM BƯỚC MÃ HÓA Y *** ---
    print("\nMa hoa bien muc tieu y...")
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
    # y_encoded không cần reset index vì nó là numpy array

    # Kiem tra xac nhan khong con gia tri thieu trong X
    missing_after = X_processed.isnull().sum().sum()
    if missing_after > 0:
        print(f"Canh bao: Van con {missing_after} gia tri thieu trong X sau khi xu ly")
    else:
        print("Xac nhan: Khong con gia tri thieu trong X")

    print(f"So luong ban ghi sau khi xu ly: {X_processed.shape[0]}")
    # print(f"So luong bien sau khi xu ly: {X_processed.shape[1]}") # Bỏ dòng này vì X đã có OneHot

    # TRẢ VỀ X ĐÃ XỬ LÝ VÀ Y ĐÃ MÃ HÓA
    return X_processed, y_encoded

def visualize_data(X, original_X=None):
    """
    Ham truc quan hoa du lieu phu hop voi du lieu da duoc xu ly tu ham clean_data
    
    Parameters:
    X: DataFrame da xu ly
    original_X: DataFrame goc (truoc khi xu ly) - optional
    """
    sns.set_theme(style="whitegrid")
    
    # So sanh phan phoi du lieu truoc va sau khi xu ly (neu co du lieu goc)
    if original_X is not None:
        # Chi chon cac cot so trong du lieu goc
        numeric_cols_original = original_X.select_dtypes(include=[np.number]).columns
        common_cols = [col for col in numeric_cols_original if col in X.columns]
        
        if common_cols:
            n_cols = len(common_cols)
            if n_cols > 0:
                fig, axes = plt.subplots(n_cols, 2, figsize=(15, 4*n_cols))
                
                # Xu ly truong hop chi co 1 cot (axes khong phai mang 2D)
                if n_cols == 1:
                    axes = np.array([axes])  # Chuyen thanh mang 2D
                
                fig.suptitle('So sanh phan phoi truoc va sau khi xu ly', fontsize=16)
                
                for i, col in enumerate(common_cols):
                    # Phan phoi truoc khi xu ly
                    if original_X[col].nunique() > 0:  # Kiem tra co du lieu khong
                        sns.histplot(original_X[col].dropna(), kde=True, ax=axes[i, 0])
                        axes[i, 0].set_title(f'{col} - Truoc khi xu ly')
                    
                    # Phan phoi sau khi xu ly
                    if X[col].nunique() > 0:  # Kiem tra co du lieu khong
                        sns.histplot(X[col].dropna(), kde=True, ax=axes[i, 1])
                        axes[i, 1].set_title(f'{col} - Sau khi xu ly')
                
                plt.tight_layout()
                plt.subplots_adjust(top=0.95)
                plt.show()
    
    # Kiem tra cac bien duoc ma hoa one-hot
    encoded_cols = [col for col in X.columns if '_' in col]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    standard_numeric_cols = [col for col in numeric_cols if col not in encoded_cols]
    
    # Histogram cho cac dac trung so tieu chuan (khong bao gom bien duoc ma hoa one-hot)
    if len(standard_numeric_cols) > 0:
        n_cols = len(standard_numeric_cols)
        
        if n_cols <= 9:  # Xu ly khi co it cot
            rows = int(np.ceil(n_cols / 3))
            fig, axes = plt.subplots(rows, min(3, n_cols), figsize=(15, 4*rows))
            fig.suptitle('Phan phoi cac dac trung so', fontsize=16)
            
            # Chuyen doi axes thanh mang 1D de de lam viec
            axes = np.array(axes).flatten()
            
            for i, col in enumerate(standard_numeric_cols):
                if i < len(axes):  # Tranh truong hop vuot qua so luong subplot
                    sns.histplot(X[col], kde=True, ax=axes[i])
                    axes[i].set_title(f'{col}')
            
            # An cac subplot khong su dung
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
                
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()
        else:
            # Neu co qua nhieu cot, su dung ham hist cua pandas
            X[standard_numeric_cols].hist(bins=15, figsize=(15, 10))
            plt.suptitle('Phan phoi cac dac trung so', fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            plt.show()
    
    # Bieu do dem cho cac bien duoc ma hoa one-hot (neu co)
    if len(encoded_cols) > 0:
        # Nhom cac cot theo tien to (ten bien goc)
        prefixes = set()
        for col in encoded_cols:
            if '_' in col:
                prefix = col.split('_')[0]
                prefixes.add(prefix)
        
        for prefix in prefixes:
            cols = [col for col in encoded_cols if col.startswith(prefix + '_')]
            if len(cols) > 1 and len(cols) <= 15:  # Chi hien thi neu khong qua nhieu gia tri
                sum_values = X[cols].sum()
                if sum_values.sum() > 0:  # Dam bao co gia tri de ve
                    plt.figure(figsize=(10, 6))
                    sum_values.sort_values(ascending=False).plot(kind='bar')
                    plt.title(f'Tan suat cac gia tri cua bien {prefix}')
                    plt.ylabel('So luong')
                    plt.tight_layout()
                    plt.show()
    
    # Boxplot cho cac dac trung so (de phat hien bat thuong sau khi xu ly)
    if len(standard_numeric_cols) > 0:
        if len(standard_numeric_cols) <= 20:  # Chi ve khi so luong bien hop ly
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=X[standard_numeric_cols])
            plt.title('Boxplot cac dac trung so sau xu ly')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()
    
    # Ma tran tuong quan va heatmap
    if len(numeric_cols) > 1:  # Chi tao ma tran tuong quan neu co it nhat 2 bien so
        # Kiem tra xem cac cot co gia tri nan khong
        corr_data = X[numeric_cols].copy()
        if corr_data.isna().sum().sum() > 0:
            print("Co gia tri NaN trong du lieu. Dang bo qua cho ma tran tuong quan...")
            corr_data = corr_data.fillna(0)  # Thay the NaN bang 0 de tinh tuong quan
            
        try:
            correlation_matrix = corr_data.corr()
            
            # Kiem tra kich thuoc ma tran
            if correlation_matrix.shape[0] <= 20:  # Chi hien thi day du khi ma tran khong qua lon
                plt.figure(figsize=(12, 10))
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Tao mask de hien thi nua ma tran
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', mask=mask, cbar=True)
                plt.title('Ma tran tuong quan giua cac dac trung')
                plt.tight_layout()
                plt.show()
            else:
                # Neu ma tran qua lon, chi hien thi cac tuong quan cao
                high_corr = correlation_matrix.abs().unstack()
                high_corr = high_corr[high_corr > 0.5]
                high_corr = high_corr[high_corr < 1.0]  # Loai bo tuong quan cua bien voi chinh no
                high_corr = high_corr.sort_values(ascending=False)
                
                if len(high_corr) > 0:
                    print("Cac cap bien co tuong quan cao (|r| > 0.5):")
                    for idx, val in high_corr.items():
                        print(f"{idx[0]} - {idx[1]}: {val:.2f}")
        except Exception as e:
            print(f"Loi khi tao ma tran tuong quan: {e}")
    
    # Phat hien cac moi quan he phi tuyen thong qua bieu do pairplot
    if 1 < len(standard_numeric_cols) <= 5:  # Chi tao pairplot khi so luong bien hop ly
        try:
            # Kiem tra xem co qua nhieu diem du lieu khong
            if X.shape[0] > 1000:
                # Su dung mau du lieu neu co qua nhieu diem
                sample_size = min(1000, X.shape[0])
                sample_df = X.sample(sample_size)
                sns.pairplot(sample_df[standard_numeric_cols])
                plt.suptitle(f'Moi quan he giua cac dac trung so (mau {sample_size} diem)', y=1.02)
            else:
                sns.pairplot(X[standard_numeric_cols])
                plt.suptitle('Moi quan he giua cac dac trung so', y=1.02)
            plt.show()
        except Exception as e:
            print(f"Loi khi tao pairplot: {e}")
    
    # Thong ke mo ta cho cac dac trung so
    if len(numeric_cols) > 0:
        print("\nThong ke mo ta cho cac dac trung so:")
        desc_stats = X[numeric_cols].describe().T
        display_cols = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        available_cols = [col for col in display_cols if col in desc_stats.columns]
        print(desc_stats[available_cols])
    
    return

# Hàm chuyển đổi và chuẩn hóa dữ liệu, sau đó chia Train-Test
def preprocess_data(X, y, test_size=0.25, random_state=42):
    
    print("\n--- Starting Data Preprocessing (Scaling, Encoding, Splitting) ---")

    # 1. Split Data FIRST to prevent leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y# Stratify for classification
    )

    # 2. Identify column types FROM THE TRAINING SET
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include='object').columns.tolist()

    # 3. Create Preprocessing Pipelines for Numeric and Categorical Features
    numeric_pipeline = Pipeline([
        ('scaler',RobustScaler())
    ])
    
    # Pipeline cho biến phân loại: chỉ cần OneHotEncoder sau khi đã impute
    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 4. Create ColumnTransformer
    # Use lists of column names
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough'
    )

    # 6. Transform BOTH training and testing data
    print("Transforming training and testing data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 7. Get feature names after transformation (important for interpretability)
    try:
        # Get feature names from the fitted transformer
        feature_names_out = preprocessor.get_feature_names_out()
        print(f"  Number of features after preprocessing: {len(feature_names_out)}")

        # Convert processed arrays back to DataFrames with proper column names
        X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names_out, index=X_train.index)
        X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names_out, index=X_test.index)

    except Exception as e:
        print(f"Warning: Could not retrieve feature names automatically. Error: {e}")
        print("Proceeding with NumPy arrays. Feature importance plots might lack labels.")
        # Keep as numpy arrays if get_feature_names_out fails (older sklearn versions?)

    print(f"\nPreprocessing finished.")
    print(f"  Processed Train set shape: X={X_train_processed.shape}, y={y_train.shape}")
    print(f"  Processed Test set shape:  X={X_test_processed.shape}, y={y_test.shape}")
    print("--- End Data Preprocessing ---")

    # Return the fitted preprocessor along with the data
    # Useful if you need to process new data later
    return X_train_processed, X_test_processed, y_train, y_test

# Hàm đánh giá mô hình với xử lý zero_division
def evaluate_model(model, X_test, y_test, model_name):
    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)
    
    # Tính các metric đánh giá với zero_division=0
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
    
    # In classification report với zero_division=0
    print("\nBao cao phan loai chi tiet:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Tạo và hiển thị confusion matrix
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

# Hệ số trọng số của các đặc trưng trong một số mô hình
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

# Cap nhat ham so sanh cac mo hinh
def compare_models(results):
    """
    So sanh hieu suat cua cac mo hinh va hien thi ket qua duoi dang bang va bieu do.
    
    Parameters:
    results: List chua ket qua danh gia cua cac mo hinh
    """
    # Tao DataFrame tu ket qua cua cac mo hinh
    df_results = pd.DataFrame(results)
    
    # Them cac cot thoi gian huan luyen va cross-validation neu co
    display_cols = ['model_name', 'accuracy', 'precision', 'recall', 'f1']
    if 'training_time' in df_results.columns:
        display_cols.append('training_time')
    if 'cv_accuracy' in df_results.columns:
        display_cols.append('cv_accuracy')
    if 'cv_std' in df_results.columns:
        display_cols.append('cv_std')
    
    # Hien thi ket qua duoi dang bang
    print("\n===== So sanh hieu suat cac mo hinh =====")
    print(df_results[display_cols])

    # --- Plotting ---
    df_plot = pd.DataFrame(results).set_index('model_name')
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    
   # --- Biểu đồ Metrics ---
    # Vấn đề có thể ở đây: Gọi plt.figure() và df_plot.plot() với figsize riêng
    # plt.figure(figsize=(12, 6)) # <-- Dòng này có thể không cần thiết
    # Thay đổi cách gọi plot này:
    ax1 = df_plot[metrics_to_plot].plot(kind='bar', figsize=(12, 7), grid=True) # Để pandas tự tạo figure và axes
    ax1.set_title('So sanh hieu suat cac mo hinh (Metrics)', fontsize=14)
    ax1.set_xlabel('Mo hinh', fontsize=12)
    ax1.set_ylabel('Diem so', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=len(metrics_to_plot)) # Điều chỉnh vị trí legend
    plt.tight_layout(rect=[0, 0, 1, 1]) # Điều chỉnh layout
    plt.show() # Hiển thị biểu đồ metrics
    
    # --- Biểu đồ Thời gian huấn luyện ---
    if 'training_time' in df_plot.columns:
            plt.figure(figsize=(10, 5))
            # --- Dòng cần sửa đổi ---
            sns.barplot(
                x=df_plot.index,        # Biến cho trục X
                y='training_time',      # Biến cho trục Y
                data=df_plot,           # Dữ liệu
                hue=df_plot.index,      # <-- Gán biến X vào HUE
                palette='viridis',      # Giữ lại bảng màu bạn muốn
                legend=False            # <-- Tắt chú giải (legend)
            )
            # -------------------------
            plt.title('So sanh thoi gian huan luyen cac mo hinh', fontsize=14)
            plt.xlabel('Mo hinh', fontsize=12)
            plt.ylabel('Thoi gian (giay)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show() # Hiển thị biểu đồ thời gian
        
    # Tim mo hinh tot nhat dua tren F1 score
    best_model_idx = df_results['f1'].idxmax()
    best_model = df_results.iloc[best_model_idx]
    
    print("\n===== Mo hinh tot nhat =====")
    print(f"Ten mo hinh: {best_model['model_name']}")
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    print(f"Precision: {best_model['precision']:.4f}")
    print(f"Recall: {best_model['recall']:.4f}")
    print(f"F1 Score: {best_model['f1']:.4f}")
    
    if 'cv_accuracy' in best_model:
        print(f"Cross-validation Accuracy: {best_model['cv_accuracy']:.4f} ± {best_model.get('cv_std', 0):.4f}")

# Hàm tạo và huấn luyện các mô hình
def train_models(X_train, X_test, y_train, y_test):
    """
    Tao va huan luyen cac mo hinh phan loai cho du lieu da duoc lam sach.
    
    Parameters:
    X_train: DataFrame chua cac dac trung cua tap huan luyen
    X_test: DataFrame chua cac dac trung cua tap kiem tra
    y_train: Series chua nhan cua tap huan luyen
    y_test: Series chua nhan cua tap kiem tra
    
    Returns:
    models: Dictionary chua cac mo hinh da duoc huan luyen
    results: List chua ket qua danh gia cua cac mo hinh
    """
    # Kiem tra xem y co can duoc chuyen doi khong
    y_train_values = y_train.values.ravel() if hasattr(y_train, 'values') else y_train
    y_test_values = y_test.values.ravel() if hasattr(y_test, 'values') else y_test
    
    # Kiem tra so luong lop phan loai
    n_classes = len(np.unique(y_train_values))
    print(f"So luong lop phan loai: {n_classes}")
    
    # Kiem tra kich thuoc cua tap du lieu
    print(f"Kich thuoc X_train: {X_train.shape}")
    print(f"Kich thuoc X_test: {X_test.shape}")
    
    # Danh sach cac mo hinh can huan luyen voi tham so da dieu chinh
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=2000,  # Tang so vong lap toi da
            # C=1.0,          # Tham so dieu chinh regularization
            C=0.05,
            solver='liblinear',  # Toi uu cho du lieu co nhieu dac trung
            class_weight='balanced',  # Can bang trong so cac lop
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=6,   # Gioi han do sau de tranh overfitting
            min_samples_split=30,  # Yeu cau it nhat 20 mau de phan tach
            min_samples_leaf=15,   # Yeu cau it nhat 10 mau o moi la
            class_weight='balanced',  # Can bang trong so cac lop
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,  # Tang so luong cay
            max_depth=15,      # Gioi han do sau toi da cua moi cay
            min_samples_split=15,  # Yeu cau it nhat 15 mau de phan tach
            min_samples_leaf=5,    # Yeu cau it nhat 5 mau o moi la
            bootstrap=True,        # Su dung bootstrap sampling
            class_weight='balanced',  # Can bang trong so cac lop
            n_jobs=-1,              # Su dung tat ca CPU cores
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,     # So luong cay boosting stages
            learning_rate=0.1,    # Toc do hoc
            max_depth=8,          # Gioi han do sau cua cay
            min_samples_split=15, # Yeu cau it nhat 15 mau de phan tach
            min_samples_leaf=5,   # Yeu cau it nhat 5 mau o moi la
            subsample=0.8,        # Su dung 80% mau cho moi cay
            random_state=42
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=7,        # So luong neighbors
            weights='distance',   # Trong so dua tren khoang cach
            p=2,                  # Su dung khoang cach Euclidean
            n_jobs=-1             # Su dung tat ca CPU cores
        ),
        'Naive Bayes': GaussianNB(
            var_smoothing=1e-9    # Phan phuong sai duoc them vao cho stability
        )
    }
    
    # Kiem tra neu so luong dac trung qua lon, them SVM voi kernel linear
    if X_train.shape[1] < 1000:  # Chi su dung SVM neu so luong dac trung vua phai
        models['SVM'] = SVC(
            C=0.1,               # Tham so regularization
            kernel='rbf',        # Kernel radial basis function
            gamma='scale',       # Tham so gamma tu dong theo so luong dac trung
            probability=True,    # Cho phep tinh xac suat
            class_weight='balanced',  # Can bang trong so cac lop
            random_state=42,
            max_iter=1000        # So vong lap toi da
        )
    
    # Ket qua danh gia cua cac mo hinh
    results = []
    
    # Huan luyen va danh gia tung mo hinh
    for name, model in models.items():
        print(f"\nDang huan luyen mo hinh: {name}...")
        try:
            # Huan luyen mo hinh voi xu ly ngoai le
            import time  # Them import time o day de dam bao co the su dung
            start_time = time.time()  # Bat dau do thoi gian
            model.fit(X_train, y_train_values)
            training_time = time.time() - start_time  # Ket thuc do thoi gian
            print(f"Thoi gian huan luyen: {training_time:.2f} giay")
            
            # Cross-validation de danh gia mo hinh
            cv_scores = cross_val_score(model, X_train, y_train_values, cv=5, scoring='accuracy', n_jobs=-1)
            print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Danh gia mo hinh tren tap kiem tra
            result = evaluate_model(model, X_test, y_test_values, name)
            result['training_time'] = training_time  # Them thoi gian huan luyen vao ket qua
            result['cv_accuracy'] = cv_scores.mean()
            result['cv_std'] = cv_scores.std()
            results.append(result)
            
            # Hien thi feature importance neu mo hinh ho tro
            plot_feature_importance(model, X_train, name)
            
            # Hien thi learning curve neu so luong mau lon
            if X_train.shape[0] > 1000:
                plot_learning_curve(model, X_train, y_train_values, name)
                
        except Exception as e:
            print(f"Loi khi huan luyen mo hinh {name}: {str(e)}")
    
    # So sanh cac mo hinh
    if results:
        compare_models(results)
    else:
        print("Khong co mo hinh nao duoc huan luyen thanh cong!")
    
    return models, results

# Them ham ve learning curve
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


# Hàm chính để thực hiện các bước
def main():
    # 1. Tải dữ liệu
    X, y = load_data()
    print(f"So luong mau ban dau: {X.shape[0]}, so luong dac trung: {X.shape[1]}")
    
    # 2. Làm sạch dữ liệu
    X, y = clean_data(X, y)
    print(f"So luong sau khi lam sach: {X.shape[0]}, so luong dac trung: {X.shape[1]}")
   
    # 3. Trực quan hóa dữ liệu
    # visualize_data(X)
   
    # 4. Chuyển đổi, chuẩn hóa và chia thành tập Train - Test
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # 5. Huấn luyện và đánh giá các mô hình
    models, results = train_models(X_train, X_test, y_train, y_test)
    
if __name__ == "__main__":
    main()