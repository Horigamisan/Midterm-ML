import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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