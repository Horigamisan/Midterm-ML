import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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