from data.load_data import load_data
from data.clean_data import clean_data
from visualization.visualize_data import visualize_data
from data.preprocessing import preprocess_data
from training.train_models import train_models

def main():
    # 1. Tải dữ liệu
    X, y = load_data()
    print(f"So luong mau ban dau: {X.shape[0]}, so luong dac trung: {X.shape[1]}")
    
    # 2. Làm sạch dữ liệu
    X, y = clean_data(X, y)
    print(f"So luong sau khi lam sach: {X.shape[0]}, so luong dac trung: {X.shape[1]}")
   
    # 3. Trực quan hóa dữ liệu
    visualize_data(X)
   
    # 4. Chuyển đổi, chuẩn hóa và chia thành tập Train - Test
    X_train, X_test, y_train, y_test = preprocess_data(X, y, test_size=0.25, random_state=42)
    
    # 5. Huấn luyện và đánh giá các mô hình
    models, results = train_models(X_train, X_test, y_train, y_test) # Hoặc 'overfitting'
    
if __name__ == "__main__":
    main()