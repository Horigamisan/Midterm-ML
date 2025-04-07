import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

from visualization.plot import ( plot_feature_importance, plot_learning_curve )
from training.compare_models import compare_models
from training.evaluate import evaluate_model


def train_models(X_train, X_test, y_train, y_test, type_model='before_overfitting'):
    """
    Tao va huan luyen cac mo hinh phan loai cho du lieu da duoc lam sach.
    
    Parameters:
    X_train: DataFrame chua cac dac trung cua tap huan luyen
    X_test: DataFrame chua cac dac trung cua tap kiem tra
    y_train: Series chua nhan cua tap huan luyen
    y_test: Series chua nhan cua tap kiem tra
    type_model: before hoac after overfitting, chi dinh loai mo hinh can huan luyen
    
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

    # Overfitting models - Cấu hình để dễ bị Overfitting hơn
    # Các mô hình được cấu hình để dễ bị overfitting
    over_models = {
        'Logistic Regression': LogisticRegression(
            max_iter=3000,       # Đảm bảo đủ số vòng lặp
            C=10000,             # Giá trị C rất lớn làm giảm tác dụng của regularization, giúp mô hình học thuộc lòng dữ liệu training
            solver='liblinear',
            class_weight='balanced',
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=None,      # Không giới hạn độ sâu của cây, dẫn đến việc mô hình tạo ra nhiều nhánh phức tạp
            min_samples_split=2, # Yêu cầu chỉ 2 mẫu để chia tách, dễ tạo ra các phân chia quá chi tiết
            min_samples_leaf=1,  # Mỗi lá có thể chứa duy nhất 1 mẫu, khiến mô hình học thuộc lòng từng trường hợp riêng lẻ
            class_weight='balanced',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,      # Số lượng cây cố định
            max_depth=None,        # Cho phép các cây có độ sâu không giới hạn, tăng khả năng học quá khớp
            min_samples_split=2,   # Chia tách với số lượng mẫu rất ít
            min_samples_leaf=1,    # Lá chứa 1 mẫu, dẫn đến sự chi tiết quá mức
            bootstrap=True,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=500,     # Số lượng cây boosting cao, mỗi cây cố gắng sửa lỗi của cây trước đó
            learning_rate=0.5,    # Learning rate lớn làm cho quá trình học diễn ra nhanh, dễ dẫn đến quá khớp
            max_depth=None,       # Không giới hạn độ sâu cho từng cây, mô hình có thể học quá chi tiết dữ liệu training
            min_samples_split=2,  # Phân chia với số lượng mẫu rất ít
            min_samples_leaf=1,   # Lá chứa 1 mẫu
            subsample=1.0,        # Sử dụng toàn bộ dữ liệu cho mỗi cây, giảm tính ngẫu nhiên
            random_state=42
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=1,        # Chỉ lấy 1 láng giềng, khiến mô hình nhạy cảm với nhiễu và các điểm dữ liệu ngoại lai
            weights='uniform',
            p=2,
            n_jobs=-1
        ),
        'Naive Bayes': GaussianNB(
            var_smoothing=1e-9    # Naive Bayes vốn ít bị overfitting, giữ nguyên tham số
        )
    }

    # Nếu số lượng feature nhỏ hơn 1000, thêm SVM với cấu hình overfitting
    if X_train.shape[1] < 1000:
        over_models['SVM'] = SVC(
            C=10000,             # C rất lớn làm yếu tác dụng của regularization
            kernel='rbf',        # Sử dụng kernel RBF cho khả năng tạo đường biên phi tuyến phức tạp
            gamma=1,             # Gamma lớn giúp mô hình tạo ra đường biên phân chia rất chi tiết
            probability=True,
            class_weight='balanced',
            random_state=42,
            max_iter=2000
        )

    
    main_models = models if type_model == 'before_overfitting' else over_models
    
    # Ket qua danh gia cua cac mo hinh
    results = []
    
    # Huan luyen va danh gia tung mo hinh
    for name, model in main_models.items():
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
    
    return main_models, results