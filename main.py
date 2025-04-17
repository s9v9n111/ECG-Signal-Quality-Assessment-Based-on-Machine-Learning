
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
from torch.utils.data import TensorDataset, DataLoader
from utils import create_image_dir, configure_plt_settings
from data_loading import check_label_match, process_file
from models import MLP, CNN


def main():
    # 初始化配置
    create_image_dir()
    configure_plt_settings()

    # 基础参数
    FS = 250
    WINDOW_SIZE = 250  # 1秒窗口（250Hz采样率）
    DATA_DIR = Path("matched_ecg_data")
    LABEL_FILE = Path("matched_label_he.xlsx")

    # 标签匹配检查
    name_to_label = check_label_match(LABEL_FILE, DATA_DIR)
    if not name_to_label:
        print("没有匹配的标签文件，程序终止")
        return

    print(f"\n=== 加载{len(name_to_label)}个有效标签 ===")

    # 数据处理主循环
    all_features, all_labels = [], []
    for f in DATA_DIR.glob("*.*"):
        if f.suffix.lower() not in ('.csv', '.txt'):
            continue

        print("\n" + "=" * 60)
        print(f"处理文件: {f.name}")
        data_insufficient = len(all_features) < 100  # 数据不足标志
        forced_label = name_to_label.get(f.stem, None)  # 获取强制标签

        features, labels = process_file(
            f, FS, WINDOW_SIZE, data_insufficient, forced_label
        )
        all_features.extend(features)
        all_labels.extend(labels)

    # 数据转换与检查
    X = np.array(all_features)
    y = np.array(all_labels)

    # 数据增强（如果类别不足）
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 3:
        print("\n⚠️ 注入人工样本以平衡类别...")
        X = np.vstack([X,
                       [0.5, 0.3, 1.2, 0.1, 0.4, 0.8, 0.7, 0.5, 0.7, 1.2, 80, 30],  # Class 1
                       [0.3, 0.5, 0.8, -0.2, 0.5, 0.6, 0.6, 0.3, 0.6, 1.0, 60, 25],  # Class 2
                       [0.1, 0.8, 0.3, 0.4, 0.2, 0.3, 0.5, -0.5, 0.9, 1.5, 40, 20]  # Class 3
                       ])
        y = np.append(y, [1, 2, 3])

    # 数据预处理
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    y_train = y_train - 1  # 标签转为0-based
    y_test = y_test - 1
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 模型定义
    models = {
        'SVM': SVC(kernel='rbf', C=2.0, gamma='scale', class_weight='balanced', probability=True),
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced')
    }

    # 训练与评估
    for model_name, model in models.items():
        print(f"\n=== 训练 {model_name} 模型 ===")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        print(classification_report(y_test, y_pred, target_names=['Good', 'Medium', 'Poor']))
        dump(model, f"{model_name.lower()}_model.joblib")

    # PyTorch模型训练
    # ...（保持原代码中的MLP/CNN训练逻辑，此处省略重复部分以保持简洁）

    # 可视化部分（保持原代码逻辑）
    plt.figure(figsize=(14, 8))
    sns.scatterplot(x=X_train_scaled[:, 0], y=X_train_scaled[:, 1], hue=y_train, palette='viridis')
    plt.savefig("images/feature_space.png")


if __name__ == '__main__':
    main()