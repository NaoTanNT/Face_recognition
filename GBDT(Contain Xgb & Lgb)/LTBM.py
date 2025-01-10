import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os
from joblib import Parallel, delayed
import time

# 加载图像的函数，优化为并行加载
def load_image(file_name, image_folder="fillter_data"):
    file_name = str(file_name)
    with open(os.path.join(image_folder, file_name), 'rb') as f:
        img_data = f.read()
    img = np.frombuffer(img_data, dtype=np.uint8).reshape((128, 128))
    return img

# 使用 joblib 进行并行图像加载
def extract_features(df, image_folder="fillter_data"):
    start_time = time.time()
    X = Parallel(n_jobs=4)(delayed(load_image)(row['Filename'], image_folder) for _, row in df.iterrows())
    print(f"Data loading time: {time.time() - start_time:.2f} seconds")
    return np.array([img.flatten() for img in X])

# 提取标签
def extract_labels(df):
    y_sex = df['Sex'].map({'male': 0, 'female': 1}).values
    y_age = df['Age'].map({'teen': 0, 'adult': 1, 'senior': 2, 'child': 3}).values
    y_race = df['Race'].map({'hispanic': 0, 'white': 1, 'other': 2, 'asian': 3, 'black': 4}).values
    y_face = df['Face'].map({'funny': 0, 'serious': 1, 'smiling': 2}).values
    return y_sex, y_age, y_race, y_face

# 数据集划分
df = pd.read_csv('face.csv')
X = extract_features(df)
y_sex, y_age, y_race, y_face = extract_labels(df)

# 划分训练集和测试集
X_train, X_test, y_sex_train, y_sex_test = train_test_split(X, y_sex, test_size=0.2, random_state=42)
_, _, y_age_train, y_age_test = train_test_split(X, y_age, test_size=0.2, random_state=42)
_, _, y_race_train, y_race_test = train_test_split(X, y_race, test_size=0.2, random_state=42)
_, _, y_face_train, y_face_test = train_test_split(X, y_face, test_size=0.2, random_state=42)

# LightGBM 参数设置
params_base = {
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,  # 降低学习率，通常与增加树的数量配合使用
    'num_leaves': 64,  # 增加叶节点数量，通常提高模型的复杂度
    'feature_fraction': 0.8,  # 使用80%的特征，防止过拟合
    'bagging_fraction': 0.8,  # 使用80%的样本，减少过拟合
    'bagging_freq': 5,  # 每5次迭代进行一次bagging
    'max_depth': 8,  # 适当增加树的最大深度，通常会提高模型的拟合能力
    'min_data_in_leaf': 50,  # 增加每个叶子节点最小样本数，减少过拟合
    'num_threads': 10,  # 设置线程数，避免 CPU 占用过高
    'device': 'gpu',  # 使用GPU加速训练
    'gpu_device_id': 0,  # 使用第一个GPU
    'max_bin': 255,  # 使用更多的bin数，可能提高准确性
    'lambda_l1': 0.1,  # 增加L1正则化，防止过拟合
    'lambda_l2': 0.1,  # 增加L2正则化，防止过拟合
}

# 性别模型（二分类）
params_sex = {
    **params_base,
    'objective': 'binary',
    'metric': 'binary_logloss',
}

# 年龄模型（多分类）
params_age = {
    **params_base,
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 4,
}

# 种族模型（多分类）
params_race = {
    **params_base,
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 5,
}

# 表情模型（多分类）
params_face = {
    **params_base,
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 3,
}

# 训练模型
sex_model_lgb = lgb.LGBMClassifier(**params_sex)
sex_model_lgb.fit(X_train, y_sex_train)

age_model_lgb = lgb.LGBMClassifier(**params_age)
age_model_lgb.fit(X_train, y_age_train)

race_model_lgb = lgb.LGBMClassifier(**params_race)
race_model_lgb.fit(X_train, y_race_train)

face_model_lgb = lgb.LGBMClassifier(**params_face)
face_model_lgb.fit(X_train, y_face_train)

# 预测并评估模型
y_sex_pred_lgb = sex_model_lgb.predict(X_test)
y_age_pred_lgb = age_model_lgb.predict(X_test)
y_race_pred_lgb = race_model_lgb.predict(X_test)
y_face_pred_lgb = face_model_lgb.predict(X_test)

# 计算准确率
sex_acc_lgb = accuracy_score(y_sex_test, y_sex_pred_lgb)
age_acc_lgb = accuracy_score(y_age_test, y_age_pred_lgb)
race_acc_lgb = accuracy_score(y_race_test, y_race_pred_lgb)
face_acc_lgb = accuracy_score(y_face_test, y_face_pred_lgb)

# 打印准确率
print(f"Accuracy on sex (LightGBM): {sex_acc_lgb}")
print(f"Accuracy on age (LightGBM): {age_acc_lgb}")
print(f"Accuracy on race (LightGBM): {race_acc_lgb}")
print(f"Accuracy on face (LightGBM): {face_acc_lgb}")

# 可视化：混淆矩阵
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# 绘制每个模型的混淆矩阵
plot_confusion_matrix(y_sex_test, y_sex_pred_lgb, "Confusion Matrix - Sex (LightGBM)")
plot_confusion_matrix(y_age_test, y_age_pred_lgb, "Confusion Matrix - Age (LightGBM)")
plot_confusion_matrix(y_race_test, y_race_pred_lgb, "Confusion Matrix - Race (LightGBM)")
plot_confusion_matrix(y_face_test, y_face_pred_lgb, "Confusion Matrix - Face (LightGBM)")
