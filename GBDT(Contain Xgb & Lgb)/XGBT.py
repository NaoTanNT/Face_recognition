import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os

# 假设数据已加载为DataFrame
df = pd.read_csv('face.csv')

# 图像二进制文件的加载函数（优化为并行加载）
from joblib import Parallel, delayed

def load_image(file_name, image_folder="fillter_data"):
    file_name = str(file_name)
    with open(os.path.join(image_folder, file_name), 'rb') as f:
        img_data = f.read()
    img = np.frombuffer(img_data, dtype=np.uint8).reshape((128, 128))
    return img

# 使用joblib进行并行图像加载
def extract_features(df, image_folder="fillter_data"):
    X = Parallel(n_jobs=-1)(delayed(load_image)(row['Filename'], image_folder) for _, row in df.iterrows())
    return np.array([img.flatten() for img in X])

# 提取标签
def extract_labels(df):
    y_sex = df['Sex'].map({'male': 0, 'female': 1}).values
    y_age = df['Age'].map({'teen': 0, 'adult': 1, 'senior': 2, 'child': 3}).values
    y_race = df['Race'].map({'hispanic': 0, 'white': 1, 'other': 2, 'asian': 3, 'black': 4}).values
    y_face = df['Face'].map({'funny': 0, 'serious': 1, 'smiling': 2}).values
    return y_sex, y_age, y_race, y_face

# 数据集划分
X = extract_features(df)
y_sex, y_age, y_race, y_face = extract_labels(df)

# 划分训练集和测试集
X_train, X_test, y_sex_train, y_sex_test = train_test_split(X, y_sex, test_size=0.2, random_state=42)
_, _, y_age_train, y_age_test = train_test_split(X, y_age, test_size=0.2, random_state=42)
_, _, y_race_train, y_race_test = train_test_split(X, y_race, test_size=0.2, random_state=42)
_, _, y_face_train, y_face_test = train_test_split(X, y_face, test_size=0.2, random_state=42)

# 使用 XGBoost 进行训练（开启GPU加速，并优化计算量）
sex_model_xgb = xgb.XGBClassifier(tree_method='hist', device= 'cuda', n_estimators=70, learning_rate=0.05, max_depth=5)
sex_model_xgb.fit(X_train, y_sex_train)

age_model_xgb = xgb.XGBClassifier(tree_method='hist', device= 'cuda', n_estimators=70, learning_rate=0.05, max_depth=5)
age_model_xgb.fit(X_train, y_age_train)

race_model_xgb = xgb.XGBClassifier(tree_method='hist', device= 'cuda', n_estimators=70, learning_rate=0.05, max_depth=5)
race_model_xgb.fit(X_train, y_race_train)

face_model_xgb = xgb.XGBClassifier(tree_method='hist', device= 'cuda', n_estimators=70, learning_rate=0.05, max_depth=5)
face_model_xgb.fit(X_train, y_face_train)

# 预测并评估模型
y_sex_pred_xgb = sex_model_xgb.predict(X_test)
y_age_pred_xgb = age_model_xgb.predict(X_test)
y_race_pred_xgb = race_model_xgb.predict(X_test)
y_face_pred_xgb = face_model_xgb.predict(X_test)

# 计算准确率
sex_acc_xgb = accuracy_score(y_sex_test, y_sex_pred_xgb)
age_acc_xgb = accuracy_score(y_age_test, y_age_pred_xgb)
race_acc_xgb = accuracy_score(y_race_test, y_race_pred_xgb)
face_acc_xgb = accuracy_score(y_face_test, y_face_pred_xgb)

# 打印准确率
print(f"Accuracy on sex (XGBoost): {sex_acc_xgb}")
print(f"Accuracy on age (XGBoost): {age_acc_xgb}")
print(f"Accuracy on race (XGBoost): {race_acc_xgb}")
print(f"Accuracy on face (XGBoost): {face_acc_xgb}")

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
plot_confusion_matrix(y_sex_test, y_sex_pred_xgb, "Confusion Matrix - Sex (XGBoost)")
plot_confusion_matrix(y_age_test, y_age_pred_xgb, "Confusion Matrix - Age (XGBoost)")
plot_confusion_matrix(y_race_test, y_race_pred_xgb, "Confusion Matrix - Race (XGBoost)")
plot_confusion_matrix(y_face_test, y_face_pred_xgb, "Confusion Matrix - Face (XGBoost)")
