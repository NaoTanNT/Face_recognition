import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle

# 1.预处理数据
def prepare_data(cleaned_data):
    """
    解析标签并将数据准备成适合机器学习模型的格式。

    """
    if not cleaned_data:
        raise ValueError("无已清洗数据")

    print(f"正在处理 {len(cleaned_data)} 条已清洗数据")  # 调试信息
    X, y_sex, y_age, y_race, y_face = [], [], [], [], []  # 存储图像数据和四个特征的标签
    for item in cleaned_data:
        try:
            # 加载图像并进行预处理
            img = Image.open(item['path']).convert('L')  # 将图像转换为灰度模式
            img = img.resize((64, 64))  # 调整图像大小为64x64像素
            img_array = np.array(img).flatten() / 255.0  # 归一化并且展平图像数据

            # 计算图像亮度（平均像素值）
            brightness = np.mean(img_array)
            # 检查亮度是否在允许范围内
            if not (0.03 <= brightness <= 0.4):
                print(f"剔除亮度异常图像: {item['path']}, 亮度: {brightness}")
                continue

            X.append(img_array)

            # 编码标签
            y_sex.append(item.get('_sex', 'unknown'))  # 如果标签不存在则设为'unknown'
            y_age.append(item.get('_age', 'unknown'))
            y_race.append(item.get('_race', 'unknown'))
            y_face.append(item.get('_face', 'unknown'))

        except Exception as e:
            print(f"处理失败图像路径 {item['path']}: {e}")
            continue  # 跳过有问题的图像

    # 将列表转换为NumPy数组
    X = np.array(X)

    # 使用LabelEncoder将字符串标签编码为整数
    le_sex = LabelEncoder()
    le_age = LabelEncoder()
    le_race = LabelEncoder()
    le_face = LabelEncoder()

    y_sex = le_sex.fit_transform(y_sex)
    y_age = le_age.fit_transform(y_age)
    y_race = le_race.fit_transform(y_race)
    y_face = le_face.fit_transform(y_face)

    # 确保没有任何数组为空
    if len(X) == 0 or any(len(y) == 0 for y in [y_sex, y_age, y_race, y_face]):
        raise ValueError("检测到空特征矩阵或目标变量")

    # 返回X以及对应的y，并且创建一个包含LabelEncoders的字典
    label_encoders_dict = {
        '_sex': le_sex,
        '_age': le_age,
        '_race': le_race,
        '_face': le_face
    }

    return X, y_sex, y_age, y_race, y_face, label_encoders_dict

# 2. 数据模拟
np.random.seed(42)
n_samples = 2000  # 样本总数
n_features = 128  # 特征数

# 生成特征数据
X = np.random.rand(n_samples, n_features)

# 生成标签
# 性别（0: 男性, 1: 女性）
y_gender = np.random.randint(0, 2, n_samples)
# 年龄（0: 青少年, 1: 青年, 2: 中年, 3: 老年）
y_age = np.random.randint(0, 4, n_samples)
# 人种（0: 亚洲, 1: 非洲, 2: 欧洲, 3: 拉丁美洲, 4: 大洋洲）
y_ethnicity = np.random.randint(0, 5, n_samples)
# 表情（0: 高兴, 1: 悲伤, 2: 愤怒, 3: 惊讶, 4: 中性）
y_expression = np.random.randint(0, 5, n_samples)

# 引入噪声增强随机性
X[:, 0] += np.random.normal(0, 0.5, n_samples)

# 合并标签到一个字典中（方便管理）
tasks = {
    "Gender": y_gender,
    "Age": y_age,
    "Ethnicity": y_ethnicity,
    "Expression": y_expression
}

# 3. 多任务分类
results = {}  # 存储每个任务的结果
for task_name, y in tasks.items():
    print(f"\n---- Task: {task_name} ----")

    # 数据集拆分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 决策树模型
    clf = DecisionTreeClassifier(
        random_state=42,
        max_depth=15,  # 最大深度
        min_samples_split=10,  # 内部节点最小分裂样本数
        min_samples_leaf=5  # 叶节点最小样本数
    )
    clf.fit(X_train, y_train)

    # 使用GridSearchCV进行超参数调优
    clf = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    # 预测
    y_pred: object = clf.predict(X_test)

    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"Accuracy for {task_name}: {accuracy:.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # 存储结果
    results[task_name] = {
        "accuracy": accuracy,
        "report": report
    }

# 4. 可视化结果
# 假设 classifiers 是包含训练好的分类器的字典
# 假设 X_test 是测试集的特征矩阵，y_test 是测试集的真实标签

# 评估每个特征的分类器
for feature_name, clf in classifiers.items():
    # 进行预测
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)  # 获取预测概率

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {feature_name} classifier')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # 二值化标签
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_binarized.shape[1]

    # 计算 ROC 曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制 ROC 曲线
    plt.figure(figsize=(8, 6))
    colors = cycle(
        ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue', 'gray',
         'indigo', 'orange', 'turquoise', 'darkgreen', 'lavender', 'brown', 'violet'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {feature_name} classifier')
    plt.legend(loc="lower right")
    plt.show()

    # 绘制每个任务的准确率
    plt.figure(figsize=(10, 6))
    task_names = list(results.keys())
    accuracies = [results[task]["accuracy"] for task in task_names]

    plt.bar(task_names, accuracies, color=["blue", "green", "orange", "purple"], alpha=0.7)
    plt.title("Task-wise Accuracy")
    plt.xlabel("Tasks")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(axis="y")
    plt.show()
