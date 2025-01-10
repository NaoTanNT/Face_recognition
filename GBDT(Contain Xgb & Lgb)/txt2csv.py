import pandas as pd
import re

# 输入的face文本文档路径
fillter_file = 'face_fillter'  # 替换为你的文件路径

# 创建一个空的列表，用来保存每一行的提取数据
data = []

# 打开face文件进行读取
with open(raw_file, 'r') as input_file:
    for line in input_file:
        # 去除行首尾空白字符
        line = line.strip()

        # 提取最前面的数字作为 index
        index_match = re.match(r'^\d+', line)
        if not index_match:
            continue  # 如果没有找到数字，跳过这一行

        filename = index_match.group(0)  # 获取数字并保留为字符串

        # 提取各个字段内容
        # 使用正则提取每个字段：sex, age, race, face
        sex = re.search(r'_sex\s+(\w+)', line)
        age = re.search(r'_age\s+(\w+)', line)
        race = re.search(r'_race\s+(\w+)', line)
        face = re.search(r'_face\s+(\w+)', line)

        # 如果找不到某些字段，则默认设置为空字符串或空格
        sex = sex.group(1) if sex else ''
        age = age.group(1) if age else ''
        race = race.group(1) if race else ''
        face = face.group(1) if face else ''

        # 将数据添加到列表中
        data.append([filename, sex, age, race, face])

# 将数据转化为pandas DataFrame
columns = ['Filename', 'Sex', 'Age', 'Race', 'Face']
df = pd.DataFrame(data, columns=columns)

# 强制将 'Filename' 列转换为字符串类型
df['Filename'] = df['Filename'].astype(str)
# 输出到CSV文件
df.to_csv('face.csv', index=False)

# 读取CSV文件
csv_file = 'face.csv'
data_df = pd.read_csv(csv_file)

# 提取每一列的集合，排除 'Filename' 列
column_sets = {column: set(data_df[column]) for column in data_df.columns if column != 'Filename'}

# 打印每一列的集合
for column, column_set in column_sets.items():
    print(f"Column '{column}' has the set: {column_set}")
