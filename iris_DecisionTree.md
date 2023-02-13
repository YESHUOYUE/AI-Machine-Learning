```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# 导入数据
data = pd.read_csv("iris.csv")
data.columns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Class"]
data = data.dropna()
# print(data)

# 数据划分
X = data.drop(columns=['Class'])
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# 用matplotlib展示数据
plt.figure(figsize=(10, 10))
for column_index, column in enumerate(data.columns):
    if column == 'Class':
        continue
    plt.subplot(2, 2, column_index + 1)
    sns.violinplot(x='Class', y=column, data=data)
# plt.show()

# 决策树
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 初始化
model = DecisionTreeClassifier()
# 训练
model.fit(X_train, y_train)
# 预测
predictions = model.predict(X_test)

# 模型评估/交叉验证
score = accuracy_score(y_test, predictions)
print("Decision Tree:", score)
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X, y, cv=10)
print("Decision Tree(cross):", np.mean(cv_scores))
```
结果展示
![image](https://user-images.githubusercontent.com/116483698/218399257-9db3aedc-a7f1-4601-a8a3-c2e22903199d.png)
