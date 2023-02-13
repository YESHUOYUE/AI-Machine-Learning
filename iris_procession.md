

# 导入库
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
```

# 导入csv数据（鸢尾花数集）
```python
data = pd.read_csv("iris.csv")
data.columns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Class"]
data = data.dropna()
print(data)
```

# 数据划分
```python
X = data.drop(columns=['Class'])
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
```

# 用matplotlib展示数据
```python
plt.figure(figsize=(10, 10))
for column_index, column in enumerate(data.columns):
    if column == 'Class':
        continue
    plt.subplot(2, 2, column_index + 1)
    sns.violinplot(x='Class', y=column, data=data)
plt.show()
```
展示图片如下：
![image](https://user-images.githubusercontent.com/116483698/218401455-cb9c6b89-b6dd-40fd-bafb-5d2d72bd082c.png)



# 用不同算法处理鸢尾花数集
## DecisionTree（决策树）
```python

# 导入库
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 初始化、训练、预测
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 模型评估/交叉验证
score = accuracy_score(y_test, predictions)
print("Decision Tree:", score)

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X, y, cv=10)
print("Decision Tree(cross):", np.mean(cv_scores))
```
结果展示：

![image](https://user-images.githubusercontent.com/116483698/218399257-9db3aedc-a7f1-4601-a8a3-c2e22903199d.png)

## KNN算法
```python
from sklearn.neighbors import KNeighborsClassifier

# 训练
knn = KNeighborsClassifier(n_neighbors=5)  # 设置邻居数K
knn.fit(X_train, y_train)

# 测试评估模型
prediction2 = knn.predict(X_test)
score2 = accuracy_score(y_test, prediction2)
print("KNN :", score2)
cv_scores2 = cross_val_score(knn, X, y, cv=10)
print("KNN(cross):", np.mean(cv_scores2))

```
结果展示：

![image](https://user-images.githubusercontent.com/116483698/218405474-ec507de9-f02f-4499-b627-e9e5de0e9d45.png)

## SVM算法
```python
from sklearn import svm

# 训练
clf = svm.SVC(C=0.5, kernel='linear', decision_function_shape='ovr')
clf.fit(X_train, y_train)

# 测试评估模型
acc = clf.predict(X_train) == y_train
print('SVM:', np.mean(acc))
```

结果展示：

![image](https://user-images.githubusercontent.com/116483698/218408805-8ac25fa9-5370-4a0f-9849-0133f56ff7ba.png)


