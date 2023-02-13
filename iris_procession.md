[导入库]()

# 导入库
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
import seaborn as sns

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
### 结果展示：

![image](https://user-images.githubusercontent.com/116483698/218399257-9db3aedc-a7f1-4601-a8a3-c2e22903199d.png)

### 优缺点：
#### 优点
1. 决策树易于理解和解释，可以可视化分析，容易提取出规则。
2. 可以同时处理标称型和数值型数据。
3. 比较适合处理有缺失属性的样本。
4. 能够处理不相关的特征。
5. 测试数据集时，运行速度比较快。
6. 在相对短的时间内能够对大型数据源做出可行且效果良好的结果。

#### 缺点
1. 容易发生过拟合（随机森林可以很大程度上减少过拟合）。
2. 容易忽略数据集中属性的相互关联。
3. 对于那些各类别样本数量不一致的数据，在决策树中，进行属性划分时，不同的判定准则会带来不同的属性选择倾向。

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
### 结果展示：

![image](https://user-images.githubusercontent.com/116483698/218405474-ec507de9-f02f-4499-b627-e9e5de0e9d45.png)

### 优缺点：
#### 优点
1.理论成熟，思想简单，既可以用来做分类又可以做回归
2.可以用于非线性分类
3.训练时间复杂度比支持向量机之类的算法低
4.和朴素贝叶斯之类的算法比，对数据没有假设，准确度高，对异常点不敏感
5.由于KNN方法主要靠周围有限的邻近的样本，而不是靠判别类域的方法来确定所属的类别，因此对于类域的交叉或重叠较多的待分类样本集来说，KNN方法较其他方法更为适合
6.该算法比较适用于样本容量比较大的类域的自动分类，而那些样本容量比较小的类域采用这种算法比较容易产生误分类情况
#### 缺点
1.计算量大，尤其是特征数非常多的时候
2.样本不平衡的时候，对稀有类别的预测准确率低
3.KD树，球树之类的模型建立需要大量的内存
4.是慵懒散学习方法，基本上不学习，导致预测时速度比起逻辑回归之类的算法慢
5.相比决策树模型，KNN模型的可解释性不强

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

### 优缺点
#### 算法优点：
（1）使用核函数可以向高维空间进行映射
（2）使用核函数可以解决非线性的分类
（3）分类思想很简单，就是将样本与决策面的间隔最大化
（4）分类效果较好
#### 算法缺点：
（1）SVM算法对大规模训练样本难以实施
（2）用SVM解决多分类问题存在困难
（3）对缺失数据敏感，对参数和核函数的选择敏感
