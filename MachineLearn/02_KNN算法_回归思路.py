#思路:
    # 1，计算测试集和每个训练的样本之间的距离。
    # 2．基于距离进行升序排列。
    # 3，找到最近的K个样本。
    # 4.基于K个样本的标签值取平均值
    # 5.将上述计算出的平均值，作为最终预测值
#k值过大 会欠拟合 k=4 x_test数据无论如何变化都会是定值 因为都会是0.1+0.2+0.3+0.4的和除以4
#过小过拟合

# 1．导包.
from sklearn.neighbors import KNeighborsRegressor #KNN算法的回归模型

# 2，准备数据集（测试集和训练集）
# 先求差值 再平方和 再开方
x_train = [
    [0,0,1], #差值：3，11，9 平方和 9+121+81=211 开方：211开根号=14.53
    [1,1,0],#同上 14.28
    [3,10,10],#1
    [4,11,12] #2.24
]
y_train = [0.1,0.2,0.3,0.4]
x_test = [[3,11,10]]

# 3．创建(KNN分类模型）模型对象.
estimator=KNeighborsRegressor(n_neighbors=3)

# 4．模型训练.
estimator.fit(x_train, y_train)

# 5．模型预测
estimator.predict(x_test)

#输出
print(estimator.predict(x_test))