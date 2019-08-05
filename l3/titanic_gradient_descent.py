import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# 数据预处理，优化等等，使得数据变得标准化
content = pd.read_csv('F:\lesson02\lesson03\w_titanic\w_train.csv')
content = content.dropna()
age_with_fares = content[
    (content['Age'] > 22) & (content['Fare'] < 400) & (content['Fare'] > 130)
]
sub_fare = age_with_fares['Fare']
sub_age = age_with_fares['Age']

plt.scatter(sub_age,sub_fare)
plt.show()

# 如何搭建神经网络模型，如cnn等，其实就是函数
def func(age, k, b): return k * age + b
# 损失函数
def loss(y, yhat):
    """

    :param y:  the real fares
    :param yhat: the estimated fares
    :return: how good is the estimated fares
    """
    return np.mean(np.abs(y-yhat))    # L1
    # return np.mean(np.sqrt(y - yhat))   L^1/2
    # return np.mean(np.square(y, yhat))  # L2....等等


loop_times = 10000

losses = []

# 数据的初始化问题
k_hat = random.random() * 20 - 10
b_hat = random.random() * 20 - 10


def derivative_k(y, yhat, x):
    abs_values = [1 if (y_i - yhat_i) > 0 else -1 for y_i, yhat_i in zip(y, yhat)]

    return np.mean([a * -x_i for a, x_i in zip(abs_values, x)])


def derivative_b(y, yhat):
    abs_values = [1 if (y_i - yhat_i) > 0 else -1 for y_i, yhat_i in zip(y, yhat)]
    return np.mean([a * -1 for a in abs_values])


# 在某个方向上只变化某种比例，以防变化太多错失最优值
learning_rate = 0.1
# 优化器的优化方法

while loop_times > 0:

    # 变化方向--
    k_delta = -1 * learning_rate * derivative_k(sub_fare, func(sub_age, k_hat, b_hat), sub_age)
    b_delta = -1 * learning_rate * derivative_b(sub_fare, func(sub_age, k_hat, b_hat))

    k_hat += k_delta
    b_hat += b_delta

    estimated_fares = func(sub_age, k_hat, b_hat)
    error_rate = loss(sub_fare, yhat=estimated_fares)

    print('loop == {}'.format(loop_times))
    print('f(age) = {} * age + {}, with error rate: {}'.format(k_hat, b_hat, error_rate))

    # losses.append(error_rate)
    loop_times -= 1

plt.scatter(sub_age, sub_fare)
plt.plot(sub_age, func(sub_age, k_hat, b_hat), c='r')
plt.show()
# plt.plot(range(len(losses)), losses)
# plt.show()
