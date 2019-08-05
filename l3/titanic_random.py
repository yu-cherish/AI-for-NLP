import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random


content = pd.read_csv('F:\lesson02\lesson03\w_titanic\w_train.csv')
content = content.dropna()
age_with_fares = content[
    (content['Age'] > 22) & (content['Fare'] < 400) & (content['Fare'] > 130)
]
sub_fare = age_with_fares['Fare']
sub_age = age_with_fares['Age']

plt.scatter(sub_age,sub_fare)
plt.show()

def func(age, k, b): return k * age + b

def loss(y, yhat):
    """

    :param y:  the real fares
    :param yhat: the estimated fares
    :return: how good is the estimated fares
    """
    return np.mean(np.abs(y-yhat))


# k_hat = random.randint(-10, 10)
# b_hat = random.randint(-10, 10)

min_error_rate = float('inf')
best_k, best_b = None, None

loop_times = 10000

losses = []

while loop_times > 0:
    k_hat = random.random() * 20 - 10
    b_hat = random.random() * 20 - 10
    # k_hat = random.randint(-10, 10)
    # b_hat = random.randint(-10, 10)
    estimated_fares = func(sub_age, k_hat, b_hat)
    error_rate = loss(sub_fare, yhat=estimated_fares)
    # print(error_rate)

    if error_rate < min_error_rate:
        min_error_rate = error_rate
        best_k, best_b = k_hat, b_hat
        print(min_error_rate)
        print('loop == {}'.format(loop_times))
        losses.append(min_error_rate)
        print('f(age) = {} * age + {}, with error rate: {}'.format(best_k, best_b, min_error_rate))

    loop_times -= 1

# plt.scatter(sub_age, sub_fare)
# plt.plot(sub_age, func(sub_age, best_k, best_b), c='r')
# plt.show()
plt.plot(range(len(losses)), losses)
plt.show()
