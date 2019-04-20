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
    return np.mean(np.abs(y-yhat))    # L1
    # return np.mean(np.sqrt(y - yhat))   L^1/2
    # return np.mean(np.square(y, yhat))  # L2....等等


min_error_rate = float('inf')


loop_times = 10000

losses = []

change_directions = [
    # (k, b)
    (+1, -1), # k increase, b decrease
    (+1, +1),
    (-1, +1),
    (-1, -1)  # k decrease, b decrease
]

k_hat = random.random() * 20 - 10
b_hat = random.random() * 20 - 10

best_k, best_b = k_hat, b_hat

best_direction = None


def step(): return random.random() * 1


direction = random.choice(change_directions)


while loop_times > 0:
    k_delta_direction, b_delta_direction = direction

    k_delta = k_delta_direction * step()
    b_delta = b_delta_direction * step()

    new_k = best_k + k_delta
    new_b = best_b + b_delta

    estimated_fares = func(sub_age, new_k, new_b)
    error_rate = loss(sub_fare, yhat=estimated_fares)
    # print(error_rate)

    if error_rate < min_error_rate:
        min_error_rate = error_rate
        best_k, best_b = new_k, new_b
        # 保留最好的方向
        direction = (k_delta_direction, b_delta_direction)

        print(min_error_rate)
        print('loop == {}'.format(loop_times))
        losses.append(min_error_rate)
        print('f(age) = {} * age + {}, with error rate: {}'.format(best_k, best_b, min_error_rate))
    else:
        direction = random.choice(list(set(change_directions) - {(k_delta_direction, b_delta_direction)}))
    loop_times -= 1

plt.scatter(sub_age, sub_fare)
plt.plot(sub_age, func(sub_age, best_k, best_b), c='r')
plt.show()
# plt.plot(range(len(losses)), losses)
# plt.show()
