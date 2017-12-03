# @time    : 2017/12/3 19:43
# @Author  : chiew
# @File    : K_means.py

import numpy as np
from sklearn.cluster import KMeans


def load_date(file_path):
    with open(file_path, 'r+') as f:
        lines = f.readlines()
        ret_data = []
        ret_city_name = []
        for line in lines:
            items = line.strip().split(',')
            ret_city_name.append(items[0])
            ret_data.append([float(items[i]) for i in range(1, len(items))])
    return ret_data, ret_city_name


if __name__ == '__main__':
    data, city_name = load_date('city.txt')
    # print(data)
    # print(city_name)
    km = KMeans(n_clusters=4)
    label = km.fit_predict(data)
    expenses = np.sum(km.cluster_centers_, axis=1)

    city_cluster = [[], [], [], []]
    for i in range(len(city_name)):
        city_cluster[label[i]].append(city_name[i])
    for i in range(len(city_cluster)):
        print("Expenses:%.2f" % expenses[i])
        print(city_cluster[i])
