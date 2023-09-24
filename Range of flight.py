import json
import numpy as np
import matplotlib.pyplot as plt
#Спочатку прочитаємо наш json файл
with open("./config.json", "rt") as f:
    conf = json.load(fp=f)
#Візьмемо звідти рандомні значення для вибірок
np.random.seed(conf["seed"])
n_samples = conf["n_samples"]
v_normal = np.random.normal(
    loc=conf["v"]["normal"]["mean"],
    scale=conf["v"]["normal"]["std"],
    size=n_samples
)
v_uniform = np.random.uniform(
    low=conf["v"]["uniform"]["min"],
    high=conf["v"]["uniform"]["max"],
    size=n_samples
)
#Зауваження - градуси в радіанах
alpha_normal = np.random.normal(
    loc=conf["alpha"]["normal"]["mean"],
    scale=conf["alpha"]["normal"]["std"],
    size=n_samples
)
alpha_uniform = np.random.uniform(
    low=conf["alpha"]["uniform"]["min"],
    high=conf["alpha"]["uniform"]["max"],
    size=n_samples
)
#Напишемо функцію побудови графіків
def build_graph(v: np.ndarray, alpha: np.ndarray):
    #Округлимо нампайївські масиви до двух десятичних знаків
    v = np.round(v, 2)
    alpha = np.round(alpha, 2)
    #Створимо сітку із 3 графіків в одному вікні
    plt.figure(figsize=(12, 8))
    #Перший графік (графік розподілу швидкості)
    plt.subplot(2, 2, 1)
    plt.hist(v, label='v, m/s', bins=30)
    plt.xlabel('Швидкість')
    plt.ylabel('Частота')
    plt.legend()
    #Другий графік (графік розподілу кута)
    plt.subplot(2, 2, 2)
    plt.hist(alpha, label='alpha, rad', bins=30)
    plt.xlabel('Кут')
    plt.ylabel('Частота')
    plt.legend()
    #Третій графіу (графік розподілу дальності польоту) + сам розподіл для L
    plt.subplot(2, 1, 2)
    L = ((v ** 2) * np.sin(2 * alpha)) / 9.8
    L = np.round(L, 2)
    freq, bins = np.histogram(L, bins=10)
    print(freq)
    print(bins)
    plt.hist(L, label='L, m', bins=10)
    plt.xlabel('Дальність польоту')
    plt.ylabel('Частота')
    plt.legend()
    return plt.show()
#Тепер виводимо всі 12 градусів (4 сітки)
build_graph(v_normal, alpha_normal)
build_graph(v_normal, alpha_uniform)
build_graph(v_uniform, alpha_normal)
build_graph(v_uniform, alpha_uniform)
