import numpy_financial as npf

import numpy as np
from scipy import stats

duration = 4 # срок проекта
tax_rate = 0.4 # ставка налога
amortization = 30_000 # амортизация
initial_investment = 500_000 # начальные инвестиции

np.random.seed(42)

mean = 6500
std_dev = 370
num_samples = 10000

values = np.random.normal(mean, std_dev, num_samples)

low_cost = 40000
high_cost = 90000

costs = np.random.uniform(low_cost, high_cost, num_samples)

# Норма дисконта, %
low_discount = 0.08
high_discount = 0.14

discounts = np.random.uniform(low_discount, high_discount, num_samples)

# Остаточная стоимость, руб.
scale = 1 / 0.00005  # Параметр масштаба для экспоненциального распределения равен обратному значению параметра распределения

residual_values = np.random.exponential(scale, num_samples)

# Цена за штуку, руб.
left = 180
mode = 270
right = 310

prices = np.random.triangular(left, mode, right, num_samples)

left = 140
mode = 180
right = 220

variable_costs = np.random.triangular(left, mode, right, num_samples)

CFt = (1 - tax_rate) * (values * prices - variable_costs * values + amortization - costs)
NPV = CFt/(1 + discounts) + CFt/((1 + discounts)**2) + CFt/((1 + discounts)**3) + (CFt + residual_values)/((1 + discounts)**4) - initial_investment
PI = (CFt/(1 + discounts) + CFt/((1 + discounts)**2) + CFt/((1 + discounts)**3) + (CFt + residual_values)/((1 + discounts)**4)) / initial_investment

def print_statistics(values):
    print('Среднее:', round(np.mean(values), 5))
    print('Стандартная ошибка:', round(stats.sem(values), 5))
    print('Медиана:', round(np.median(values), 5))
    print('Стандартное отклонение:', round(np.std(values), 5))
    print('Дисперсия выборки:', round(np.var(values), 5))
    print('Эксцесс:', round(stats.kurtosis(values), 5))
    print('Ассиметричность:', round(stats.skew(values), 5))
    print('Интервал:', round(np.ptp(values), 5))
    print('Минимум:', round(np.min(values), 5))
    print('Максимум:', round(np.max(values), 5))
    print('Сумма:', round(np.sum(values), 5))
    print('Счет:', len(values))

initial_investment_array = np.full(num_samples, initial_investment)

# Расчет IRR
cash_flows = np.column_stack((initial_investment_array * -1, CFt, CFt, CFt, CFt + residual_values))  # Скорректированные денежные потоки с учетом начальных инвестиций и остаточных значений
IRR = [npf.irr(cf) for cf in cash_flows]  # Преобразование IRR в проценты



T = 0.20
i = 0.8

P1 = np.random.uniform(8500, 10500, num_samples) 
P2 = np.random.uniform(9000, 11000, num_samples) 
P3 = np.random.uniform(9500, 11500, num_samples) 
Q1 = np.random.normal(1500, 300, num_samples) 
Q2 = np.random.normal(1600, 325, num_samples) 
Q3 = np.random.normal(1700, 350, num_samples) 
X = np.random.normal(55, 5, num_samples) / 100 
Y = np.random.normal(15, 2, num_samples) / 100

def CFt(T, Q, P, X, Y): 
    return (1 - T) * (Q * P - Q * (X + Y))

CF1 = CFt(T, Q1, P1, X, Y)
CF2 = CFt(T, Q2, P2, X, Y)
CF3 = CFt(T, Q3, P3, X, Y)
CF = CF1 + CF2 + CF3


NPV = sum(cf / ((1 + i) ** (t + 1)) for t, cf in enumerate([CF1, CF2, CF3]))