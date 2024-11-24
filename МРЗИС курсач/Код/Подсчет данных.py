import numpy as np
import csv  # Импортируем библиотеку для работы с CSV

# Определим функции для системы уравнений
def f1(t, x, y, z):
    return 0.4 * x + z

def f2(t, x, y, z):
    return x * z - y

def f3(t, x, y, z):
    return -x + y

# Метод Рунге-Кутты четвертого порядка
def runge_kutta_system(t0, x0, y0, z0, h, n):
    t_values = [t0]
    x_values = [x0]
    y_values = [y0]
    z_values = [z0]

    t, x, y, z = t0, x0, y0, z0

    for _ in range(n):
        # Вычисление коэффициентов k для каждого уравнения
        k1_x = h * f1(t, x, y, z)
        k1_y = h * f2(t, x, y, z)
        k1_z = h * f3(t, x, y, z)
        
        k2_x = h * f1(t + 0.5 * h, x + 0.5 * k1_x, y + 0.5 * k1_y, z + 0.5 * k1_z)
        k2_y = h * f2(t + 0.5 * h, x + 0.5 * k1_x, y + 0.5 * k1_y, z + 0.5 * k1_z)
        k2_z = h * f3(t + 0.5 * h, x + 0.5 * k1_x, y + 0.5 * k1_y, z + 0.5 * k1_z)
        
        k3_x = h * f1(t + 0.5 * h, x + 0.5 * k2_x, y + 0.5 * k2_y, z + 0.5 * k2_z)
        k3_y = h * f2(t + 0.5 * h, x + 0.5 * k2_x, y + 0.5 * k2_y, z + 0.5 * k2_z)
        k3_z = h * f3(t + 0.5 * h, x + 0.5 * k2_x, y + 0.5 * k2_y, z + 0.5 * k2_z)
        
        k4_x = h * f1(t + h, x + k3_x, y + k3_y, z + k3_z)
        k4_y = h * f2(t + h, x + k3_x, y + k3_y, z + k3_z)
        k4_z = h * f3(t + h, x + k3_x, y + k3_y, z + k3_z)
        
        # Обновление значений переменных
        x += (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        y += (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
        z += (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6
        t += h

        # Сохранение результатов
        t_values.append(t)
        x_values.append(x)
        y_values.append(y)
        z_values.append(z)

    return t_values, x_values, y_values, z_values

# Начальные условия и параметры
t0 = 0.0
x0 = 0.1
y0 = 0.1
z0 = 1.0
h = 0.1
n_steps = int(100 / h)

# Решение системы методом Рунге-Кутты
t_values, x_values, y_values, z_values = runge_kutta_system(t0, x0, y0, z0, h, n_steps)

# Печать всех значений после завершения
print("\nВсе значения:")
print(f"t = {t0}, X = {x0}, Y = {y0}, Z = {z0}")
for t, x, y, z in zip(t_values, x_values, y_values, z_values):
    print(f"t = {t:.2f}, X = {x:.6f}, Y = {y:.6f}, Z = {z:.6f}")

# Сохранение результатов в CSV файл
with open("D:/7 семестр/МРЗИС курсач/Код/results_for_me.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    # Записываем заголовок
    writer.writerow(['t', 'X', 'Y', 'Z'])
    # Записываем данные
    for t, x, y, z in zip(t_values, x_values, y_values, z_values):
        writer.writerow([t, x, y, z])

print("Результаты сохранены в файл 'results.csv'.")