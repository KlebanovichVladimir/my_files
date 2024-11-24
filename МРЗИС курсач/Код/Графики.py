import numpy as np
import csv  # Импортируем библиотеку для работы с CSV
import matplotlib.pyplot as plt  # Импортируем библиотеку для построения графиков

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
x0 = 0.5
y0 = 0.5
z0 = 0.5
h = 0.1
n_steps = int(100 / h)

# Решение системы методом Рунге-Кутты
t_values, x_values, y_values, z_values = runge_kutta_system(t0, x0, y0, z0, h, n_steps)

# Печать всех значений после завершения
print("\nВсе значения:")
print(f"t = {t0}, X = {x0}, Y = {y0}, Z = {z0}")
for t, x, y, z in zip(t_values, x_values, y_values, z_values):
    print(f"t = {t:.2f}, X = {x:.6f}, Y = {y:.6f}, Z = {z:.6f}")

# Построение графика в 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Отображение линии функции тонкой черной линией
ax.plot(x_values, y_values, z_values, color='black', linestyle='-', linewidth=0.1, label='Путь в 3D пространстве')

# Отображение точек на линии
ax.scatter(x_values, y_values, z_values, color='black', s=0.3, label='Точки')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D график функции')
ax.legend()
plt.show()  # Показываем график

# Проекции на плоскости XY, XZ и YZ
# Плоскость XY
plt.figure()
plt.plot(x_values, y_values, color='blue', linewidth=0.5, label='Проекция на XY')
plt.scatter(x_values, y_values, color='blue', s=0.3)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Проекция на плоскость XY')
plt.legend()
plt.grid()
plt.show()

# Плоскость XZ
plt.figure()
plt.plot(x_values, z_values, color='green', linewidth=0.5, label='Проекция на XZ')
plt.scatter(x_values, z_values, color='green', s=0.3)
plt.xlabel('X')
plt.ylabel('Z')
plt.title('Проекция на плоскость XZ')
plt.legend()
plt.grid()
plt.show()

# Плоскость YZ
plt.figure()
plt.plot(y_values, z_values, color='red', linewidth=0.5, label='Проекция на YZ')
plt.scatter(y_values, z_values, color='red', s=0.3)
plt.xlabel('Y')
plt.ylabel('Z')
plt.title('Проекция на плоскость YZ')
plt.legend()
plt.grid()
plt.show()

# Графики зависимости X(t), Y(t) и Z(t)
# График X(t)
plt.figure()
plt.plot(t_values, x_values, color='purple', linewidth=0.5, label='X(t)')
plt.xlabel('t')
plt.ylabel('X')
plt.title('Зависимость X(t)')
plt.legend()
plt.grid()
plt.show()

# График Y(t)
plt.figure()
plt.plot(t_values, y_values, color='orange', linewidth=0.5, label='Y(t)')
plt.xlabel('t')
plt.ylabel('Y')
plt.title('Зависимость Y(t)')
plt.legend()
plt.grid()
plt.show()

# График Z(t)
plt.figure()
plt.plot(t_values, z_values, color='brown', linewidth=0.5, label='Z(t)')
plt.xlabel('t')
plt.ylabel('Z')
plt.title('Зависимость Z(t)')
plt.legend()
plt.grid()
plt.show()