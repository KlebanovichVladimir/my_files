import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

def f1(t, x, y, z):
    return -y + z**2

def f2(t, x, y, z):
    return x + 0.5 * y

def f3(t, x, y, z):
    return x - z

def runge_kutta_system(t0, x0, y0, z0, h, n):
    t_values = [t0]
    x_values = [x0]
    y_values = [y0]
    z_values = [z0]
    t, x, y, z = t0, x0, y0, z0

    for _ in range(n):
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

        x += (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        y += (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
        z += (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6
        t += h

        t_values.append(t)
        x_values.append(x)
        y_values.append(y)
        z_values.append(z)

    return t_values, x_values, y_values, z_values

# Параметры системы
t0, x0, y0, z0, step_size, n_steps = 0.0, 0.1, 0.1, 1.0, 0.1, int(100 / 0.1)
t_values, x_values, y_values, z_values = runge_kutta_system(t0, x0, y0, z0, step_size, n_steps)

# Определение индекса для отрезка [0; 33.4]
train_steps = int(33.4 / step_size)

# Подготовка данных для обучения и тестирования
X_train = np.array([x_values[:train_steps], y_values[:train_steps], z_values[:train_steps]]).T
Y_train = np.array([x_values[1:train_steps+1], y_values[1:train_steps+1], z_values[1:train_steps+1]]).T

X_test = np.array([x_values[train_steps:-1], y_values[train_steps:-1], z_values[train_steps:-1]]).T
Y_test = np.array([x_values[train_steps+1:], y_values[train_steps+1:], z_values[train_steps+1:]]).T

# Масштабирование данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
Y_train_scaled = scaler.fit_transform(Y_train)
X_test_scaled = scaler.transform(X_test)
Y_test_scaled = scaler.transform(Y_test)

# Преобразование стандартизированных данных в тензоры
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test_scaled, dtype=torch.float32)

# Построение многослойного персептрона с одним скрытым слоем и произвольным количеством нейронов
class MLP(nn.Module):
    def __init__(self, hidden_neurons=10):  # Задаем количество нейронов в скрытом слое через параметр
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, hidden_neurons)  # Скрытый слой с H нейронами
        self.output = nn.Linear(hidden_neurons, 3)  # Выходной слой для трех переменных
        
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

# Задаем количество нейронов в скрытом слое (например, 10)
hidden_neurons = 10
model = MLP(hidden_neurons=hidden_neurons)

# Создание функции потерь и оптимизатора
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Увеличенное число эпох для улучшенной сходимости
num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Тестирование и предсказание
model.eval()
train_predictions_scaled = model(X_train_tensor).detach().numpy()
test_predictions_scaled = model(X_test_tensor).detach().numpy()

# Обратное преобразование данных в оригинальную шкалу
train_predictions = scaler.inverse_transform(train_predictions_scaled)
test_predictions = scaler.inverse_transform(test_predictions_scaled)

# Визуализация результатов для тренировочных и тестовых данных
for data_type, true_data, pred_data, data_label in [
    ("тестовых", Y_train, train_predictions, "Train"),
    ("тренировачных", Y_test, test_predictions, "Test")
]:
    for i, var in enumerate(["X", "Y", "Z"]):
        plt.figure(figsize=(12, 6))

        # Построение графиков для каждой переменной X(t+Δt), Y(t+Δt), Z(t+Δt)
        if data_label == "Train":
            plt.plot(range(len(true_data)), true_data[:, i], label='График эталонной функции')
            plt.plot(range(len(pred_data)), pred_data[:, i], label='График аппроксимированной функции', linestyle='--')
        else:
            plt.plot(range(len(true_data)), true_data[:, i], label='График эталонной функции')
            plt.plot(range(len(pred_data)), pred_data[:, i], label='График спрогнозированной функции', linestyle='--')
        
        plt.xlabel('Шаги')
        plt.ylabel(f'{var}(t+Δt)')
        plt.title(f'{var}(t+Δt) на {data_type} данных')
        
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.show()