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

class MLP(nn.Module):
    def __init__(self, n_neurons, r_neurons):  # Задаем количество нейронов в распределительном и скрытом слоях
        super(MLP, self).__init__()
        self.distributed = nn.Linear(3, n_neurons)  # Распределительный слой
        self.hidden = nn.Linear(n_neurons, r_neurons)  # Скрытый слой
        self.output = nn.Linear(r_neurons, 1)  # Выходной слой с одним нейроном
        
    def forward(self, x):
        x = torch.relu(self.distributed(x))  # Применение ReLU после распределительного слоя
        x = torch.relu(self.hidden(x))  # Применение ReLU после скрытого слоя
        x = self.output(x)  # Выходной слой
        return x

# Задаем количество нейронов в распределительном и скрытом слоях
n_neurons = 15  # Например, 15 нейронов в распределительном слое
r_neurons = 10   # Например, 10 нейронов в скрытом слое
model = MLP(n_neurons=n_neurons, r_neurons=r_neurons)

# Создание функции потерь и оптимизатора
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor[:, 0].unsqueeze(1))  # Используем только 1 выходной нейрон
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Тестирование и предсказание
model.eval()
train_predictions_scaled = model(X_train_tensor).detach().numpy()
test_predictions_scaled = model(X_test_tensor).detach().numpy()

# Обратное преобразование данных в оригинальную шкалу
train_predictions = scaler.inverse_transform(np.concatenate([train_predictions_scaled, 
                                                            np.zeros_like(train_predictions_scaled), 
                                                            np.zeros_like(train_predictions_scaled)], axis=1))[:, 0]
test_predictions = scaler.inverse_transform(np.concatenate([test_predictions_scaled, 
                                                           np.zeros_like(test_predictions_scaled), 
                                                           np.zeros_like(test_predictions_scaled)], axis=1))[:, 0]

# Визуализация результатов только для X(t)
for data_type, true_data, pred_data, data_label in [
    ("тестовых", Y_train, train_predictions, "Train"),
    ("тренировачных", Y_test, test_predictions, "Test")
]:
    plt.figure(figsize=(12, 6))

    # График эталонных данных
    plt.plot(range(len(true_data[:, 0])), true_data[:, 0], label=f'График эталонной функции', linestyle='-')

    # График спрогнозированных данных
    plt.plot(range(len(pred_data)), pred_data, label=f'График спрогнозированной функции', linestyle='--')

    plt.xlabel('Шаги')
    plt.ylabel('X(t)')
    plt.title(f'X(t) на {data_type} данных')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()