import numpy as np
import matplotlib.pyplot as plt


class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for pattern in patterns:
            pattern_flat = np.reshape(pattern, (self.size,))
            self.weights += np.outer(pattern_flat, pattern_flat)
            np.fill_diagonal(self.weights, 0)

    def predict(self, input_pattern, max_iterations=100):
        input_pattern_flat = np.reshape(input_pattern, (self.size,))
        for _ in range(max_iterations):
            output = np.sign(np.dot(self.weights, input_pattern_flat))
            if np.array_equal(output, input_pattern_flat):
                return output
            input_pattern_flat = output
        return None


# Задаємо розмірність мережі та літери для розпізнавання
network_size = 25  # 5x5
letters_to_recognize = ['П', 'Х', 'Ч']

# Створюємо мережу Хопфілда
hopfield_net = HopfieldNetwork(network_size)

# Підготовка тренувальних даних
training_data = {
    'П': np.array([
        [1, 1, 1, 1, 1],
        [1, -1, -1, -1, 1],
        [1, -1, -1, -1, 1],
        [1, -1, -1, -1, 1],
        [1, -1, -1, -1, 1]
    ]),
    'Х': np.array([
        [1, -1, -1, -1, 1],
        [-1, 1, -1, 1, -1],
        [-1, -1, 1, -1, -1],
        [-1, 1, -1, 1, -1],
        [1, -1, -1, -1, 1]
    ]),
    'Ч': np.array([
        [-1, -1, 1, -1, 1],
        [-1, -1, 1, -1, 1],
        [-1, -1, 1, 1, 1],
        [-1, -1, -1, -1, 1],
        [-1, -1, -1, -1, 1]
    ])
}

# Тренуємо мережу
hopfield_net.train(training_data.values())

# Тестуємо розпізнавання літер та візуалізуємо результати з шумом
for letter in letters_to_recognize:
    test_input = training_data[letter]

    # Додаємо шум до вхідного зображення
    noise_level = 0.1
    noisy_input = np.copy(test_input)
    noisy_pixels = np.random.choice([1, -1], size=(network_size,), p=[1 - noise_level, noise_level])
    noisy_input = noisy_input.flatten() * noisy_pixels

    predicted_output = hopfield_net.predict(noisy_input)

    if predicted_output is not None:
        # Відображення оригінальної літери з шумом
        plt.imshow(noisy_input.reshape((5, 5)), cmap='binary')
        plt.title(f'Оригінальна літера "{letter}" з шумом')
        plt.show()

        # Відображення розпізнаної літери
        plt.imshow(predicted_output.reshape((5, 5)), cmap='binary')
        plt.title(f'Розпізнана літера "{letter}"')
        plt.show()
    else:
        print(f'Не вдалося розпізнати літеру "{letter}"\n')