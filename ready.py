import numpy as np
import pandas as pd
from tkinter import *
from tkinter import messagebox


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


class OurNeuralNetwork:

    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * len(x['contact_type']) + self.w2 * len(x['incident_state']) + self.b1)
        h2 = sigmoid(self.w3 * len(x['contact_type']) + self.w4 * len(x['incident_state']) + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * len(x['contact_type']) + self.w2 * len(x['incident_state']) + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * len(x['contact_type']) + self.w4 * len(x['incident_state']) + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # Вычисление частных производных
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Нейронн o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Нейрон h1
                d_h1_d_w1 = len(x['contact_type']) * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = len(x['incident_state']) * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Нейрон h2
                d_h2_d_w3 = len(x['contact_type']) * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = len(x['incident_state']) * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # Обновление данных
                # Нейрон h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Нейрон h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Нейрон o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3


def load_data(csv_file):
    df = pd.read_csv(csv_file)
    data = np.array([])
    for i, row in df.iterrows():
        data = np.append(data, {
            'contact_type': row['contact_type'],
            'incident_state': row['incident_state']
        })
    return data


def load_data_for_output(csv_file):
    df = pd.read_csv(csv_file)
    data = np.array([])
    for i, row in df.iterrows():
        data = np.append(data, {
            'number': row['number'],
            'active': row['active'],
            'caller_id': row['caller_id'],
            'contact_type': row['contact_type'],
            'incident_state': row['incident_state'],
        })
    return data


def load_data_true(csv_file):
    df = pd.read_csv(csv_file)
    data = np.array([])
    for i, row in df.iterrows():
        data = np.append(data, df.values[i])
    return data


# Определение набора данных
train_data = load_data('book-20-values.csv')

all_y_trues = load_data_true('true.csv')

# Тренируем нейронную сеть
network = OurNeuralNetwork()
network.train(train_data, all_y_trues)

input_data = load_data_for_output('book-50-values.csv')


# Вывод
def calculate_bmi():
    while True:
        j = 0
        for i in input_data:
            if network.feedforward(i) < 0.5:
                response = "Появился инцидент с номером: " + input_data[j]["number"] + " с типом контакта - " + \
                           input_data[j]["contact_type"] + " от участника: " + \
                           input_data[j]["caller_id"] + ": " + input_data[j]["incident_state"]
                messagebox.showinfo('Новый инцидент', response)
                print(response)
            j = j + 1
        break


window = Tk()  # Создаём окно приложения.
window.title("Поиск инцидентов")
window.geometry('400x300')

frame = Frame(
    window,
    padx=10,
    pady=10
)
frame.pack(expand=True)

cal_btn = Button(
    frame,
    text='Начать сканирование',
    command=calculate_bmi
)
cal_btn.grid(row=5, column=2)
window.mainloop()
