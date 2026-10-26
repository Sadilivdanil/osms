import numpy as np
import matplotlib.pyplot as plt

f = 14
fs = 3 * f 
N = fs

t = np.linspace(0, 1, 1000)
y = 12 * np.cos(2 * np.pi * f * t + np.pi / 6)

sample_t = [i / fs for i in range(N)]
sample_y = [12 * np.cos(2 * np.pi * f * ti + np.pi / 6) for ti in sample_t]

spectrum = []
for x in range(N):
    sum_real = sum(sample_y[n] * np.cos(2 * np.pi * x * n / N) for n in range(N))
    sum_imag = sum(sample_y[n] * (-np.sin(2 * np.pi * x * n / N)) for n in range(N))
    spectrum.append(complex(sum_real, sum_imag))

amplitudes = [abs(k) / N for k in spectrum]
freq = np.arange(N) * (fs / N)

porog = max(amplitudes) * 0.1
sush_freq = [fr for fr, a in zip(freq, amplitudes) if a > porog]
total_bytes = N * 8

print(*sample_t, '\n')
print(*sample_y, '\n')
print("Freq:", freq)
print("Amplitudes:", amplitudes)
print("Significant freq:", sush_freq)
print("Total bytes:", total_bytes)

plt.figure(1, figsize=(10, 5))
plt.plot(t, y, 'b-', label='Непрерывный сигнал')
plt.stem(sample_t, sample_y, linefmt='-r', markerfmt='ro', basefmt=' ', label='Отсчеты')
plt.title('Дискретизация сигнала по времени')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid(True)
plt.savefig("graph1_discretization.png")
plt.close()

plt.figure(2, figsize=(10, 5))
plt.stem(freq, amplitudes, linefmt='-g', markerfmt='go', basefmt=' ')
plt.title('Спектр')
plt.xlabel('Частота (Гц)')
plt.ylabel('|C(k)|')
plt.xlim(0, fs / 2)
plt.xticks(np.arange(0, int(fs / 2) + 1, 10))
plt.grid(True)
plt.savefig("graph2_spectrum.png")
plt.close()

plt.figure(3, figsize=(10, 5))
plt.plot(t, y, 'b-', label='Оригинальный сигнал')
plt.plot(sample_t, sample_y, 'r.-', label='Восстановлено по отсчётам')
plt.title('Оригинал vs восстановленный из отсчётов')
plt.legend()
plt.grid(True)
plt.savefig("graph3_reconstruction.png")
plt.close()