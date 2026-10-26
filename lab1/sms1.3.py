import numpy as np
import matplotlib.pyplot as plt

def my_fft(x):
    N = len(x)
    if N == 1:
        return np.asarray(x, dtype=complex)

    if N % 2 == 0:
        even = my_fft(x[::2])
        odd = my_fft(x[1::2])
        W = np.exp(-2j * np.pi * np.arange(N // 2) / N)
        T = W * odd
        return np.concatenate([even + T, even - T])

    for p in range(3, int(np.sqrt(N)) + 1, 2):
        if N % p == 0:
            M = N // p
            parts = [my_fft(x[r::p]) for r in range(p)]
            return np.array([
                sum(parts[r][k % M] * np.exp(-2j * np.pi * r * k / N)
                    for r in range(p))
                for k in range(N)
            ])

    W = np.exp(-2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N)
    return W @ x


def quantize(y, bits):
    levels = 2 ** bits
    a, b = y.min(), y.max()
    q = np.round((y - a) / (b - a) * (levels - 1))
    return q / (levels - 1) * (b - a) + a


f = 6
fs = 12 * f
t = np.arange(0, 1, 1 / fs)
y = 4 * np.sin(2 * np.pi * f * t + np.pi / 3)

N = len(y)
freq = np.arange(N) * fs / N
half = N // 2 + 1
amp = np.abs(my_fft(y)) / N

bits_list = [3, 4, 5, 6]
errors = []
spectrums = []

for bits in bits_list:
    yq = quantize(y, bits)
    spectrums.append(np.abs(my_fft(yq)) / N)
    errors.append(np.mean(np.abs(y - yq)))
    print(f"{bits} бит: средняя ошибка = {errors[-1]:.6f}")


plt.figure(1, figsize=(10, 10))
for i, bits in enumerate(bits_list):
    plt.subplot(4, 1, i + 1)
    plt.plot(t, y, 'g-o', markersize=3, label="Исходный")
    plt.plot(t, quantize(y, bits), 'b-o', markersize=3, label=f"{bits} бит")
    plt.title(f"Квантование сигнала ({bits} бит)")
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.savefig("graph11_quantization.png", dpi=300)
plt.close()


plt.figure(2, figsize=(10, 10))
for i, bits in enumerate(bits_list):
    plt.subplot(4, 1, i + 1)
    plt.plot(freq[:half], amp[:half], 'b-', label="Исходный")
    plt.plot(freq[:half], spectrums[i][:half], 'r-', label=f"{bits} бит")
    plt.title(f"Спектр сигнала ({bits} бит)")
    plt.xlabel("Частота (Гц)")
    plt.ylabel("|C(k)|")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.savefig("graph12_spectrum_quantization.png", dpi=300)
plt.close()


plt.figure(3, figsize=(10, 5))
plt.plot(bits_list, errors, 'o-')
plt.title("Ошибка квантования")
plt.xlabel("Разрядность АЦП (бит)")
plt.ylabel("Средняя ошибка")
plt.xticks(bits_list)
plt.grid(True)
plt.tight_layout()
plt.savefig("graph13_quantization_error.png", dpi=300)
plt.close()
