import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def my_fft(x):
    N = len(x)
    if N == 1:
        return x.astype(complex)
    if N % 2 != 0:
        raise ValueError("N должно быть степенью двойки")
    even = my_fft(x[0::2])
    odd  = my_fft(x[1::2])
    W = np.exp(-2j * np.pi * np.arange(N // 2) / N)
    T = W * odd
    return np.concatenate([even + T, even - T])

def compute_spectrum(signal, fs, max_N=16384):
    if len(signal) > max_N:
        signal = signal[:max_N]
    N = 2 ** int(np.floor(np.log2(len(signal))))
    signal = signal[:N]
    spec = my_fft(signal)
    amp = np.abs(spec) / N
    freq = np.arange(N) * (fs / N)
    half = N // 2
    return freq[:half], amp[:half]

fs, y = wavfile.read("voice.wav")

if y.ndim > 1:
    y = y[:, 0]
y = y.astype(np.float64)

y1 = y[::10]
fs_10 = fs // 10

freq_o, amp_o = compute_spectrum(y, fs)
freq_d, amp_d = compute_spectrum(y1, fs_10)

porog_o = amp_o.max() * 0.02
idx_o = np.where(amp_o > porog_o)[0][-1]
f_max_o = freq_o[idx_o]

porog_d = amp_d.max() * 0.02
idx_d = np.where(amp_d > porog_d)[0][-1]
f_max_d = freq_d[idx_d]

print(f"Оригинал:    f_max = {f_max_o:.1f} Гц, требуется fs >= {2*f_max_o:.1f} Гц, реальная fs = {fs} Гц")
print(f"Прореженный: f_max = {f_max_d:.1f} Гц, требуется fs >= {2*f_max_d:.1f} Гц, реальная fs = {fs_10} Гц")

plt.figure(1, figsize=(10, 5))
plt.plot(freq_o, amp_o, 'b-', linewidth=0.7)
plt.title(f'П.12. Спектр оригинала (fs = {fs} Гц)')
plt.xlabel('Частота (Гц)')
plt.ylabel('|C(k)|')
plt.xlim(0, 12000)
plt.grid(True)
plt.savefig("graph8_spectrum_original.png")
plt.close()

plt.figure(2, figsize=(10, 5))
plt.plot(freq_d, amp_d, 'r-', linewidth=0.7)
plt.title(f'П.12. Спектр прореженного (fs = {fs_10} Гц)')
plt.xlabel('Частота (Гц)')
plt.ylabel('|C(k)|')
plt.xlim(0, 2400)
plt.grid(True)
plt.savefig("graph9_spectrum_downsampled.png")
plt.close()

plt.figure(3, figsize=(10, 5))
plt.plot(freq_o, amp_o, 'b-', linewidth=0.7, label=f'Оригинал (fs = {fs} Гц)')
plt.plot(freq_d, amp_d, 'r-', linewidth=0.7, alpha=0.7,
         label=f'Прореженный (fs = {fs_10} Гц)')
plt.title('П.12. Сравнение спектров')
plt.xlabel('Частота (Гц)')
plt.ylabel('|C(k)|')
plt.xlim(0, 6000)
plt.legend()
plt.grid(True)
plt.savefig("graph10_spectrum_compare.png")
plt.close()
