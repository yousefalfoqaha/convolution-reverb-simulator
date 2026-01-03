import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import convolve

sample_rate = 44100
duration = 0.5
t = np.linspace(0, duration, int(sample_rate * duration))

input_signal = np.zeros_like(t)
input_signal[1000] = 1.0


def create_simple_echo(delay_samples, decay):
    impulse_response = np.zeros(delay_samples + 1)
    impulse_response[0] = 1.0
    impulse_response[delay_samples] = decay
    return impulse_response


def create_reverb(decay_time, sample_rate):
    num_samples = int(decay_time * sample_rate)
    impulse_response = np.zeros(num_samples)
    impulse_response[0] = 1.0

    for i in range(1, num_samples):
        impulse_response[i] = np.random.randn() * 0.1 * np.exp(-5 * i / num_samples)

    return impulse_response


ir_echo = create_simple_echo(delay_samples=8820, decay=0.5)
ir_multi_echo = create_simple_echo(4410, 0.6)
temp = np.zeros(13230)
temp[: len(ir_multi_echo)] = ir_multi_echo
temp[8820] = 0.4
ir_multi_echo = temp

ir_reverb = create_reverb(decay_time=2.0, sample_rate=sample_rate)

output_echo = convolve(input_signal, ir_echo, mode="full")
output_multi = convolve(input_signal, ir_multi_echo, mode="full")
output_reverb = convolve(input_signal, ir_reverb, mode="full")

input1 = np.zeros_like(input_signal)
input1[1000] = 0.5
input2 = np.zeros_like(input_signal)
input2[1500] = 0.3

output_sum = convolve(input1 + input2, ir_echo, mode="full")
output1 = convolve(input1, ir_echo, mode="full")
output2 = convolve(input2, ir_echo, mode="full")
sum_of_outputs = output1 + output2

linearity_error = np.max(np.abs(output_sum - sum_of_outputs))
print(f"Linearity test error: {linearity_error:.2e} (should be ~0)")

shift_amount = 500
input_shifted = np.roll(input_signal, shift_amount)
output_original = convolve(input_signal, ir_echo, mode="full")
output_shifted = convolve(input_shifted, ir_echo, mode="full")

time_inv_error = np.max(np.abs(np.roll(output_original, shift_amount) - output_shifted))
print(f"Time-invariance test error: {time_inv_error:.2e} (should be ~0)")

fig, axes = plt.subplots(4, 2, figsize=(14, 12))

axes[0, 0].plot(ir_echo[:2000])
axes[0, 0].set_title("Simple Echo Impulse Response")
axes[0, 0].set_xlabel("Sample")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].grid(True)

axes[0, 1].plot(output_echo[:50000])
axes[0, 1].set_title("Output with Simple Echo")
axes[0, 1].set_xlabel("Sample")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].grid(True)

axes[1, 0].plot(ir_multi_echo[:20000])
axes[1, 0].set_title("Multi-Echo Impulse Response")
axes[1, 0].set_xlabel("Sample")
axes[1, 0].set_ylabel("Amplitude")
axes[1, 0].grid(True)

axes[1, 1].plot(output_multi[:50000])
axes[1, 1].set_title("Output with Multi-Echo")
axes[1, 1].set_xlabel("Sample")
axes[1, 1].set_ylabel("Amplitude")
axes[1, 1].grid(True)

axes[2, 0].plot(ir_reverb[:20000])
axes[2, 0].set_title("Reverb Impulse Response")
axes[2, 0].set_xlabel("Sample")
axes[2, 0].set_ylabel("Amplitude")
axes[2, 0].grid(True)

axes[2, 1].plot(output_reverb[:50000])
axes[2, 1].set_title("Output with Reverb")
axes[2, 1].set_xlabel("Sample")
axes[2, 1].set_ylabel("Amplitude")
axes[2, 1].grid(True)

axes[3, 0].plot(output_sum[:10000], label="h(x1 + x2)", alpha=0.7)
axes[3, 0].plot(sum_of_outputs[:10000], "--", label="h(x1) + h(x2)", alpha=0.7)
axes[3, 0].set_title("Linearity Verification")
axes[3, 0].set_xlabel("Sample")
axes[3, 0].set_ylabel("Amplitude")
axes[3, 0].legend()
axes[3, 0].grid(True)

axes[3, 1].plot(
    np.roll(output_original, shift_amount)[:10000], label="Shifted h(x)", alpha=0.7
)
axes[3, 1].plot(output_shifted[:10000], "--", label="h(shifted x)", alpha=0.7)
axes[3, 1].set_title("Time-Invariance Verification")
axes[3, 1].set_xlabel("Sample")
axes[3, 1].set_ylabel("Amplitude")
axes[3, 1].legend()
axes[3, 1].grid(True)

plt.tight_layout()
plt.savefig("reverb_analysis.png", dpi=150, bbox_inches="tight")
print("\nPlots saved as 'reverb_analysis.png'")
plt.show()


def save_audio(signal, filename, sample_rate):
    normalized = signal / np.max(np.abs(signal))
    audio_int = np.int16(normalized * 32767)
    wavfile.write(filename, sample_rate, audio_int)


save_audio(output_echo, "output_echo.wav", sample_rate)
save_audio(output_reverb, "output_reverb.wav", sample_rate)
print("\nAudio files saved: output_echo.wav, output_reverb.wav")
