# Ideal, Natural, & Flat-top -Sampling

# Aim
Write a simple Python program for the construction and reconstruction of ideal, natural, and flattop sampling.
s

# Tools required
Google colab

# Program

# 1.Impulse sampling

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import resample

fs = 100

t = np.arange(0, 1, 1/fs) 

f = 5

signal = np.sin(2 * np.pi * f * t)

plt.figure(figsize=(10, 4))

plt.plot(t, signal, label='Continuous Signal')

plt.title('Continuous Signal (fs = 100 Hz)')

plt.xlabel('Time [s]')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.show()

t_sampled = np.arange(0, 1, 1/fs)

signal_sampled = np.sin(2 * np.pi * f * t_sampled)

plt.figure(figsize=(10, 4))

#plt.plot(t, signal, label='Continuous Signal', alpha=0.7)

plt.stem(t_sampled, signal_sampled, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled Signal (fs = 100 Hz)')

plt.title('Sampling of Continuous Signal (fs = 100 Hz)')

plt.xlabel('Time [s]')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.show()

reconstructed_signal = resample(signal_sampled, len(t))

plt.figure(figsize=(10, 4))

#plt.plot(t, signal, label='Continuous Signal', alpha=0.7)

plt.plot(t, reconstructed_signal, 'r--', label='Reconstructed Signal (fs = 100 Hz)')

plt.title('Reconstruction of Sampled Signal (fs = 100 Hz)')

plt.xlabel('Time [s]')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.show()


import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import resample

fs = 100

t = np.arange(0, 1, 1/fs) 

f = 5

signal = np.sin(2 * np.pi * f * t)

plt.figure(figsize=(10, 4))

plt.plot(t, signal, label='Continuous Signal')

plt.title('Continuous Signal (fs = 100 Hz)')

plt.xlabel('Time [s]')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.show()

t_sampled = np.arange(0, 1, 1/fs)

signal_sampled = np.sin(2 * np.pi * f * t_sampled)

plt.figure(figsize=(10, 4))

#plt.plot(t, signal, label='Continuous Signal', alpha=0.7)

plt.stem(t_sampled, signal_sampled, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled Signal (fs = 100 Hz)')

plt.title('Sampling of Continuous Signal (fs = 100 Hz)')

plt.xlabel('Time [s]')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.show()

reconstructed_signal = resample(signal_sampled, len(t))

plt.figure(figsize=(10, 4))

#plt.plot(t, signal, label='Continuous Signal', alpha=0.7)

plt.plot(t, reconstructed_signal, 'r--', label='Reconstructed Signal (fs = 100 Hz)')

plt.title('Reconstruction of Sampled Signal (fs = 100 Hz)')

plt.xlabel('Time [s]')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.show()


# 2. Natural sampling
   
import numpy as np
import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter


fs = 1000  # Sampling frequency (samples per second)

T = 1      # Duration in seconds

t = np.arange(0, T, 1/fs)  # Time vector

fm = 5     # Frequency of message signal (Hz)

message_signal = np.sin(2 * np.pi * fm * t)

pulse_rate = 50  # pulses per second

pulse_train_indices = np.arange(0, len(t), int(fs / pulse_rate))

pulse_train = np.zeros_like(t)

pulse_train[pulse_train_indices] = 1

flat_top_signal = np.zeros_like(t)

sample_times = t[pulse_train_indices]

pulse_width_samples = int(fs / (2 * pulse_rate)) # Adjust pulse width as needed


for i, sample_time in enumerate(sample_times):
    index = np.argmin(np.abs(t - sample_time))
    if index < len(message_signal):
        sample_value = message_signal[index]
        start_index = index
        end_index = min(index + pulse_width_samples, len(t))
        flat_top_signal[start_index:end_index] = sample_value

def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

cutoff_freq = 2 * fm  # Nyquist rate or slightly higher

reconstructed_signal = lowpass_filter(flat_top_signal, cutoff_freq, fs)

plt.figure(figsize=(14, 10))


plt.subplot(4, 1, 1)

plt.plot(t, message_signal, label='Original Message Signal')

plt.title('Original Message Signal')

plt.xlabel('Time (s)')

plt.ylabel('Amplitude')

plt.legend()

plt.grid(True)


plt.subplot(4, 1, 2)

plt.stem(t[pulse_train_indices], pulse_train[pulse_train_indices], basefmt=" ", label='Ideal Sampling Instances')

plt.title('Ideal Sampling Instances')

plt.xlabel('Time (s)')

plt.ylabel('Amplitude')

plt.legend()

plt.grid(True)


plt.subplot(4, 1, 3)

plt.plot(t, flat_top_signal, label='Flat-Top Sampled Signal')

plt.title('Flat-Top Sampled Signal')

plt.xlabel('Time (s)')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()


plt.subplot(4, 1, 4)

plt.plot(t, reconstructed_signal, label=f'Reconstructed Signal (Low-pass Filter, Cutoff={cutoff_freq} Hz)', color='green')

plt.title('Reconstructed Signal')

plt.xlabel('Time (s)')

plt.ylabel('Amplitude')

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# 3. Flat-top sampling

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter


fs = 1000  # Sampling frequency (samples per second)

T = 1      # Duration in seconds

t = np.arange(0, T, 1/fs)  # Time vector

fm = 5     # Frequency of message signal (Hz)

message_signal = np.sin(2 * np.pi * fm * t)

pulse_rate = 50  # pulses per second

pulse_train_indices = np.arange(0, len(t), int(fs / pulse_rate))

pulse_train = np.zeros_like(t)

pulse_train[pulse_train_indices] = 1

flat_top_signal = np.zeros_like(t)

sample_times = t[pulse_train_indices]

pulse_width_samples = int(fs / (2 * pulse_rate)) # Adjust pulse width as needed


for i, sample_time in enumerate(sample_times):
    index = np.argmin(np.abs(t - sample_time))
    if index < len(message_signal):
        sample_value = message_signal[index]
        start_index = index
        end_index = min(index + pulse_width_samples, len(t))
        flat_top_signal[start_index:end_index] = sample_value

def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

cutoff_freq = 2 * fm  # Nyquist rate or slightly higher

reconstructed_signal = lowpass_filter(flat_top_signal, cutoff_freq, fs)

plt.figure(figsize=(14, 10))


plt.subplot(4, 1, 1)

plt.plot(t, message_signal, label='Original Message Signal')

plt.title('Original Message Signal')

plt.xlabel('Time (s)')

plt.ylabel('Amplitude')

plt.legend()

plt.grid(True)


plt.subplot(4, 1, 2)

plt.stem(t[pulse_train_indices], pulse_train[pulse_train_indices], basefmt=" ", label='Ideal Sampling Instances')

plt.title('Ideal Sampling Instances')

plt.xlabel('Time (s)')

plt.ylabel('Amplitude')

plt.legend()

plt.grid(True)


plt.subplot(4, 1, 3)

plt.plot(t, flat_top_signal, label='Flat-Top Sampled Signal')

plt.title('Flat-Top Sampled Signal')

plt.xlabel('Time (s)')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()


plt.subplot(4, 1, 4)

plt.plot(t, reconstructed_signal, label=f'Reconstructed Signal (Low-pass Filter, Cutoff={cutoff_freq} Hz)', color='green')

plt.title('Reconstructed Signal')

plt.xlabel('Time (s)')

plt.ylabel('Amplitude')

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# Output Waveform

1.Impulse sampling output waveform
<img width="866" height="393" alt="Exp1a-DC" src="https://github.com/user-attachments/assets/ac34218e-52fb-4357-9f7b-836eec5395b8" />


2. Natural sampling Output waveform
<img width="1390" height="989" alt="Exp1b-DC" src="https://github.com/user-attachments/assets/25073906-810c-4f36-b6d1-87445b50301a" />


3. Flat-top sampling Output waveform
<img width="1398" height="990" alt="Exp1-DC" src="https://github.com/user-attachments/assets/060a2ff9-d46e-45f1-9b0b-5b395eda78ac" />



# Results

Impulse sampling gives perfect reconstruction, while natural and flat-top sampling allow approximate reconstruction, with flat-top introducing slight amplitude distortion.
