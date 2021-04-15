from .dataset_utils import BASE_SAMPLE_RATE
import librosa
import numpy as np
import torch
import math

def apply_all_audio_transformations(x, e0):
    return gaussian_noise_injection(pitch_shift(x, e0), e0)

def gaussian_noise_injection(signal, SNR, is_tp):
    if SNR == 0: return signal
    if is_tp:
        SNR = 12 - SNR * 100

    RMS_s = math.sqrt(np.mean(signal.numpy()**2))
    RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
    STD_n=RMS_n
    noise = np.random.normal(0, STD_n, signal.shape[0])
    return signal + noise

def pitch_shift(signal, factor, is_tp):
    if is_tp: factor = factor * 100
    return librosa.effects.pitch_shift(signal.squeeze().numpy(), BASE_SAMPLE_RATE, factor)
