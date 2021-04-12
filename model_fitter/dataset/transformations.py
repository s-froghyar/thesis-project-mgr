from .dataset_utils import BASE_SAMPLE_RATE
import librosa
import numpy as np
import torch
import math

def apply_all_audio_transformations(x, e0):
    return gaussian_noise_injection(pitch_shift(x, e0), e0)

def gaussian_noise_injection(signal, SNR, is_tp):
    # g_noise = np.random.randn(len(x))

    # noise_power = np.mean(np.power(g_noise, 2))
    # sig_power = np.mean(np.power(x, 2))

    # snr_linear = 10**(snr / 10.0)
    # noise_factor = (sig_power / noise_power) * (1 / snr_linear)


    RMS_s = math.sqrt(np.mean(signal.numpy()**2))
    #RMS values of noise
    RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
    #Additive white gausian noise. Thereore mean=0
    #Because sample length is large (typically > 40000)
    #we can use the population formula for standard daviation.
    #because mean=0 STD=RMS
    STD_n=RMS_n
    noise = np.random.normal(0, STD_n, signal.shape[0])
    return signal + noise

    # return x + np.sqrt(noise_factor) * g_noise

def pitch_shift(signal, factor, is_tp):
    if is_tp: factor = factor * 100
    return librosa.effects.pitch_shift(signal.squeeze().numpy(), BASE_SAMPLE_RATE, factor)
