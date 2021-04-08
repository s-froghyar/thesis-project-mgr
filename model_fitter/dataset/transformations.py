from .dataset_utils import BASE_SAMPLE_RATE
import librosa
import numpy as np

def apply_all_audio_transformations(x, e0):
    return gaussian_noise_injection(pitch_shift(x, e0), e0)

def gaussian_noise_injection(x, snr):
    g_noise = np.random.randn(len(x))

    noise_power = np.mean(np.power(g_noise, 2))
    sig_power = np.mean(np.power(x, 2))

    snr_linear = 10**(snr / 10.0)
    noise_factor = (sig_power / noise_power) * (1 / snr_linear)

    return x + np.sqrt(noise_factor) * g_noise

def pitch_shift(x, factor):
    return librosa.effects.pitch_shift(x.squeeze().numpy(), BASE_SAMPLE_RATE, factor * 100)
