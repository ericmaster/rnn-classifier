import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio.functional as F_audio
import random
import numpy as np

class AudioAugmentations(nn.Module):
    """
    Implementación simple y eficiente de augmentations de audio
    """
    
    def __init__(
        self, 
        apply_prob: float = 0.5,
        gain_range: tuple = (0.8, 1.2),
        noise_snr_range: tuple = (20, 40),
        sample_rate: int = 22050,
        time_stretch_range: tuple = (0.9, 1.1)
    ):
        super().__init__()
        self.apply_prob = apply_prob
        self.gain_range = gain_range
        self.noise_snr_range = noise_snr_range
        self.sample_rate = sample_rate
        self.time_stretch_range = time_stretch_range
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Aplica una sola augmentation aleatoria por llamada
        Args:
            waveform: [channels, samples]
        Returns:
            waveform augmentado
        """
        if random.random() > self.apply_prob:
            return waveform
            
        # Elegir una augmentation aleatoria
        aug_type = random.choice(['gain', 'noise', 'time_stretch'])
        
        if aug_type == 'gain':
            return self._apply_gain(waveform)
        elif aug_type == 'noise':
            return self._add_noise(waveform)
        else:
            return self._time_stretch(waveform)
    
    def _apply_gain(self, waveform: torch.Tensor) -> torch.Tensor:
        """Aplica ganancia simple"""
        gain = random.uniform(*self.gain_range)
        return waveform * gain
    
    def _add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Añade ruido gaussiano simple"""
        # Calcular SNR objetivo
        snr_db = random.uniform(*self.noise_snr_range)
        
        # Calcular potencia de la señal
        signal_power = torch.mean(waveform ** 2)
        
        # Calcular potencia del ruido necesaria
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise_std = torch.sqrt(noise_power)
        
        # Generar ruido
        noise = torch.randn_like(waveform) * noise_std
        
        return waveform + noise

    def _time_stretch(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Aplica time stretching usando resampling
        Modifica la velocidad de reproducción sin cambiar el pitch
        """
        # Factor de velocidad aleatorio (0.9x - 1.1x para cambios sutiles)
        speed_factor = random.uniform(*self.time_stretch_range)
        
        # Si el factor es muy cercano a 1.0, no aplicar transformación
        if abs(speed_factor - 1.0) < 0.01:
            return waveform
            
        # Calcular el nuevo sample rate objetivo
        target_sr = int(self.sample_rate * speed_factor)
        
        if target_sr == self.sample_rate:
            return waveform
        
        try:
            # Crear resamplers para el time stretching
            # Paso 1: Resample a la nueva velocidad
            resampler_stretch = T.Resample(
                orig_freq=self.sample_rate, 
                new_freq=target_sr,
                resampling_method='sinc_interpolation'
            )
            
            # Paso 2: Resample de vuelta al sample rate original
            resampler_back = T.Resample(
                orig_freq=target_sr, 
                new_freq=self.sample_rate,
                resampling_method='sinc_interpolation'
            )
            
            # Aplicar time stretching
            stretched = resampler_stretch(waveform)
            result = resampler_back(stretched)
            
            return result
            
        except Exception:
            # En caso de error, devolver el waveform original
            return waveform


class SpectrogramAugmentations(nn.Module):
    """
    Implementación simple de SpecAugment sin dependencias externas
    """
    
    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 20,
        apply_prob: float = 0.5
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.apply_prob = apply_prob
        
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Aplica una máscara aleatoria al espectrograma
        Args:
            spectrogram: [channels, freq_bins, time_frames]
        Returns:
            spectrogram con máscara aplicada
        """
        if random.random() > self.apply_prob:
            return spectrogram
            
        # Elegir tipo de máscara
        if random.random() < 0.5:
            return self._apply_freq_mask(spectrogram)
        else:
            return self._apply_time_mask(spectrogram)
    
    def _apply_freq_mask(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Aplica máscara de frecuencia manualmente
        """
        # Crear una copia para no modificar el original
        masked_spec = spectrogram.clone()
        
        freq_bins = spectrogram.size(-2)
        
        # Calcular tamaño de la máscara
        mask_size = random.randint(0, min(self.freq_mask_param, freq_bins // 4))
        
        if mask_size == 0:
            return masked_spec
            
        # Calcular posición de inicio
        mask_start = random.randint(0, freq_bins - mask_size)
        mask_end = mask_start + mask_size
        
        # Aplicar máscara (poner a cero)
        masked_spec[..., mask_start:mask_end, :] = 0
        
        return masked_spec
    
    def _apply_time_mask(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Aplica máscara de tiempo manualmente
        """
        # Crear una copia para no modificar el original
        masked_spec = spectrogram.clone()
        
        time_frames = spectrogram.size(-1)
        
        # Calcular tamaño de la máscara
        mask_size = random.randint(0, min(self.time_mask_param, time_frames // 4))
        
        if mask_size == 0:
            return masked_spec
            
        # Calcular posición de inicio
        mask_start = random.randint(0, time_frames - mask_size)
        mask_end = mask_start + mask_size
        
        # Aplicar máscara (poner a cero)
        masked_spec[..., mask_start:mask_end] = 0
        
        return masked_spec

class EnvironmentalAugmentations(nn.Module):
    """
    Augmentaciones específicas para sonidos ambientales optimizadas
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        apply_prob: float = 0.2  # Reducido para menos overhead
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.apply_prob = apply_prob
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Aplica augmentaciones ambientales simplificadas"""
        if random.random() > self.apply_prob:
            return waveform
            
        # Solo usar speed perturbation que es más eficiente
        return self._speed_perturbation(waveform)
    
    def _speed_perturbation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Perturbación de velocidad simplificada"""
        # Factor de velocidad aleatorio muy sutil
        speed_factor = random.uniform(0.98, 1.02)
        target_sr = int(self.sample_rate * speed_factor)
        
        if target_sr == self.sample_rate:
            return waveform
            
        # Usar interpolación simple en lugar de resample completo
        if abs(speed_factor - 1.0) < 0.01:  # Cambio muy pequeño
            return waveform
            
        # Para cambios muy pequeños, usar interpolación linear
        original_length = waveform.size(-1)
        new_length = int(original_length / speed_factor)
        
        if new_length == original_length:
            return waveform
            
        # Resample usando interpolación más eficiente
        resampled = torch.nn.functional.interpolate(
            waveform.unsqueeze(0), size=new_length, mode='linear', align_corners=False
        ).squeeze(0)
        
        # Padding o truncate para mantener el tamaño original
        if new_length > original_length:
            resampled = resampled[:, :original_length]
        elif new_length < original_length:
            pad_amount = original_length - new_length
            resampled = torch.nn.functional.pad(resampled, (0, pad_amount))
            
        return resampled

class MixupAugmentation(nn.Module):
    """
    Implementa Mixup para espectrogramas de forma eficiente
    """
    
    def __init__(self, alpha: float = 0.2, apply_prob: float = 0.3):  # Reducido apply_prob
        super().__init__()
        self.alpha = alpha
        self.apply_prob = apply_prob
        
    def forward(self, batch_spectrograms: torch.Tensor, batch_labels: torch.Tensor):
        """
        Aplica mixup a un batch de espectrogramas de forma optimizada
        Args:
            batch_spectrograms: [batch_size, channels, freq, time]
            batch_labels: [batch_size, num_classes]
        Returns:
            tuple (mixed_spectrograms, mixed_labels)
        """
        if random.random() > self.apply_prob:
            return batch_spectrograms, batch_labels
            
        return self._apply_mixup(batch_spectrograms, batch_labels)
    
    def _apply_mixup(self, spectrograms: torch.Tensor, labels: torch.Tensor):
        """Aplica mixup in-place cuando es posible"""
        batch_size = spectrograms.size(0)
        
        # Generar lambda de distribución Beta
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        # Permutar índices para mezclar
        index = torch.randperm(batch_size, device=spectrograms.device)
        
        # Mezclar espectrogramas in-place para ahorrar memoria
        spectrograms.mul_(lam).add_(spectrograms[index], alpha=1-lam)
        
        # Mezclar labels
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        
        return spectrograms, mixed_labels


class LightweightAugmentations(nn.Module):
    """
    Versión ultraligera de augmentations para máximo ahorro de memoria
    Solo incluye las transformaciones más esenciales y eficientes
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        apply_prob: float = 0.5,
        gain_prob: float = 0.3,
        noise_prob: float = 0.2
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.apply_prob = apply_prob
        self.gain_prob = gain_prob
        self.noise_prob = noise_prob
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Aplica solo las augmentations más básicas y eficientes"""
        if random.random() > self.apply_prob:
            return waveform
            
        # Solo aplicar una augmentation por vez
        if random.random() < self.gain_prob:
            # Ganancia simple
            gain = random.uniform(0.7, 1.3)
            waveform.mul_(gain)
        elif random.random() < self.noise_prob:
            # Ruido blanco simple
            noise_scale = random.uniform(0.001, 0.01)
            noise = torch.randn_like(waveform) * noise_scale
            waveform.add_(noise)
            
        return waveform


class SimpleSpecAugment(nn.Module):
    """
    Versión simplificada de SpecAugment que usa menos memoria
    """
    
    def __init__(
        self,
        freq_mask_param: int = 8,
        time_mask_param: int = 15,
        apply_prob: float = 0.4
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.apply_prob = apply_prob
        
        # Pre-crear las máscaras una sola vez
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)
        
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Aplica solo una máscara por vez para ahorrar memoria"""
        if random.random() > self.apply_prob:
            return spectrogram
            
        # Aplicar solo una de las dos máscaras
        if random.random() < 0.5:
            return self.freq_masking(spectrogram)
        else:
            return self.time_masking(spectrogram)
