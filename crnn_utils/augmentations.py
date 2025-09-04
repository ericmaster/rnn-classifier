import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio.functional as F_audio
import random
import numpy as np

class AudioAugmentations(nn.Module):
    """
    Implementa técnicas de data augmentation para audio
    """
    
    def __init__(
        self, 
        sample_rate: int = 22050,
        apply_prob: float = 0.8,
        noise_prob: float = 0.4,
        gain_prob: float = 0.3,
        time_stretch_prob: float = 0.2
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.apply_prob = apply_prob
        self.noise_prob = noise_prob
        self.gain_prob = gain_prob
        self.time_stretch_prob = time_stretch_prob
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Aplica augmentations en el dominio temporal
        Args:
            waveform: tensor de forma [channels, time]
        Returns:
            waveform aumentado
        """
        if random.random() > self.apply_prob:
            return waveform
            
        # Aplicar augmentations secuencialmente con probabilidades independientes
        if random.random() < self.gain_prob:
            waveform = self._apply_volume_gain(waveform)
            
        if random.random() < self.noise_prob:
            waveform = self._add_gaussian_noise(waveform)
            
        if random.random() < self.time_stretch_prob:
            waveform = self._time_stretch(waveform)
            
        return waveform
    
    def _apply_volume_gain(self, waveform: torch.Tensor) -> torch.Tensor:
        """Aplica ganancia de volumen"""
        # Ganancia aleatoria entre -6dB y +6dB
        gain_db = random.uniform(-6.0, 6.0)
        
        # Aplicar ganancia manualmente (más seguro)
        gain_linear = 10 ** (gain_db / 20.0)
        return waveform * gain_linear
    
    def _add_gaussian_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Añade ruido gaussiano"""
        # Generar SNR aleatorio entre 10-30 dB
        snr_db = random.uniform(10, 30)
        
        # Agregar ruido manualmente (más seguro)
        noise = torch.randn_like(waveform)
        signal_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(noise ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_scaled = noise * torch.sqrt(signal_power / (snr_linear * noise_power))
        return waveform + noise_scaled
    
    def _time_stretch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Estiramiento temporal usando resample"""
        # Factor de velocidad aleatorio (0.9x - 1.1x para cambios sutiles)
        speed_factor = random.uniform(0.9, 1.1)
        
        # Usar resample para time stretching (más seguro)
        target_sr = int(self.sample_rate * speed_factor)
        if target_sr == self.sample_rate:
            return waveform
            
        resampler = T.Resample(self.sample_rate, target_sr)
        resampler_back = T.Resample(target_sr, self.sample_rate)
        stretched = resampler(waveform)
        stretched = resampler_back(stretched)
        return stretched


class SpectrogramAugmentations(nn.Module):
    """
    Implementa SpecAugment
    """
    
    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        apply_prob: float = 0.5
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.apply_prob = apply_prob
        
        # Inicializar transformaciones de SpecAugment
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)
        
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Aplica SpecAugment en el dominio frecuencial
        Args:
            spectrogram: tensor de forma [channels, freq, time]
        Returns:
            spectrogram aumentado
        """
        if random.random() > self.apply_prob:
            return spectrogram
        
        augmented_spec = spectrogram.clone()
        
        # Aplicar SpecAugment
        augmented_spec = self._apply_spec_augment(augmented_spec)
        
        return augmented_spec
    
    def _apply_spec_augment(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Aplica frequency masking y time masking"""
        # Aplicar múltiples frequency masks
        for _ in range(self.num_freq_masks):
            spectrogram = self.freq_masking(spectrogram)
        
        # Aplicar múltiples time masks
        for _ in range(self.num_time_masks):
            spectrogram = self.time_masking(spectrogram)
            
        return spectrogram

class EnvironmentalAugmentations(nn.Module):
    """
    Augmentaciones específicas para sonidos ambientales del ESC-50
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        apply_prob: float = 0.3
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.apply_prob = apply_prob
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Aplica augmentaciones ambientales"""
        if random.random() > self.apply_prob:
            return waveform
            
        # Seleccionar técnica relevante para sonidos ambientales
        techniques = [
            self._background_noise_mixing,
            self._speed_perturbation
        ]
        
        technique = random.choice(techniques)
        return technique(waveform)
    
    def _background_noise_mixing(self, waveform: torch.Tensor) -> torch.Tensor:
        """Mezcla con ruido de fondo ambiental"""
        # Generar ruido coloreado (más realista que ruido blanco)
        noise_type = random.choice(['pink', 'brown', 'white'])
        
        if noise_type == 'pink':
            # Ruido rosa (1/f)
            noise = self._generate_pink_noise(waveform.shape)
        elif noise_type == 'brown':
            # Ruido marrón (1/f²)
            noise = self._generate_brown_noise(waveform.shape)
        else:
            # Ruido blanco
            noise = torch.randn_like(waveform)
        
        # Nivel de ruido aleatorio
        noise_level = random.uniform(0.01, 0.1)
        noise = noise * noise_level
        
        # Mezclar con el audio original
        mixed_waveform = waveform + noise
        
        # Normalizar para evitar clipping
        max_val = torch.max(torch.abs(mixed_waveform))
        if max_val > 1.0:
            mixed_waveform = mixed_waveform / max_val
            
        return mixed_waveform
    
    def _generate_pink_noise(self, shape):
        """Genera ruido rosa usando filtrado de ruido blanco"""
        white_noise = torch.randn(shape)
        
        # Filtro simple para ruido rosa
        if len(shape) == 2:
            # Para cada canal
            pink_noise = torch.zeros_like(white_noise)
            for ch in range(shape[0]):
                # Aplicar filtro de diferencia simple
                filtered = torch.zeros(shape[1])
                filtered[0] = white_noise[ch, 0]
                for i in range(1, shape[1]):
                    filtered[i] = 0.99 * filtered[i-1] + 0.01 * white_noise[ch, i]
                pink_noise[ch] = filtered
        else:
            # Para un solo canal
            pink_noise = torch.zeros_like(white_noise)
            pink_noise[0] = white_noise[0]
            for i in range(1, len(white_noise)):
                pink_noise[i] = 0.99 * pink_noise[i-1] + 0.01 * white_noise[i]
                
        return pink_noise
    
    def _generate_brown_noise(self, shape):
        """Genera ruido marrón usando integración de ruido blanco"""
        white_noise = torch.randn(shape)
        brown_noise = torch.cumsum(white_noise, dim=-1)
        
        # Normalizar
        brown_noise = brown_noise / torch.std(brown_noise)
        
        return brown_noise
    
    def _speed_perturbation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Perturbación de velocidad para simular variaciones naturales"""
        # Factor de velocidad aleatorio (0.95x - 1.05x para cambios muy sutiles)
        speed_factor = random.uniform(0.95, 1.05)
        
        # Usar resample para speed perturbation (más seguro)
        target_sr = int(self.sample_rate * speed_factor)
        if target_sr == self.sample_rate:
            return waveform
            
        resampler = T.Resample(self.sample_rate, target_sr)
        resampler_back = T.Resample(target_sr, self.sample_rate)
        perturbed = resampler(waveform)
        perturbed = resampler_back(perturbed)
        return perturbed

class MixupAugmentation(nn.Module):
    """
    Implementa Mixup para espectrogramas
    """
    
    def __init__(self, alpha: float = 0.2, apply_prob: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.apply_prob = apply_prob
        
    def forward(self, batch_spectrograms: torch.Tensor, batch_labels: torch.Tensor):
        """
        Aplica mixup a un batch de espectrogramas
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
        """Aplica mixup mezclando samples aleatorios del batch"""
        batch_size = spectrograms.size(0)
        
        # Generar lambda de distribución Beta
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        # Permutar índices para mezclar
        index = torch.randperm(batch_size)
        
        # Mezclar espectrogramas
        mixed_spectrograms = lam * spectrograms + (1 - lam) * spectrograms[index]
        
        # Mezclar labels
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        
        return mixed_spectrograms, mixed_labels
