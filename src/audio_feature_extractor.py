import logging
from torch import nn
import torchaudio
import torch
import numpy as np

#region Logging


# Set logging config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter(
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%y-%m-%d %H:%M:%S',
    )

# Set a logging stream handler
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setLevel(logging.INFO)
logger_stream_handler.setFormatter(logger_formatter)

# Add handlers
logger.addHandler(logger_stream_handler)
#endregion




class SpectrogramExtractor(nn.Module):
    """
    This class is a feature extractor that uses the MelSpectrogram transformation from torchaudio.

    The output of the feature extractor is a tensor with the shape [batch_size, time, mel_bands]. 
    """

    def __init__(self, input_parameters) -> None:
        super().__init__()

        self.init_feature_extractor(input_parameters)

    def init_feature_extractor(self, params):
        # Implement your feature extractor initialization code here

        # TODO: Ask Javier about these magic things.
        self.feature_extractor = torchaudio.transforms.MelSpectrogram(
            n_fft = 2048,
            win_length = int(params.sample_rate * 0.025),
            hop_length = int(params.sample_rate * 0.01),
            n_mels = params.feature_extractor_output_vectors_dimension,
            mel_scale = "slaney",
            window_fn = torch.hamming_window,
            f_max = params.sample_rate // 2,
            center = False,
            normalized = False,
            norm = "slaney",
        )
    
    def extract_features(self, audio_signal):
        features = self.feature_extractor(audio_signal)

        # HACK it seems that the feature extractor output spectrogram has mel bands as rows and time as columns
        features = features.transpose(1, 2)

        return features
    
    def __call__(self, waveforms):
        logger.debug(f"waveforms shape: {waveforms.shape}")
        logger.debug(f"wavefor.size(): {waveforms.size()}")
        features = self.extract_features(waveforms)
        logger.debug(f"features.size(): {features.size()}")
        return features
    

class WavLMExtractor(nn.Module):
    """
    The WavLMExtractor class is a feature extractor that uses the WavLM model from torchaudio.

    The output of the feature extractor is a tensor with the shape [batch_size, time, features].

    The features dimension depends on the WavLM model used. For example oin the WAVLM_BASE model, the features have 768 dimensions,
    while in the WAVLM_LARGE model, the features have 1024 dimensions.
    """
    def __init__(self, input_parameters) -> None:
        super().__init__()
        self.wavlm_flavor = input_parameters.wavlm_flavor
        self.select_extractor()
        self.init_layers_weights()

    def select_extractor(self):
        
        if self.wavlm_flavor == "WAVLM_BASE":
            bundle = torchaudio.pipelines.WAVLM_BASE
            self.num_layers = 12 # Layers of the Transformer of the WavLM model
            # every layer has features with 768 dimension
        elif self.wavlm_flavor == "WAVLM_BASE_PLUS":
            bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
            self.num_layers = 12 # Layers of the Transformer of the WavLM model
            # every layer has features with 768 dimension
        elif self.wavlm_flavor == "WAVLM_LARGE":
            bundle = torchaudio.pipelines.WAVLM_LARGE
            self.num_layers = 24 # Layers of the Transformer of the WavLM model
            # every layer has features with 1024 dimension
        elif self.wavlm_flavor == "WAV2VEC2_LARGE_LV60K":
            bundle = torchaudio.pipelines.WAV2VEC2_LARGE_LV60K
            self.num_layers = 24 # Layers of the Transformer of the WavLM model
            # every layer has features with 1024 dimension
        elif self.wavlm_flavor == "WAV2VEC2_XLSR_300M":
            bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
            self.num_layers = 24 # Layers of the Transformer of the WavLM model
            # every layer has features with 1024 dimension
        elif self.wavlm_flavor == "WAV2VEC2_XLSR_1B":
            bundle = torchaudio.pipelines.WAV2VEC2_XLSR_1B
            self.num_layers = 48 # Layers of the Transformer of the WavLM model
            # every layer has features with 1280 dimension
        elif self.wavlm_flavor == "HUBERT_LARGE":
            bundle = torchaudio.pipelines.HUBERT_LARGE
            self.num_layers = 24 # Layers of the Transformer of the WavLM model
            # every layer has features with 1024 dimension
        else:
            raise Exception('No wavlm_flavor choice found.') 

        self.feature_extractor = bundle.get_model()

    
    def init_layers_weights(self):
        """
        Initializes the weights of the layers of the WavLM model.
        """
        self.layer_weights = nn.Parameter(nn.functional.softmax((torch.ones(self.num_layers) / self.num_layers), dim=-1))

    def extract_features(self, waveform):
        """
        We make use of the WavLM model to extract features from the waveform.

        1. We extract a list of features.
        2. We concatenate the features along a new dimension.
        3 We sum the features with the weights of the layers. (weighted average) 
        """

        # we extract a list of features from the WavLM model
        features, _ = self.feature_extractor.extract_features(waveform)
        
        # Concatenates a sequence of tensors along a new dimension.
        hidden_states = torch.stack(features, dim=1)

        # Weighted average of the hidden states
        averaged_hidden_states = (hidden_states * self.layer_weights.view(1, -1, 1, 1)).sum(dim=1)

        return averaged_hidden_states
    
    def __call__(self, waveforms):
        logger.debug(f"waveform.size(): {waveforms.size()}")
        features = self.extract_features(waveforms)
        logger.debug(f"features.size(): {features.size()}")
        return features