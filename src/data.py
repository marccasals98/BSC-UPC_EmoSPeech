import logging 
from torch.utils import data
import copy
import numpy as np
import torchaudio
import random
from random import randint
import pandas as pd
import torch
from augmentation import DataAugmentator


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

class TrainDataset(data.Dataset):
    def __init__(self, utterances_paths, input_parameters, random_crop_secs, augmentation_prob = 0, sample_rate = 16000, waveforms_mean = None, waveforms_std = None):
        
        self.utterances_paths = utterances_paths
        # I suspect when instantiating two datasets the parameters are overrided
        # TODO maybe avoid defining self.parameters to reduce memory usage
        self.parameters = copy.deepcopy(input_parameters) 
        self.augmentation_prob = augmentation_prob
        self.sample_rate = sample_rate
        self.waveforms_mean = waveforms_mean
        self.waveforms_std = waveforms_std
        self.random_crop_secs = random_crop_secs
        self.random_crop_samples = int(self.random_crop_secs * self.sample_rate)
        self.num_samples = len(utterances_paths)
        if self.augmentation_prob > 0: self.init_data_augmentator()
        if self.parameters.text_feature_extractor != 'NoneTextExtractor': self.init_tokenizer()

    def __len__(self):
        return self.num_samples
    
    def get_classes_weights(self):

        dataset_labels = [path.strip().split('\t')[1] for path in self.utterances_paths]

        weights_series = pd.Series(dataset_labels).value_counts(normalize = True, dropna = False)
        weights_df = pd.DataFrame(weights_series).reset_index()
        weights_df.columns = ["class_id", "weight"]
        weights_df["weight"] = 1 / weights_df["weight"]
        weights_df = weights_df.sort_values("class_id", ascending=True)
        
        weights = weights_df["weight"].to_list()

        for class_id in range(len(weights)):
            logger.info(f"Class_id {class_id} weight: {weights[class_id]}")
        
        return weights   
    
    def pad_waveform(self, waveform, padding_type, random_crop_samples):

        if padding_type == "zero_pad":
            pad_left = max(0, self.random_crop_samples - waveform.shape[-1])
            padded_waveform = torch.nn.functional.pad(waveform, (pad_left, 0), mode = "constant")
        elif padding_type == "repetition_pad":
            necessary_repetitions = int(np.ceil(random_crop_samples / waveform.size(-1)))
            padded_waveform = waveform.repeat(necessary_repetitions)
        else:
            raise Exception('No padding choice found.') 
        
        return padded_waveform
    
    def sample_audio_window(self, waveform, random_crop_samples):

        waveform_total_samples = waveform.size()[-1]
        
        # TODO maybe we can add an assert to check that random_crop_samples <= waveform_total_samples (will it slow down the process?)
        random_start_index = randint(0, waveform_total_samples - random_crop_samples)
        end_index = random_start_index + random_crop_samples
        
        cropped_waveform =  waveform[random_start_index : end_index]

        return cropped_waveform
    
    def normalize(self, waveform):

        if self.waveforms_mean is not None and self.waveforms_std is not None:
            normalized_waveform = (waveform - self.waveforms_mean) / (self.waveforms_std + 0.000001)
        else:
            normalized_waveform = waveform

        return normalized_waveform
    
    def init_data_augmentator(self):

        self.data_augmentator = DataAugmentator(
            self.parameters.augmentation_noises_directory,
            self.parameters.augmentation_noises_labels_path,
            self.parameters.augmentation_rirs_directory,
            self.parameters.augmentation_rirs_labels_path,
            self.parameters.augmentation_window_size_secs,
            self.parameters.augmentation_effects,
        )
    
    def process_waveform(self, waveform: torch.Tensor, original_sample_rate: int):

        # Check if sampling rate is the same as the one we want
        if original_sample_rate != self.sample_rate:
            # TODO: Check if this works. I think that it doesnt
            waveform = torchaudio.functional.resample(
                waveform = waveform,
                orig_freq = original_sample_rate, 
                new_freq = self.sample_rate, 
                )

        # Apply data augmentation if it falls within the probability
        if random.uniform(0, 0.999) > 1 - self.augmentation_prob:
            waveform = self.data_augmentator(waveform, self.sample_rate)

        # torchaudio.load returns tensor, sample_rate
        # tensor is a Tensor with shape [channels, time]
        # we use squeeze to get ride of channels, that should be mono
        # librosa has an option to force to mono, torchaudio does not
        
        waveform_mono = torch.mean(waveform, dim=0)
        waveform = waveform_mono.squeeze(0)


        # we have a waveform with shape waveform.size(): torch.Size([0, 397385]), so we want to remove the first dimension
        # TODO: Check if this works. I think that it does 
        # waveform = waveform[1]

        if self.random_crop_secs > 0:
            
            # We make padding to allow cropping longer segments
            # (If not, we can only crop at most the duration of the shortest audio)
            if self.random_crop_samples > waveform.size(-1):
                waveform = self.pad_waveform(waveform, self.parameters.padding_type, self.random_crop_samples)
            
            # TODO torchaudio.load has frame_offset and num_frames params. Providing num_frames and frame_offset arguments is more efficient
            waveform = self.sample_audio_window(
                waveform, 
                random_crop_samples = self.random_crop_samples,
                )
        else:
            # HACK don't understand why, I have to do this slicing (which sample_audio_window does) to make dataloader work
            waveform =  waveform[:]

        waveform = self.normalize(waveform)
        
        return waveform
    
    def init_tokenizer(self):
        logger.info(f"The tokenizer is initializing...")

        if self.parameters.bert_flavor == "BERT_BASE_UNCASED":
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
        elif self.parameters.bert_flavor == "BERT_BASE_CASED":
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
        elif self.parameters.bert_flavor == "BERT_LARGE_UNCASED":
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-large-uncased')
            logger.info(f"Tokenizer model max length: {self.tokenizer.model_max_length}")
        elif self.parameters.bert_flavor == "BERT_LARGE_CASED":
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-large-cased')
        elif self.parameters.bert_flavor == "ROBERTA_LARGE":
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'roberta-large')
        elif self.parameters.bert_flavor == "BETO_BASE":
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'dccuchile/bert-base-spanish-wwm-uncased')
        elif self.parameters.bert_flavor == "BETO_EMO":
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'ignacio-ave/beto-sentiment-analysis-spanish')        
        elif self.parameters.bert_flavor == "ROBERTA_LARGE_SPANISH":
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'llange/xlm-roberta-large-spanish')
        else:
            raise Exception('No bert_flavor choice found.')

        logger.info(f"The tokenizer has been initialized!")
        

    def get_transcription_tokens(self, transcription: str) -> torch.Tensor:
        indexed_tokens = self.tokenizer.encode(transcription, add_special_tokens=True)
        #tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = torch.tensor(indexed_tokens)

        return tokens_tensor
        
    def __getitem__(self, index):

        utterance_tuple = self.utterances_paths[index].strip().split('\t')
        
        #logger.debug(f"utterance_tuple: {utterance_tuple}")

        utterance_path = utterance_tuple[0]
        utterance_label = utterance_tuple[1]
        utterance_transcription = utterance_tuple[2]

        # By default, the resulting tensor object has dtype=torch.float32 and its value range is normalized within [-1.0, 1.0]!
        waveform, original_sample_rate = torchaudio.load(utterance_path)       

        # We transcribe before processing the waveform
        if self.parameters.text_feature_extractor != 'NoneTextExtractor':
            # HACK using dataset transcriptions as a quick test
            #transcription = self.get_transcription(waveform)
            # transcription = self.get_transcription(utterance_path)
            transcription = utterance_transcription
            transcription_tokens = self.get_transcription_tokens(transcription)

            if len(transcription_tokens) > self.tokenizer.model_max_length:
                logger.info(f"Transcription tokens length: {len(transcription_tokens)}, the transcription will be cut up")
                transcription_tokens = transcription_tokens[:self.tokenizer.model_max_length]

        waveform = self.process_waveform(waveform, original_sample_rate)
        
        labels = np.array(int(utterance_label))


        if self.parameters.text_feature_extractor != 'NoneTextExtractor':
            return waveform, labels, transcription_tokens
        else:
            return waveform, labels
