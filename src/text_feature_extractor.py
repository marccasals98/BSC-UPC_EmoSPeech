from torch import nn
import torch
import logging

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

class TextBERTExtractor(nn.Module):
    
    def __init__(self, parameters) -> None:
        super().__init__()
        self.bert_flavor = parameters.bert_flavor
        self.select_extractor()

    def select_extractor(self):
        """
        This method selects the BERT model to use. 

        The models are stored in the huggingface hub. 
        """

        if self.bert_flavor == "BERT_BASE_UNCASED":
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
            # model outputs features with 768 dimension
        elif self.bert_flavor == "BERT_BASE_CASED":
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
            # model outputs features with 768 dimension
        elif self.bert_flavor == "BERT_LARGE_UNCASED":
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-large-uncased')
            # model outputs features features with 1024 dimension
        elif self.bert_flavor == "BERT_LARGE_CASED":
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-large-cased')
            # model outputs features features with 1024 dimension
        elif self.bert_flavor == "ROBERTA_LARGE":
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'roberta-large')
            # model outputs features features with 1024 dimension
        elif self.bert_flavor == "BETO_BASE":
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'dccuchile/bert-base-spanish-wwm-uncased')
            # model outputs features features with 768 dimension
        elif self.bert_flavor == "BETO_EMO":
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'ignacio-ave/beto-sentiment-analysis-spanish')
            # model outputs features features with 768 dimension            
        elif self.bert_flavor == "ROBERTA_LARGE_SPANISH":
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'llange/xlm-roberta-large-spanish')
        else:
            raise Exception('No bert_flavor choice found.')
        
    def extract_features(self, trancription_tokens_padded, transcription_tokens_mask):
        """
        This method extracts the features from the BERT model.

        By default, Fede wants to extract the last hidden of the Bert Model, and use it as a feature vector. 
        """

        with torch.no_grad():
            model_output = self.model(trancription_tokens_padded, attention_mask = transcription_tokens_mask)
            
            # we will use the last hidden state of the model as features. 
            features = model_output.last_hidden_state

            logger.debug(f"Features size: {features.size()}")
        return features
    
    def __call__(self, trancription_tokens_padded, transcription_tokens_mask):
        features = self.extract_features(trancription_tokens_padded, transcription_tokens_mask)
        return features