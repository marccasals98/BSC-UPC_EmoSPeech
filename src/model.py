import logging
from torch import nn 
import torch
from text_feature_extractor import TextBERTExtractor
from audio_feature_extractor import SpectrogramExtractor, WavLMExtractor
from audio_front_end import VGG, Resnet34, Resnet101, NoneFrontEnd
from adapter import NoneAdapter, LinearAdapter, NonLinearAdapter
import numpy as np
from poolings import NoneSeqToSeq, SelfAttention, MultiHeadAttention, AttentionPooling, StatisticalPooling
from classifier_layer import ClassifierLayer

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

class Classifier(nn.Module):
    """
    Classifier model
    ----------------
    """
    def __init__(self, parameters, device) -> None:
        super().__init__()
        self.device = device
        self.init_text_feature_extractor(parameters)
        self.init_audio_feature_extractor(parameters)
        self.init_audio_front_end(parameters)
        self.init_adapter(parameters)
        self.init_pooling_components(parameters)
        self.init_classifier_layer(parameters)



    def init_text_feature_extractor(self, parameters):
        """
        This method initializes the text feature extractor.
        """

        # BERT Feature Extractor
        if parameters.text_feature_extractor == 'TextBERTExtractor':
            self.text_feature_extractor = TextBERTExtractor(parameters)
        
            # named_parameters() returns an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.
            for name, parameter in self.text_feature_extractor.named_parameters():
                # TODO: Allow to train some parameters.
                # I understand that the parameters of the text feature extractor are not trainable
                logger.info(f"Setting {name} to requires_grad = False")
                parameter.requires_grad = False
            
            # implements Layer Normalization: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
            self.text_feature_extractor_norm_layer = nn.LayerNorm(parameters.feature_extractor_output_vectors_dimension)
        else:
            self.text_feature_extractor = None
            logger.info(f"No Text Feature Extractor Selected :(")

    def init_audio_feature_extractor(self, parameters):
        """
        This method initializes the audio feature extractor.

        There are two options:
        * SpectrogramExtractor
        * WavLMExtractor

        After this, it will be applied the Layer Normalization.

        .. math::
                y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

            The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
            is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
            is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
            the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
            :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
            :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
            The standard-deviation is calculated via the biased estimator, equivalent to
            `torch.var(input, unbiased=False)`.


        """
        if parameters.feature_extractor == 'SpectrogramExtractor':
            self.feature_extractor = SpectrogramExtractor(parameters)
        elif parameters.feature_extractor == 'WavLMExtractor':
            self.feature_extractor = WavLMExtractor(parameters)
        else:
            raise ValueError(f"Audio feature extractor {parameters.feature_extractor} not found")
        
        # Freeze all wavLM parameter except layers weights
        for name, parameter in self.feature_extractor.named_parameters():          
            if name != "layer_weights":
                logger.info(f"Setting {name} to requires_grad = False")
                parameter.requires_grad = False
        
        logger.debug(f"Feature extractor output vectors dimension: {parameters.feature_extractor_output_vectors_dimension}. Check if it suits the LayerNorm dimensions.")
        self.feature_extractor_norm_layer = nn.LayerNorm(parameters.feature_extractor_output_vectors_dimension)

    def init_audio_front_end(self, parameters):
        """
        This method initializes the front end.
        """
        if parameters.front_end == 'VGG':

            # Set the front-end component that will take the spectrogram and generate complex features
            self.front_end = VGG(parameters.vgg_n_blocks, parameters.vgg_channels)
                
            # Calculate the size of the hidden state vectors (output of the front-end)
            self.front_end_output_vectors_dimension = self.front_end.get_output_vectors_dimension(
                parameters.feature_extractor_output_vectors_dimension, 
                )

        elif parameters.front_end == 'Resnet34':

            # Set the front-end component that will take the spectrogram and generate complex features
            self.front_end = Resnet34(
                256, # HACK set as parameter?
                )
                
            # Calculate the size of the hidden state vectors (output of the front-end)
            self.front_end_output_vectors_dimension = self.front_end.get_output_vectors_dimension(
                parameters.feature_extractor_output_vectors_dimension, 
                256, # HACK set as parameter?
                )
        
        elif parameters.front_end == 'Resnet101':

            # Set the front-end component that will take the spectrogram and generate complex features
            self.front_end = Resnet101(256)
                
            # Calculate the size of the hidden state vectors (output of the front-end)
            self.front_end_output_vectors_dimension = self.front_end.get_output_vectors_dimension(
                parameters.feature_extractor_output_vectors_dimension, 
                256,
                )

        elif parameters.front_end == 'NoneFrontEnd':

            # Set the front-end component that will take the spectrogram and generate complex features
            self.front_end = NoneFrontEnd()
                
            # Calculate the size of the hidden state vectors (output of the front-end)
            self.front_end_output_vectors_dimension = self.front_end.get_output_vectors_dimension(
                parameters.feature_extractor_output_vectors_dimension, 
                )
        
        else:
            raise Exception('No Front End choice found.')       

    def init_adapter(self, parameters):
        """
        This method initializes the adapter.
        """
        if parameters.adapter == 'NoneAdapter':
            self.adapter_layer = NoneAdapter()
            parameters.adapter_output_vectors_dimension = self.front_end_output_vectors_dimension
        elif parameters.adapter == 'LinearAdapter':
            if parameters.adapter_output_vectors_dimension is None:
                raise ValueError("adapter_output_vectors_dimension is not defined.")
            self.adapter_layer = LinearAdapter(self.front_end_output_vectors_dimension, parameters.adapter_output_vectors_dimension)
        elif parameters.adapter == 'NonLinearAdapter':
            if parameters.adapter_output_vectors_dimension is None:
                raise ValueError("adapter_output_vectors_dimension is not defined.")            
            self.adapter_layer = NonLinearAdapter(self.front_end_output_vectors_dimension, parameters.adapter_output_vectors_dimension)
        else:
            raise Exception('No Adapter choice found.')


    def init_seq_to_seq_layer(self, parameters):
        
            self.seq_to_seq_method = parameters.seq_to_seq_method
            self.seq_to_seq_input_vectors_dimension = parameters.adapter_output_vectors_dimension

            self.seq_to_seq_input_dropout = nn.Dropout(parameters.seq_to_seq_input_dropout)

            # HACK ReducedMultiHeadAttention seq to seq input and output dimensions don't match
            if self.seq_to_seq_method == 'ReducedMultiHeadAttention':
                if parameters.seq_to_seq_heads_number is None:
                    raise ValueError("seq_to_seq_heads_number is not defined. It is required for ReducedMultiHeadAttention.")
                self.seq_to_seq_output_vectors_dimension = self.seq_to_seq_input_vectors_dimension // parameters.seq_to_seq_heads_number
            else:
                self.seq_to_seq_output_vectors_dimension = self.seq_to_seq_input_vectors_dimension

            if self.seq_to_seq_method == 'NoneSeqToSeq':
                self.seq_to_seq_layer = NoneSeqToSeq()
            
            elif self.seq_to_seq_method == 'SelfAttention':
                self.seq_to_seq_layer = SelfAttention()

            elif self.seq_to_seq_method == 'MultiHeadAttention':
                self.seq_to_seq_layer = MultiHeadAttention(
                    emb_in = self.seq_to_seq_input_vectors_dimension,
                    heads = parameters.seq_to_seq_heads_number,
                )


            else:
                raise Exception('No Seq to Seq choice found.')  
    

    def init_seq_to_one_layer(self, parameters):

            self.seq_to_one_method = parameters.seq_to_one_method
            self.seq_to_one_input_vectors_dimension = self.seq_to_seq_output_vectors_dimension
            self.seq_to_one_output_vectors_dimension = self.seq_to_one_input_vectors_dimension

            self.seq_to_one_input_dropout = nn.Dropout(parameters.seq_to_one_input_dropout)

            if self.seq_to_one_method == 'StatisticalPooling':
                self.seq_to_one_layer = StatisticalPooling(
                        emb_in = self.seq_to_one_input_vectors_dimension,
                    )

            elif self.seq_to_one_method == 'AttentionPooling':
                self.seq_to_one_layer = AttentionPooling(
                    emb_in = self.seq_to_one_input_vectors_dimension,
                )
            
            else:
                raise Exception('No Seq to One choice found.') 
        
    def init_pooling_components(self, parameters):
        """
        Set the pooling component that will take the front-end features and summarize them in a context vector
        This component applies first a sequence to sequence layer and then a sequence to one layer.
        """
        self.init_seq_to_seq_layer(parameters)
        self.init_seq_to_one_layer(parameters)

    def init_classifier_layer(self, parameters):

        if self.text_feature_extractor:

            # Option 1: concatenate text and acoustic pooled vectors
            #self.classifier_layer_input_vectors_dimension = 2 * self.seq_to_one_output_vectors_dimension

            # Option 2: use seq_to_one Attention Pooling
            # we are assuming text and acoustic pooled vector have same dimension
            #self.classifier_layer_input_vectors_dimension = self.seq_to_one_output_vectors_dimension 
            #self.text_acoustic_pooling_layer = AttentionPooling(emb_in = self.seq_to_one_input_vectors_dimension)

            # Option 3: all acoustic and text features goes into the same seq_to_one component
            #self.classifier_layer_input_vectors_dimension = self.seq_to_one_output_vectors_dimension

            # Option 4: all acoustic and text features goes into the same seq_to_seq component
            self.classifier_layer_input_vectors_dimension = self.seq_to_one_output_vectors_dimension
        else:
            self.classifier_layer_input_vectors_dimension = self.seq_to_one_output_vectors_dimension
        
        logger.debug(f"self.classifier_layer_input_vectors_dimension: {self.classifier_layer_input_vectors_dimension}")
        
        self.classifier_layer = ClassifierLayer(parameters, self.classifier_layer_input_vectors_dimension)


    def forward(self, input_tensor: torch.Tensor, transcription_tokens_padded = None, transcription_tokens_mask=None) -> torch.Tensor:
        """
        Forward pass
        ------------
        """

        logger.debug(f"Input tensor size: {input_tensor.size()}")

        #region TextComponents
        if self.text_feature_extractor:
            logger.debug(f"Transcription tokens padded size: {transcription_tokens_padded.size()}")
            # executing the method __call__ of the TextBERTExtractor
            text_feature_extractor_output = self.text_feature_extractor(transcription_tokens_padded, transcription_tokens_mask)
            try:
                text_feature_extractor_output = self.text_feature_extractor_norm_layer(text_feature_extractor_output)
            except:
                raise ValueError(f"Error normalizing text_feature_extractor_output with dimension {text_feature_extractor_output.size()} \n The dimension must coincide with parameters.feature_extractor_output_vectors_dimension")
            logger.debug(f"Text feature extractor output size: {text_feature_extractor_output.size()}")
        #endregion

        #region AcousticComponents
        # Feature Extraction:
        feature_extractor_output = self.feature_extractor(input_tensor)
        # logger.debug(f"Output of the feature extractor.  Size: {feature_extractor_output.size()}")
        feature_extractor_output = self.feature_extractor_norm_layer(feature_extractor_output) 
        logger.debug(f"Feature extractor output size: {feature_extractor_output.size()}")

        # Front End Network
        encoder_output = self.front_end(feature_extractor_output)
        logger.debug(f"Encoder output size: {encoder_output.size()}")

        # Adapter Output
        adapter_output = self.adapter_layer(encoder_output)
        logger.debug(f"Adapter output size: {adapter_output.size()}")

        # adapter
        
        #endregion

        if self.text_feature_extractor:
            # HACK
            # Option 1: concatenate text and acoustic pooled vectors
            #seq_to_seq_output = self.seq_to_seq_layer(adapter_output)

            # HACK
            # Option 2: use seq_to_one Attention Pooling
            #seq_to_seq_output = self.seq_to_seq_layer(adapter_output)

            # HACK
            # Option 3: all acoustic and text features goes into the same seq_to_one component
            #seq_to_seq_output = self.seq_to_seq_layer(adapter_output)
            
            # HACK
            # Option 4: all acoustic and text features goes into the same seq_to_seq component
            seq_to_seq_output = self.seq_to_seq_layer(torch.cat((adapter_output, text_feature_extractor_output), dim = 1))
            logger.debug(f"seq_to_seq_output.size(): {seq_to_seq_output.size()}")
        else:
            seq_to_seq_output = self.seq_to_seq_layer(adapter_output)
            logger.debug(f"seq_to_seq_output.size(): {seq_to_seq_output.size()}")

        seq_to_seq_output = self.seq_to_one_input_dropout(seq_to_seq_output)
        logger.debug(f"seq_to_seq_output.size(): {seq_to_seq_output.size()}")
        if self.text_feature_extractor:
            # HACK
            # Option 1: concatenate text and acoustic pooled vectors
            #seq_to_one_output = self.seq_to_one_layer(seq_to_seq_output)
            # HACK
            # Option 2: use seq_to_one Attention Pooling
            #seq_to_one_output = self.seq_to_one_layer(seq_to_seq_output)
            # HACK
            # Option 3: all acoustic and text features goes into the same seq_to_one component
            #seq_to_one_output = self.seq_to_one_layer(torch.cat((seq_to_seq_output, text_feature_extractor_output), dim = 1))
            # HACK
            # Option 4: all acoustic and text features goes into the same seq_to_seq component
            seq_to_one_output = self.seq_to_one_layer(seq_to_seq_output)
        else:
            seq_to_one_output = self.seq_to_one_layer(seq_to_seq_output)
        logger.debug(f"seq_to_one_output.size(): {seq_to_one_output.size()}")

        #region Classifier
        # classifier_output are logits, SoftMax will be applied in the loss function
        if self.text_feature_extractor:
            # HACK
            # Option 1: concatenate text and acoustic pooled vectors
            #classifier_input = torch.cat([seq_to_one_output, text_feature_extractor_output], 1)
            # HACK
            # Option 2: use seq_to_one Attention Pooling
            #classifier_input = self.text_acoustic_pooling_layer(torch.stack([seq_to_one_output, text_feature_extractor_output], dim = 1))
            # HACK
            # Option 3: all acoustic and text features goes into the same seq_to_one component
            #classifier_input = seq_to_one_output
            # HACK
            # Option 4: all acoustic and text features goes into the same seq_to_seq component
            classifier_input = seq_to_one_output

        else:
            classifier_input = seq_to_one_output
        logger.debug(f"classifier_input.size(): {classifier_input.size()}")

        classifier_output = self.classifier_layer(classifier_input)
        logger.debug(f"classifier_output.size(): {classifier_output.size()}")
    
        #endregion
        return classifier_output        

    def predict(self, input_tensor, transcription_tokens_padded = None, transcription_tokens_mask = None, thresholds_per_class = None):

        # HACK awfull hack, we are going to assume that we are going to predict over single tensors (no batches)
        
        predicted_logits = self.forward(input_tensor, transcription_tokens_padded, transcription_tokens_mask)
        predicted_probas = torch.nn.functional.log_softmax(predicted_logits, dim = 1)
        predicted_probas = predicted_probas.squeeze().to("cpu").numpy()
        logger.debug(f"predicted_probas: {predicted_probas}")

        if thresholds_per_class is not None:
            logger.debug("Entered threshold_per_class")
            max_proba_class = np.argmax(predicted_probas)
            logger.debug(f"max_proba_class: {max_proba_class}")
            threshold_check = predicted_probas[max_proba_class] >= thresholds_per_class[max_proba_class]
            logger.debug(f"threshold_check: {threshold_check}, {predicted_probas[max_proba_class]}, {thresholds_per_class[max_proba_class]}")

            if threshold_check == True:
                logger.debug("Entered threshold_check")
                predicted_class = max_proba_class
            else:
                logger.debug("Entered filtered_probas")
                filtered_probas = predicted_probas.copy()
                filtered_probas[max_proba_class] = -np.inf
                logger.debug(f"filtered_probas: {filtered_probas}")
                predicted_class = np.argmax(filtered_probas)
        else:
            logger.debug("Entered normal prediction")
            predicted_class = np.argmax(predicted_probas)

        return torch.tensor([predicted_class]).int()

