import argparse
from settings import TRAIN_DEFAULT_SETTINGS

class ArgsParser:
    
    def __init__(self) -> None:
        self.initialize_parser()

    def initialize_parser(self):

        self.parser = argparse.ArgumentParser(
            description = 'Train a Text and Speech Emotion Recognition model.',
            )
        
    def add_parser_args(self):

        #region DataPaths
        self.parser.add_argument(
            '--train_data_dir',
            type=str,
            default=TRAIN_DEFAULT_SETTINGS['audios_path'],
            help='Path to the directory containing the training data.'
        )
        self.parser.add_argument(
            '--train_labels_path', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['train_labels_path'],
            help = 'Path of the file containing the training examples paths and labels.',
        )
        self.parser.add_argument(
            '--validation_data_dir',
            type=str,
            default=TRAIN_DEFAULT_SETTINGS['audios_path'],
            help='Path to the directory containing the test data.'
        )
        self.parser.add_argument(
            '--validation_labels_path', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['validation_labels_path'],
            help = 'Path of the file containing the validation examples paths and labels.',
        )
        #endregion 
    
        #region TrainingParameters
        self.parser.add_argument(
            '--max_epochs',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['max_epochs'],
            help = 'Max number of epochs to train.',
            )

        self.parser.add_argument(
            '--training_batch_size', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['training_batch_size'],
            help = "Size of training batches.",
            )

        self.parser.add_argument(
            '--evaluation_batch_size', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['evaluation_batch_size'],
            help = "Size of evaluation batches.",
            )

        self.parser.add_argument(
            '--eval_and_save_best_model_every', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['eval_and_save_best_model_every'],
            help = "The model is evaluated on train and validation sets and saved every eval_and_save_best_model_every steps. \
                Set to 0 if you don't want to execute this utility.",
            )
        
        self.parser.add_argument(
            '--print_training_info_every', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['print_training_info_every'],
            help = "Training info is printed every print_training_info_every steps. \
                Set to 0 if you don't want to execute this utility.",
            )

        self.parser.add_argument(
            '--early_stopping', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['early_stopping'],
            help = "Training is stopped if there are early_stopping consectuive validations without improvement. \
                Set to 0 if you don't want to execute this utility.",
            )

        self.parser.add_argument(
            '--load_checkpoint',
            action = argparse.BooleanOptionalAction,
            default = TRAIN_DEFAULT_SETTINGS['load_checkpoint'],
            help = 'Set to True if you want to load a previous checkpoint and continue training from that point. \
                Loaded parameters will overwrite all inputted parameters.',
            )    
        #endregion    
        
        #region OptimizationArguments
        self.parser.add_argument(
            '--optimizer', 
            type = str, 
            choices = ['adam', 'sgd', 'rmsprop', 'adamw'], 
            default = TRAIN_DEFAULT_SETTINGS['optimizer'],
            )

        self.parser.add_argument(
            '--learning_rate', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['learning_rate'],
            )
        
        self.parser.add_argument(
            '--learning_rate_multiplier', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['learning_rate_multiplier'],
            )
        self.parser.add_argument(
            '--learning_rate_scheduler',
            type = str,
            choices= ['NoneScheduler', 'LinearLR'],
            default = TRAIN_DEFAULT_SETTINGS['learning_rate_scheduler'],
        )
        self.parser.add_argument(
            '--weight_decay', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['weight_decay'],
            )
        
        self.parser.add_argument(
            '--update_optimizer_every', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['update_optimizer_every'],
            help = "Some optimizer parameters will be updated every update_optimizer_every consecutive validations without improvement. \
                Set to 0 if you don't want to execute this utility.",
            )

        self.parser.add_argument(
            '--loss', 
            type = str, 
            choices = ['CrossEntropy', 'FocalLoss'], 
            default = TRAIN_DEFAULT_SETTINGS['loss'],
            )
        
        self.parser.add_argument(
            "--weighted_loss", 
            action = argparse.BooleanOptionalAction,
            default = TRAIN_DEFAULT_SETTINGS['weighted_loss'],
            help = "Set the weight parameter of the loss to a tensor representing the inverse frequency of each class.",
            )
        #endregiCram&98!on

        #region DataAugmentationParams
        self.parser.add_argument(
            '--training_augmentation_prob', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['training_augmentation_prob'],
            help = 'Probability of applying data augmentation to each file. Set to 0 if not augmentation is desired.'
            )

        self.parser.add_argument(
            '--evaluation_augmentation_prob', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['evaluation_augmentation_prob'],
            help = 'Probability of applying data augmentation to each file. Set to 0 if not augmentation is desired.'
            )

        self.parser.add_argument(
            '--augmentation_window_size_secs', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['augmentation_window_size_secs'],
            help = 'Cut the audio with augmentation_window_size_secs length at a random starting point. \
                If 0, the full audio is loaded.'
            )

        self.parser.add_argument(
            '--augmentation_effects', 
            type = str, 
            nargs = '+',
            choices = ["apply_speed_perturbation", "apply_reverb", "add_background_noise"],
            help = 'Effects to augment the data. One or many can be choosen.'
            )       

        self.parser.add_argument(
            '--augmentation_noises_labels_path', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['augmentation_noises_labels_path'],
            help = 'Path of the file containing the background noises audio paths and labels.'
            )
        
        self.parser.add_argument(
            '--augmentation_noises_directory', 
            type = str,
            help = 'Optional additional directory to prepend to the augmentation_labels_path paths.',
            )

        self.parser.add_argument(
            '--augmentation_rirs_labels_path', 
            type = str,
            default = TRAIN_DEFAULT_SETTINGS['augmentation_rirs_labels_path'],
            help = 'Path of the file containing the RIRs audio paths.'
            )
        
        self.parser.add_argument(
            '--augmentation_rirs_directory', 
            type = str, 
            help = 'Optional additional directory to prepend to the rirs_labels_path paths.',
            )         
        #endregion

        #region DataParameters
        self.parser.add_argument(
            '--sample_rate', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['sample_rate'],
            help = "Sample rate that you want to use (every audio loaded is resampled to this frequency)."
            )
        
        self.parser.add_argument(
            '--training_random_crop_secs', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['training_random_crop_secs'], 
            help = 'Cut the training input audio with random_crop_secs length at a random starting point. \
                If 0, the full audio is loaded.'
            )

        self.parser.add_argument(
            '--evaluation_random_crop_secs', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['evaluation_random_crop_secs'], 
            help = 'Cut the evaluation input audio with random_crop_secs length at a random starting point. \
                If 0, the full audio is loaded.'
            )

        self.parser.add_argument(
            '--num_workers', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['num_workers'],
            help = 'num_workers to be used by the data loader.'
            )
        
        self.parser.add_argument(
            '--padding_type', 
            type = str, 
            choices = ["zero_pad", "repetition_pad"],
            help = 'Type of padding to apply to the audios. \
                zero_pad does zero left padding, repetition_pad repeats the audio.'
            )        
        #endregion

        #region output
        self.parser.add_argument(
            '--model_output_folder', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['model_output_folder'], 
            help = 'Directory where model outputs and configs are saved.',
            )


        self.parser.add_argument(
            '--log_file_folder',
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['log_file_folder'],
            help = 'Name of folder that will contain the log file.',
            )

        self.parser.add_argument(
            '--json_output_folder',
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['json_output_folder'],
            help = 'Name of folder that will contain the json file.',
            )        
        #endregion

        #region NetworkParameters
        self.parser.add_argument(
            '--feature_extractor', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['feature_extractor'],
            choices = ['SpectrogramExtractor', 'WavLMExtractor'], 
            help = 'Type of Feature Extractor used. It should take an audio waveform and output a sequence of vector (features).' 
            )

        self.parser.add_argument(
            '--wavlm_flavor', 
            type = str, 
            choices = ['WAVLM_BASE', 'WAVLM_BASE_PLUS', 'WAVLM_LARGE', 'WAV2VEC2_LARGE_LV60K', 'WAV2VEC2_XLSR_300M', 'WAV2VEC2_XLSR_1B', 'HUBERT_LARGE'], 
            help = 'wavLM model flavor, considered only if WavLMExtractor is used.' 
            )
        
        self.parser.add_argument(
            '--feature_extractor_output_vectors_dimension', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['feature_extractor_output_vectors_dimension'], 
            help = 'Dimension of each vector that will be the output of the feature extractor (usually number of mels in mel-spectrogram).'
            )

        self.parser.add_argument(
            '--text_feature_extractor', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['text_feature_extractor'], 
            choices = ['TextBERTExtractor', 'NoneTextExtractor'], 
            help = 'Type of Text Feature Extractor used. It should take an audio waveform and output a sequence of vector (features).' 
            )

        self.parser.add_argument(
            '--bert_flavor', 
            type = str, 
            choices = ['BERT_BASE_UNCASED', 'BERT_BASE_CASED', 'BERT_LARGE_UNCASED', 'BERT_LARGE_CASED', 'ROBERTA_LARGE', 'BETO_BASE', 'BETO_EMO', 'ROBERTA_LARGE_SPANISH'], 
            help = 'BERT model flavor, considered only if TextBERTExtractor is used.' 
            )
        
        self.parser.add_argument(
            '--front_end', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['front_end'],
            choices = ['VGG', 'Resnet34', 'Resnet101', 'NoneFrontEnd'], 
            help = 'Type of Front-end used. \
                VGG for a N-block VGG architecture.'
            )
        
        self.parser.add_argument(
            '--vgg_n_blocks', 
            type = int, 
            help = 'Number of blocks the VGG front-end block will have.\
                Each block consists in two convolutional layers followed by a max pooling layer.',
            )

        self.parser.add_argument(
            '--vgg_channels', 
            nargs = '+',
            type = int,
            help = 'Number of channels each VGG convolutional block will have. \
                The number of channels must be passed in order and consisently with vgg_n_blocks.',
            )
        
        self.parser.add_argument(
            '--adapter', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['adapter'],
            choices = ['NoneAdapter', 'LinearAdapter', 'NonLinearAdapter'], 
            help = 'Type of adapter used.'
            )
        
        self.parser.add_argument(
            '--adapter_output_vectors_dimension', 
            type = int, 
            help = 'Dimension of each vector that will be the output of the adapter layer.',
            )
        
        self.parser.add_argument(
            '--seq_to_seq_method', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['seq_to_seq_method'], 
            choices = ['NoneSeqToSeq', 'SelfAttention', 'MultiHeadAttention', 'TransformerStacked', 'ReducedMultiHeadAttention'], 
            help = 'Sequence to sequence component after the linear projection layer of the model.',
            )

        self.parser.add_argument(
            '--seq_to_seq_heads_number', 
            type = int, 
            help = 'Number of heads for the seq_to_seq layer of the pooling component \
                (only for MHA based seq_to_seq options).',
            )

        self.parser.add_argument(
            '--transformer_n_blocks', 
            type = int, 
            help = 'Number of transformer blocks to stack in the seq_to_seq component of the pooling. \
                (Only for seq_to_seq_method = TransformerStacked).',
            )

        self.parser.add_argument(
            '--transformer_expansion_coef', 
            type = int, 
            help = "Number you want to multiply by the size of the hidden layer of the transformer block's feed forward net. \
                (Only for seq_to_seq_method = TransformerBlock)."
            )
        
        self.parser.add_argument(
            '--transformer_drop_out', 
            type = float, 
            help = 'Dropout probability to use in the feed forward component of the transformer block.\
                (Only for seq_to_seq_method = TransformerBlock).'
            )
        
        self.parser.add_argument(
            '--seq_to_one_method', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['seq_to_one_method'], 
            choices = ['StatisticalPooling', 'AttentionPooling'], 
            help = 'Type of pooling method applied to the output sequence to sequence component of the model.',
            )

        self.parser.add_argument(
            '--seq_to_seq_input_dropout', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['seq_to_seq_input_dropout'],
            help = 'Dropout probability to use in the seq to seq component input.'
            )

        self.parser.add_argument(
            '--seq_to_one_input_dropout', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['seq_to_one_input_dropout'],
            help = 'Dropout probability to use in the seq to one component input.'
            )
        
        self.parser.add_argument(
            '--classifier_layer_drop_out', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['classifier_layer_drop_out'],
            help = 'Dropout probability to use in the classfifer component.'
            )

        self.parser.add_argument(
            '--classifier_hidden_layers', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['classifier_hidden_layers'],
            help = 'Number of hidden layers in the classifier layer.',
            )

        self.parser.add_argument(
            '--classifier_hidden_layers_width', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['classifier_hidden_layers_width'],
            help = 'Width of every hidden layer in the classifier layer.',
            )
        
        self.parser.add_argument(
            '--number_classes', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['number_classes'],
            help = "Number of classes to classify.",
        )
        #endregion

        #region CheckpointPaths
        self.parser.add_argument(
            '--checkpoint_file_folder',
            type = str, 
            help = 'Name of folder that contain the model checkpoint file. Mandatory if load_checkpoint is True.',
            )
        
        self.parser.add_argument(
            '--checkpoint_file_name',
            type = str, 
            help = 'Name of the model checkpoint file. Mandatory if load_checkpoint is True.',
            )
        #endregion

        #region WeightsAndBiases
        self.parser.add_argument(
            '--use_weights_and_biases',
            action = argparse.BooleanOptionalAction,
            default = TRAIN_DEFAULT_SETTINGS['use_weights_and_biases'],
            help = 'Set to True if you want to use Weights and Biases.',
            )
        #endregion

    def main(self):
        self.add_parser_args()
        self.arguments = self.parser.parse_args()

    def __call__(self):
        self.main()


