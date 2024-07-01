PRETRAINED_VARIABLES = {
    'load_checkpoint': 'False',
    'epoch': 0,
}


LABELS_REDUCED_TO_IDS = {
    "neutral": 0,
    "disgust": 1,
    "anger": 2,
    "sadness": 3,
    "joy": 4,
    "fear": 5,
    }



TRAIN_DEFAULT_SETTINGS = {
    'adapter': 'NoneAdapter',
    'audios_path' : '/home/usuaris/veussd/marc.casals/EmoSPeech2024/data/train_segments/',
    'augmentation_noises_labels_path': "./labels/data_aug/data_augmentation_noises_labels.tsv",
    'augmentation_rirs_labels_path': "./labels/data_aug/data_augmentation_rirs_labels.tsv",
    'augmentation_window_size_secs' : 2.0,
    'classifier_hidden_layers_width': 512,
    'classifier_hidden_layers': 2,
    'classifier_layer_drop_out' : 0,
    'early_stopping' : 25,
    'eval_and_save_best_model_every' : 100,
    'evaluation_augmentation_prob' : 0,
    'evaluation_batch_size' : 1,
    'evaluation_random_crop_secs' : 2.0,
    'feature_extractor_output_vectors_dimension' : 80,
    'feature_extractor': 'SpectrogramExtractor',
    'front_end' : 'NoneFrontEnd',
    'json_output_folder' : 'json_output_folder',
    'learning_rate_multiplier' : 0.5,
    'learning_rate_scheduler' : 'NoneScheduler',
    'learning_rate' : 0.0001,
    'load_checkpoint' : False,
    'log_file_folder' : './slurm_logs/train/',
    'loss' : 'CrossEntropy',
    'max_epochs' : 25,
    'model_output_folder' : './models/',
    'num_workers' : 0,
    'number_classes' : 6,
    'optimizer' : 'adam',
    'print_training_info_every' : 100,
    'sample_rate': 44100,
    'seq_to_one_input_dropout': 0,
    'seq_to_one_method' : 'StatisticalPooling',
    'seq_to_seq_input_dropout' : 0,
    'seq_to_seq_method' : 'NoneSeqToSeq',
    'text_feature_extractor' : 'NoneTextExtractor',
    'train_labels_path' : 'labels/train_split.csv',
    'training_augmentation_prob' : 0.0,
    'training_batch_size' : 16,
    'training_random_crop_secs' : 9.0,
    'update_optimizer_every' : 10,
    'use_weights_and_biases' : False,
    'validation_labels_path' : 'labels/dev_split.csv',
    'weight_decay' : 0.001,
    'weighted_loss' : True,
}

