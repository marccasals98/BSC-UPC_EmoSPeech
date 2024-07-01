from args_parser import ArgsParser
import logging
from tqdm import tqdm
from settings import PRETRAINED_VARIABLES, LABELS_REDUCED_TO_IDS
import datetime
import os
from torch import optim
from torch.optim import lr_scheduler
import torch
import random
from torch.utils.data import DataLoader
import wandb
from data import TrainDataset
from utils import generate_model_name, pad_collate, format_training_labels, get_waveforms_stats, get_memory_info
import torch.nn as nn
from loss import FocalLossCriterion
import numpy as np
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import json
import sys

from  model import Classifier


#region logging
# Logging
# -------
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

class Trainer:
    def __init__(self, trainer_parameters) -> None:
        self.start_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H:%M:%S')
        if trainer_parameters.use_weights_and_biases: self.init_wandb()
        self.set_params(trainer_parameters)
        self.set_device()
        self.set_random_seed()
        self.set_log_file_handler()
        self.load_data()
        self.load_network()
        self.load_loss_function()
        self.load_optimizer()
        self.init_scheduler()
        self.initialize_training_variables()

        if self.params.use_weights_and_biases: self.config_wandb()
        self.info_mem(logger_level = "INFO")
    
    
    #region wandb
    def init_wandb(self):
        # Init a wandb projectDEBUG
            
        # TODO fix this, it should be more general to other users
        wandb_run = wandb.init(
            project = "EmoSPeech2024", 
            job_type = "training", 
            entity = "bsc-upc",
            )
        # TODO fix this, it should be a more specific name
        del wandb_run        

    def config_wandb(self):
        # 1 - Save the params
        self.wandb_config = vars(self.params)
        
        # 2 - Save additional params
        self.wandb_config["total_trainable_params"] = self.total_trainable_params
        self.wandb_config["gpus"] = self.gpus_count

        # 3 - Update the wandb config
        wandb.config.update(self.wandb_config)

    #endregion

    #region Initialization
    def set_log_file_handler(self):

        '''Set a logging file handler.'''

        if not os.path.exists(self.params.log_file_folder):
            os.makedirs(self.params.log_file_folder)
        
        if self.params.use_weights_and_biases:
            logger_file_name = f"{self.start_datetime}_{wandb.run.id}_{wandb.run.name}.log"
        else:
            logger_file_name = f"{self.start_datetime}.log"
        logger_file_name = logger_file_name.replace(':', '_').replace(' ', '_').replace('-', '_')

        logger_file_path = os.path.join(self.params.log_file_folder, logger_file_name)
        logger_file_handler = logging.FileHandler(logger_file_path, mode = 'w')
        logger_file_handler.setLevel(logging.INFO) # TODO set the file handler level as a input param
        logger_file_handler.setFormatter(logger_formatter)

        logger.addHandler(logger_file_handler)


    def info_mem(self, step = None, logger_level = "INFO"):

        '''Logs CPU and GPU free memory.'''
        
        cpu_available_pctg, gpu_free = get_memory_info()
        if step is not None:
            message = f"Step {self.step}: CPU available {cpu_available_pctg:.2f}% - GPU free {gpu_free}"
        else:
            message = f"CPU available {cpu_available_pctg:.2f}% - GPU free {gpu_free}"
        
        if logger_level == "INFO":
            logger.info(message)
        elif logger_level == "DEBUG":
            logger.debug(message)
    

    def set_device(self):
        '''Set torch device.'''

        logger.info('Setting device...')

        # Set device to GPU or CPU depending on what is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Running on {self.device} device.")
        
        if self.device == "cuda":
            self.gpus_count = torch.cuda.device_count()
            logger.info(f"{self.gpus_count} GPUs available.")
            # Batch size should be divisible by number of GPUs
        else:
            self.gpus_count = 0
        
        logger.info("Device setted.")

    def load_checkpoint(self):
        """Load trained model checkpoints to continue its training."""
        # Load checkpoint
        checkpoint_path = os.path.join(
            self.params.checkpoint_file_folder, 
            self.params.checkpoint_file_name,
        )

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        self.checkpoint = torch.load(checkpoint_path, map_location = self.device)

        logger.info(f"Checkpoint loaded.") 

    def load_checkpoint_params(self):
        logger.info("Loading checkpoint parameters...")
        self.params = self.checkpoint["settings"]
        logger.info("Checkpoint parameters loaded!")
 
    def set_random_seed(self):

        logger.info("Setting random seed...")

        random.seed(1234)
        np.random.seed(1234)

        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        torch.backends.cudnn.deterministic = True

        logger.info("Random seed setted.")

    def load_network(self):
        """Load the network."""
        logger.info("Loading network...")

        # TODO: load the model
        # Load model class
        #self.net = Classifier(self.params, self.device)
        
        # HACK: naive model
        self.net = Classifier(self.params, self.device)

        if self.params.load_checkpoint == True:
            # TODO
            self.load_checkpoint_network()
            
        # Data Parallelism
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs for training.")
            self.net = nn.DataParallel(self.net)

        self.net.to(self.device)

        logger.info(self.net)

        # Print the number of trainable parameters
        self.total_trainable_params = 0
        parms_dict = {}
        logger.info(f"Detail of every trainable layer:")

        for name, parameter in self.net.named_parameters():

            layer_name = name.split(".")[1]
            if layer_name not in parms_dict.keys():
                parms_dict[layer_name] = 0

            logger.debug(f"name: {name}, layer_name: {layer_name}")

            if not parameter.requires_grad:
                continue
            trainable_params = parameter.numel()

            logger.info(f"{name} is trainable with {parameter.numel()} parameters")
            
            parms_dict[layer_name] = parms_dict[layer_name] + trainable_params
            
            self.total_trainable_params += trainable_params
        
        # Check if this is correct
        logger.info(f"Total trainable parameters per layer:{self.total_trainable_params}")
        for layer_name in parms_dict.keys():
            logger.info(f"{layer_name}: {parms_dict[layer_name]}")

        #summary(self.net, (150, self.params.feature_extractor_output_vectors_dimension))

        logger.info(f"Network loaded, total_trainable_params: {self.total_trainable_params}")

    def load_loss_function(self):
        logger.info("Loading the loss function...")

        if self.params.loss == "CrossEntropy":
            
            # The nn.CrossEntropyLoss() criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class

            if self.params.weighted_loss:
                logger.info("Using weighted loss function...")
                self.loss_function = nn.CrossEntropyLoss(
                    weight = self.training_dataset_classes_weights,
                )
            else:
                logger.info("Using unweighted loss function...")
                self.loss_function = nn.CrossEntropyLoss()

        elif self.params.loss == "FocalLoss":

            if self.params.weighted_loss:
                logger.info("Using weighted loss function...")
                self.loss_function = FocalLossCriterion(
                    gamma = 2,
                    weights = self.training_dataset_classes_weights,
                )
            else:
                logger.info("Using unweighted loss function...")
                self.loss_function = FocalLossCriterion(
                    gamma = 2,
                )
            
        else:
            raise Exception('No Loss choice found.')  

        logger.info("Loss function loaded.")

    def load_checkpoint_optimizer(self):
        logger.info(f"Loading checkpoint optimizer...")
        self.optimizer.load_state_dict(self.checkpoint['optimizer'])

        logger.info(f"Checkpoint optimizer loaded.")
    
    def load_checkpoint_network(self):
        logger.info(f"Loading checkpoint network...")

        try:
            self.net.load_state_dict(self.checkpoint['model'])
        except RuntimeError:    
            self.net.module.load_state_dict(self.checkpoint['model'])

        logger.info(f"Checkpoint network loaded.")

    def load_optimizer(self):
        logger.info("Loading the optimizer...")

        if self.params.optimizer == 'adam':
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.params.learning_rate,
                weight_decay=self.params.weight_decay
            )
        if self.params.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(
                #self.net.parameters(), 
                filter(lambda p: p.requires_grad, self.net.parameters()), 
                lr=self.params.learning_rate, 
                weight_decay=self.params.weight_decay,
                )
        if self.params.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                #self.net.parameters(), 
                filter(lambda p: p.requires_grad, self.net.parameters()), 
                lr=self.params.learning_rate, 
                weight_decay=self.params.weight_decay,
                )       
            
        if self.params.load_checkpoint == True:
            self.load_checkpoint_optimizer()
        logger.info(f"Optimizer {self.params.optimizer} loaded!")

    def init_scheduler(self):
        """
        This function initializes the learning rate scheduler.
        """
        logger.info("Initializing learning rate scheduler...")

        if self.params.learning_rate_scheduler == "NoneScheduler":
            self.scheduler = None
            logger.info("No learning rate scheduler used.")
        elif self.params.learning_rate_scheduler == "LinearLR":
            self.scheduler = lr_scheduler.LinearLR(
                optimizer = self.optimizer,
            )
            logger.info("LinearLR learning rate scheduler used.")

    def set_params(self, input_params):

        '''Set Trainer class parameters.'''

        logger.info("Setting params...")
        
        self.params = input_params
        
        # we transform the argparse.Namespace() into a dictionary
        params_dict = vars(self.params)

        # we print the dictionary in a sorted way:
        for key, value in sorted(params_dict.items()):
            print(f"{key}: {value}")

        # if true, we use pretrained model.
        self.params.load_checkpoint = PRETRAINED_VARIABLES['load_checkpoint']

        # TODO: set the model name
        self.params.model_architecture_name = f"{self.params.feature_extractor}_{self.params.front_end}_{self.params.adapter}_{self.params.seq_to_seq_method}_{self.params.seq_to_one_method}"

        if self.params.use_weights_and_biases:
            self.params.model_name = generate_model_name(
            self.params, 
            start_datetime = self.start_datetime, 
            wandb_run_id = wandb.run.id, 
            wandb_run_name = wandb.run.name 
            )
        else:
            self.params.model_name = generate_model_name(
            self.params, 
            start_datetime = self.start_datetime, 
            )


        if self.params.load_checkpoint == True:

            self.load_checkpoint()
            self.load_checkpoint_params()
            # When we load checkpoint params, all input params are overwriten. 
            # So we need to set load_checkpoint flag to True
            self.params.load_checkpoint = True
            # TODO here we could set a new max_epochs value
            
        logger.info("params setted.")
    #endregion

    #region LoadData

    def format_train_labels(self):

        return format_training_labels(
            labels_path = self.params.train_labels_path,
            labels_to_ids = LABELS_REDUCED_TO_IDS,
            prepend_directory = self.params.train_data_dir,
            header = True,
        )

    def format_validation_labels(self):

        return format_training_labels(
            labels_path = self.params.validation_labels_path,
            labels_to_ids = LABELS_REDUCED_TO_IDS,
            prepend_directory = self.params.validation_data_dir,
            header = True,
        )

    def format_labels(self):

        '''Return (train_labels_lines, validation_labels_lines)'''

        return self.format_train_labels(), self.format_validation_labels()
    
    def load_training_data(self, train_labels_lines):
        logger.info(f"Loading training data with labels from {self.params.train_labels_path}...")

        # TODO: Know if we need this. 
        self.training_wav_mean, self.training_wav_std = get_waveforms_stats(train_labels_lines, self.params.sample_rate)

        training_dataset = TrainDataset(
            utterances_paths = train_labels_lines, 
            input_parameters = self.params,
            random_crop_secs = self.params.training_random_crop_secs,
            augmentation_prob = self.params.training_augmentation_prob,
            sample_rate = self.params.sample_rate,
            waveforms_mean = self.training_wav_mean,
            waveforms_std = self.training_wav_std,
        )

        # printing some information of the dataset.
        logger.info(f"waveform mean: {self.training_wav_mean}")
        logger.info(f"waveform std: {self.training_wav_std}")
        logger.info(f"Printing the waveform {training_dataset[0][0]}")
        logger.info(f"Printing the shape of the waveform: {training_dataset[0][0].shape}")
        logger.info(f"Printing the shape of the labels: {training_dataset[0][1].shape}")
        
        if self.params.text_feature_extractor != 'NoneTextExtractor':
            logger.info(f"Printing the shape of the transcription: {training_dataset[0][2].shape}")

        # If we have data imblance, we need to set the class weights
        if self.params.weighted_loss:
            self.training_dataset_classes_weights = training_dataset.get_classes_weights()
            self.training_dataset_classes_weights = torch.tensor(self.training_dataset_classes_weights).float().to(self.device)

        if self.params.text_feature_extractor != 'NoneTextExtractor':
            data_loader_parameters = {
                'batch_size': self.params.training_batch_size,
                'shuffle': True,
                'num_workers': self.params.num_workers,
                'collate_fn': pad_collate,
            }
        else:
            data_loader_parameters = {
                'batch_size': self.params.training_batch_size,
                'shuffle': True,
                'num_workers': self.params.num_workers,
            }

        # TODO: Dont add to the class to get lighter model
        # TODO: Ask Fede wtf does this mean
        # Instanciate a DataLoader class
        self.training_generator = DataLoader(
            training_dataset,
            **data_loader_parameters,
        )
        del training_dataset

        logger.info("Data and labels loaded!")

    def set_evaluation_batch_size(self):
        """
        This function sets the evaluation batch size to one when we use the whole audio.

        Explanation:
        When we use the whole audio, we have different-size samples. Pytorch dataloaders cannot handle this.
        So, in this case, we set the evaluation batch size to one.
        """
        if self.params.evaluation_random_crop_secs == 0:
            self.params.evaluation_batch_size = 1
            logger.info(f"Setting evaluation batch size to 1 because we are using the whole audio.")

    def load_validation_data(self, validation_labels_lines):
        logger.info(f"Loading data from {self.params.validation_labels_path}...")

        # Instanciate a Dataset class
        validation_dataset = TrainDataset(
            utterances_paths = validation_labels_lines, 
            input_parameters = self.params,
            random_crop_secs = self.params.evaluation_random_crop_secs,
            augmentation_prob = self.params.evaluation_augmentation_prob,
            sample_rate = self.params.sample_rate,
            waveforms_mean = self.training_wav_mean,
            waveforms_std = self.training_wav_std,
        ) 

        # If evaluation_tyoe is total_length, batch size must be 1 because we will ha ve different-size samples
        self.set_evaluation_batch_size()

        if self.params.text_feature_extractor != 'NoneTextExtractor':
            data_loader_parameters = {
                'batch_size': self.params.evaluation_batch_size,
                'shuffle': False,
                'num_workers': self.params.num_workers,
                'collate_fn': pad_collate,
            }
        else:
            data_loader_parameters = {
                'batch_size': self.params.evaluation_batch_size,
                'shuffle': False,
                'num_workers': self.params.num_workers,
            }

        # Instanciate a DataLoader class
        self.evaluating_generator = DataLoader(
            validation_dataset,
            **data_loader_parameters,
        )
        self.evaluation_total_batches = len(self.evaluating_generator)

        del validation_dataset

        logger.info("Data and labels loaded!")

    def load_data(self):
        train_labels_lines, validation_labels_lines = self.format_labels()
        self.load_training_data(train_labels_lines)
        self.load_validation_data(validation_labels_lines)
        del train_labels_lines, validation_labels_lines    
    #endregion
        
    #region Training
    def initialize_training_variables(self):

        logger.info("Initializing training variables...")
        
        if self.params.load_checkpoint == True:

            logger.info(f"Loading checkpoint training variables...")

            loaded_training_variables = self.checkpoint['training_variables']

            # HACK this can be refined, but we are going to continue training \
            # from the last epoch trained and from the first batch
            # (even if we may already trained with some batches in that epoch in the last training from the checkpoint).
            self.starting_epoch = loaded_training_variables['epoch']
            self.step = loaded_training_variables['step'] + 1 
            self.validations_without_improvement = loaded_training_variables['validations_without_improvement']
            self.validations_without_improvement_or_opt_update = loaded_training_variables['validations_without_improvement_or_opt_update'] 
            self.early_stopping_flag = False
            self.train_loss = loaded_training_variables['train_loss'] 
            self.training_eval_metric = loaded_training_variables['training_eval_metric'] 
            self.validation_eval_metric = loaded_training_variables['validation_eval_metric'] 
            self.best_train_loss = loaded_training_variables['best_train_loss'] 
            self.best_model_train_loss = loaded_training_variables['best_model_train_loss'] 
            self.best_model_training_eval_metric = loaded_training_variables['best_model_training_eval_metric'] 
            self.best_model_validation_eval_metric = loaded_training_variables['best_model_validation_eval_metric']
            
            logger.info(f"Checkpoint training variables loaded.") 
            logger.info(f"Training will start from:")
            logger.info(f"Epoch {self.starting_epoch}")
            logger.info(f"Step {self.step}")
            logger.info(f"validations_without_improvement {self.validations_without_improvement}")
            logger.info(f"validations_without_improvement_or_opt_update {self.validations_without_improvement_or_opt_update}")
            logger.info(f"Loss {self.train_loss:.3f}")
            logger.info(f"best_model_train_loss {self.best_model_train_loss:.3f}")
            logger.info(f"best_model_training_eval_metric {self.best_model_training_eval_metric:.3f}")
            logger.info(f"best_model_validation_eval_metric {self.best_model_validation_eval_metric:.3f}")

        else:
            self.starting_epoch = 0
            self.step = 0 
            self.validations_without_improvement = 0 
            self.validations_without_improvement_or_opt_update = 0 
            self.early_stopping_flag = False
            self.train_loss = None
            self.training_eval_metric = 0.0
            self.validation_eval_metric = 0.0
            self.best_train_loss = np.inf
            self.best_model_train_loss = np.inf
            self.best_model_training_eval_metric = 0.0
            self.best_model_validation_eval_metric = 0.0
        
        self.total_batches = len(self.training_generator)

        logger.info("Training variables initialized.")
    
    def check_update_optimizer(self):
        """
        This function checks when it is necessary to update the optimizer. 
        """
        # The conditions to not update the optimizer.
        if self.validations_without_improvement > 0 and self.validations_without_improvement_or_opt_update > 0\
            and self.params.update_optimizer_every > 0 \
            and self.validations_without_improvement_or_opt_update % self.params.update_optimizer_every == 0:
            
            # If we are using one of these optimizers:
            if self.params.optimizer == 'sgd' or self.params.optimizer == 'adam' or self.params.optimizer == 'adamw' and self.scheduler is None:

                logger.info(f"Updating optimizer...")

                for param_group in self.optimizer.param_groups:

                    param_group['lr'] = param_group['lr'] * self.params.learning_rate_multiplier
                    
                    logger.info(f"New learning rate: {param_group['lr']}")
                
                logger.info(f"Optimizer updated.")

            else:
                logger.info(f"You are not using one of the following optimizers: sgd, adam or adamw")
            # We reset the variable 
            self.validations_without_improvement_or_opt_update = 0
            
        # Calculate the actual learning rate:
        # HACK only taking one param group lr as the overall lr (our case has only one param group)
        for param_group in self.optimizer.param_groups:
            self.learning_rate = param_group['lr']
    

    def check_early_stopping(self):
        """
        Check if we have to do early stopping. 
        If the conditions are met, self.early_stopping_flag = True
        """

        if self.params.early_stopping > 0 \
            and self.validations_without_improvement >= self.params.early_stopping:

            self.early_stopping_flag = True
            logger.info(f"Doing early stopping after {self.validations_without_improvement} validations without improvement")

    def check_print_training_info(self):
        if self.step > 0 and self.params.print_training_info_every > 0 \
            and self.step % self.params.print_training_info_every == 0:

            info_to_print = f"Epoch {self.epoch} of {self.params.max_epochs}, "
            info_to_print = info_to_print + f"batch {self.batch_number} of {self.total_batches}, "
            info_to_print = info_to_print + f"step {self.step}, "
            info_to_print = info_to_print + f"Loss {self.train_loss:.3f}, "
            info_to_print = info_to_print + f"Best validation score: {self.best_model_validation_eval_metric:.3f}..."

            logger.info(info_to_print)

            # Uncomment for memory usage info 
            self.info_mem(self.step, logger_level = "DEBUG")        

    def train_single_epoch(self, epoch):

        logger.info(f"Epoch {epoch} of {self.params.max_epochs}...")

        # Switch torch to training mode
        self.net.train()

        for self.batch_number, batch_data in enumerate(self.training_generator):
            if self.params.text_feature_extractor != 'NoneTextExtractor':
                input, label, transcription_tokens_padded, transcription_tokens_mask = batch_data  
            else:
                input, label = batch_data

            # logger.info(f"Input shape: {input.shape}")
            #logger.info(f"input: {input}")
            #logger.info(f"label: {label}")
            #logger.info(f"transcription_tokens_padded: {transcription_tokens_padded}")
            #logger.info(f"transcription_tokens_mask: {transcription_tokens_mask}")

            # Assign batch data to device
            if self.params.text_feature_extractor != 'NoneTextExtractor':
                transcription_tokens_padded = transcription_tokens_padded.long().to(self.device)
                transcription_tokens_mask = transcription_tokens_mask.long().to(self.device)
    
            input, label = input.float().to(self.device), label.long().to(self.device)
            
            if self.batch_number == 0: logger.info(f"input.size(): {input.size()}")

            # Calculate prediction and loss
            if self.params.text_feature_extractor != 'NoneTextExtractor':
                prediction  = self.net(
                    input_tensor = input, 
                    transcription_tokens_padded = transcription_tokens_padded,
                    transcription_tokens_mask = transcription_tokens_mask,
                    )
            else:
                prediction  = self.net(input_tensor = input)

            self.loss = self.loss_function(prediction, label)
            self.train_loss = self.loss.item()

            # Compute backpropagation and update weights
            
            # Clears x.grad for every parameter x in the optimizer. 
            # It’s important to call this before loss.backward(), otherwise you’ll accumulate the gradients from multiple passes.
            self.optimizer.zero_grad()
            
            # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True. 
            # These are accumulated into x.grad for every parameter x.
            self.loss.backward()
            
            # optimizer.step updates the value of x using the gradient x.grad
            self.optimizer.step()

            # Calculate evaluation metrics and save the best model
            self.eval_and_save_best_model()

            # Update best loss
            if self.train_loss < self.best_train_loss:
                self.best_train_loss = self.train_loss

            self.check_update_optimizer()
            self.check_early_stopping()
            self.check_print_training_info()

            if self.params.use_weights_and_biases:
                try:
                    wandb.log(
                        {
                            "epoch" : self.epoch,
                            "batch_number" : self.batch_number,
                            "loss" : self.train_loss,
                            "learning_rate" : self.learning_rate,
                            "training_eval_metric" : self.training_eval_metric,
                            "validation_eval_metric" : self.validation_eval_metric,
                            'best_model_train_loss' : self.best_model_train_loss,
                            'best_model_training_eval_metric' : self.best_model_training_eval_metric,
                            'best_model_validation_eval_metric' : self.best_model_validation_eval_metric,
                        },
                        step = self.step
                        )
                except Exception as e:
                    logger.error('Failed at wandb.log: '+ str(e))

            if self.early_stopping_flag == True: 
                break
            
            self.step = self.step + 1
        
        if self.scheduler is not None:
            self.scheduler.step()

        logger.info(f"-"*50)
        logger.info(f"Epoch {epoch} finished with:")
        logger.info(f"Loss {self.train_loss:.3f}")
        logger.info(f"Best model training evaluation metric: {self.best_model_training_eval_metric:.3f}")
        logger.info(f"Best model validation evaluation metric: {self.best_model_validation_eval_metric:.3f}")
        logger.info(f"-"*50)

    def train(self, starting_epoch, max_epochs):
        logger.info(f'Starting training for {self.params.max_epochs} epochs.')

        for self.epoch in range(starting_epoch, max_epochs):
            self.train_single_epoch(self.epoch)

            logger.info(f"The evaluation metric is {self.validation_eval_metric}")


            if self.early_stopping_flag == True: 
                break

        logger.info('Training finished!')
    #endregion

    def get_classification_report_df(self, y_true, y_pred, labels_ids, labels_names):
        
        pd.set_option('display.width', 1000)

        classification_report_dict = classification_report(
            y_true, 
            y_pred, 
            labels = labels_ids, 
            target_names=labels_names, 
            output_dict = True,
            zero_division = 0,
        )

        df_classification_report = pd.DataFrame(classification_report_dict)
        df_classification_report.index = ['precision', 'recall', 'f1-score', 'true_support']

        pred_support = [list(y_pred).count(class_id) for class_id in labels_ids]
        pred_support = pred_support + [sum(pred_support)] * 3 # we repeat the value for the total columns

        df_classification_report.loc["pred_support"] = pred_support
        #df_classification_report = df_classification_report.round(decimals=4)

        #renamed_columns = ["micro avg" if col_name=="accuracy" else col_name for col_name in list(df_classification_report.columns)]
        #df_classification_report.columns = renamed_columns
        
        cols_to_keep = [col_name for col_name in df_classification_report.columns if col_name != "accuracy"]
        
        df_classification_report = df_classification_report[cols_to_keep]
    
        return df_classification_report

    #region evaluation
    def evaluate_training(self):

        logger.info(f"Evaluating training task...")

        with torch.no_grad():

            # Switch torch to evaluation mode
            self.net.eval()

            final_predictions, final_labels = torch.tensor([]).to("cpu"), torch.tensor([]).to("cpu")
            for batch_number, batch_data in enumerate(self.training_generator):
                
                # HACK Ask Fede where this 1000 comes from? My guess: eval_and_save_best_model_every
                if batch_number % 1000 == 0:
                    logger.info(f"Evaluating training task batch {batch_number} of {len(self.training_generator)}...")

                if self.params.text_feature_extractor != 'NoneTextExtractor':
                    input, label, transcription_tokens_padded, transcription_tokens_mask = batch_data      
                else:
                    input, label = batch_data

                # Assign batch data to device
                if self.params.text_feature_extractor != 'NoneTextExtractor':
                    transcription_tokens_padded, transcription_tokens_mask = transcription_tokens_padded.long().to(self.device), transcription_tokens_mask.long().to(self.device)
                input, label = input.float().to(self.device), label.long().to(self.device)

                if batch_number == 0: logger.info(f"input.size(): {input.size()}")
                
                # Calculate prediction and loss
                if self.params.text_feature_extractor != 'NoneTextExtractor':
                    prediction  = self.net(
                        input_tensor = input, 
                        transcription_tokens_padded = transcription_tokens_padded,
                        transcription_tokens_mask = transcription_tokens_mask,
                        )
                else:
                    prediction  = self.net(input_tensor = input)
                prediction = prediction.to("cpu")
                label = label.to("cpu")

                final_predictions = torch.cat(tensors = (final_predictions, prediction))
                final_labels = torch.cat(tensors = (final_labels, label))
                
            metric_score = f1_score(
                y_true = np.argmax(final_predictions, axis = 1), 
                y_pred = final_labels, 
                average='macro',
                )
            
            self.training_eval_metric = metric_score

            labels_ids = list(range(len(LABELS_REDUCED_TO_IDS)))
            ids_to_labels = {value: key for key, value in LABELS_REDUCED_TO_IDS.items()}
            labels_to_ids = {key: value for key, value in LABELS_REDUCED_TO_IDS.items()}
            labels_names = [ids_to_labels[class_id] for class_id in range(len(LABELS_REDUCED_TO_IDS))]            
            
            logger.info(f"final_labels shape:{final_labels.shape}")
            logger.info(f"final_predictions shape:{final_predictions.shape}")
            final_prediction = np.argmax(final_predictions, axis = 1)
            logger.info(f"final_prediction shape:{final_prediction.shape}")

            training_df_classification_report = self.get_classification_report_df(
                y_true = final_labels, 
                y_pred = final_prediction, 
                labels_ids = labels_ids, 
                labels_names= labels_names,
            )
            
            logger.info(f"Training classification report:{training_df_classification_report}")

            del final_predictions
            del final_labels

        # Return to torch training mode
        self.net.train()

        logger.info(f"Training task evaluated.")
        logger.info(f"F1-score (macro) on training set: {self.training_eval_metric:.3f}")


    def evaluate_validation(self):

        logger.info(f"Evaluating validation task...")

        with torch.no_grad():

            # Switch torch to evaluation mode
            self.net.eval()

            final_predictions, final_labels = torch.tensor([]).to("cpu"), torch.tensor([]).to("cpu")
            for batch_number, batch_data in enumerate(self.evaluating_generator):

                if batch_number % 1000 == 0:
                    logger.info(f"Evaluating validation task batch {batch_number} of {len(self.evaluating_generator)}...")

                if self.params.text_feature_extractor != 'NoneTextExtractor':
                    input, label, transcription_tokens_padded, transcription_tokens_mask = batch_data      
                else:
                    input, label = batch_data

                # Assign batch data to device
                if self.params.text_feature_extractor != 'NoneTextExtractor':
                    # This was originally in .to("cpu") but I changed it to .to(self.device)
                    transcription_tokens_padded, transcription_tokens_mask = transcription_tokens_padded.long().to(self.device), transcription_tokens_mask.long().to(self.device)
                # HACK For the moment we will leave this here.
                
                if torch.cuda.device_count() > 1:
                    input, label = input.float().to("cpu"), label.long().to("cpu")
                else:
                    input, label = input.float().to(self.device), label.long().to(self.device)
                
                
                if batch_number == 0: logger.info(f"input.size(): {input.size()}")

                # Calculate prediction and loss
                if self.params.text_feature_extractor != 'NoneTextExtractor':
                    prediction  = self.net(
                        input_tensor = input, 
                        transcription_tokens_padded = transcription_tokens_padded,
                        transcription_tokens_mask = transcription_tokens_mask,
                        )
                else:
                    prediction  = self.net(input_tensor = input)
                prediction = prediction.to("cpu")
                label = label.to("cpu")

                final_predictions = torch.cat(tensors = (final_predictions, prediction))
                final_labels = torch.cat(tensors = (final_labels, label))

            metric_score = f1_score(
                y_true = np.argmax(final_predictions, axis = 1), 
                y_pred = final_labels, 
                average='macro',
                )
            
            self.validation_eval_metric = metric_score

            labels_ids = list(range(len(LABELS_REDUCED_TO_IDS)))
            ids_to_labels = {value: key for key, value in LABELS_REDUCED_TO_IDS.items()}
            labels_to_ids = {key: value for key, value in LABELS_REDUCED_TO_IDS.items()}
            labels_names = [ids_to_labels[class_id] for class_id in range(len(LABELS_REDUCED_TO_IDS))]            
            
            logger.info(f"final_labels shape:{final_labels.shape}")
            logger.info(f"final_predictions shape:{final_predictions.shape}")
            final_prediction = np.argmax(final_predictions, axis = 1)
            logger.info(f"final_prediction shape:{final_prediction.shape}")

            validation_df_classification_report = self.get_classification_report_df(
                y_true = final_labels, 
                y_pred = final_prediction, 
                labels_ids = labels_ids, 
                labels_names= labels_names,
            )
            
            logger.info(f"Validation classification report:{validation_df_classification_report}")
            
            del final_predictions
            del final_labels

        # Return to training mode
        self.net.train()

        logger.info(f"Validation task evaluated.")
        logger.info(f"F1-score (macro) on validation set: {self.validation_eval_metric:.3f}")


    def evaluate(self):

        self.evaluate_training()
        self.evaluate_validation()
    #endregion

    #region SaveModel
    def save_model(self):

        '''Function to save the model info and optimizer parameters.'''

        # 1 - Add all the info that will be saved in checkpoint 
        
        model_results = {
            'best_model_train_loss' : self.best_model_train_loss,
            'best_model_training_eval_metric' : self.best_model_training_eval_metric,
            'best_model_validation_eval_metric' : self.best_model_validation_eval_metric,
        }

        training_variables = {
            'epoch': self.epoch,
            'batch_number' : self.batch_number,
            'step' : self.step,
            'validations_without_improvement' : self.validations_without_improvement,
            'validations_without_improvement_or_opt_update' : self.validations_without_improvement_or_opt_update,
            'train_loss' : self.train_loss,
            'training_eval_metric' : self.training_eval_metric,
            'validation_eval_metric' : self.validation_eval_metric,
            'best_train_loss' : self.best_train_loss,
            'best_model_train_loss' : self.best_model_train_loss,
            'best_model_training_eval_metric' : self.best_model_training_eval_metric,
            'best_model_validation_eval_metric' : self.best_model_validation_eval_metric,
            'total_trainable_params' : self.total_trainable_params,
        }
        
        if torch.cuda.device_count() > 1:
            checkpoint = {
                'model': self.net.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'settings': self.params,
                'model_results' : model_results,
                'training_variables' : training_variables,
                }
        else:
            checkpoint = {
                'model': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'settings': self.params,
                'model_results' : model_results,
                'training_variables' : training_variables,
                }


        end_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H:%M:%S')
        checkpoint['start_datetime'] = self.start_datetime
        checkpoint['end_datetime'] = end_datetime

        # 2 - Save the checkpoint locally

        checkpoint_folder = os.path.join(self.params.model_output_folder, self.params.model_name)
        checkpoint_file_name = f"{self.params.model_name}.chkpt"
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file_name)

        # Create directory if doesn't exists
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        
        # Save the model parameters in a json file
        logger.info(f"Saving trainable parameters in a json file...")
        self.save_trainable_params_json(self.net, self.params.json_output_folder)
        logger.info(f"The json was saved in {self.params.json_output_folder}")
        
        
        logger.info(f"Saving training and model information in {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Done.")

        # Delete variables to free memory
        del model_results
        del training_variables
        del checkpoint

        logger.info(f"Training and model information saved.")

    def save_trainable_params_json(self, net, saving_path):
        """
        Saves a json file with all of the trainable parameters of the model saved.
        
        """
        
        # create the parent folders if they don't exist
        try:
            os.makedirs(saving_path, exist_ok=True)
        except Exception as e:
            logger.error(f"The folder of the json that contains the trainable params couldn't be created: {e}")
        # print without truncating the output
        np.set_printoptions(threshold=sys.maxsize)
        torch.set_printoptions(threshold=sys.maxsize)

        # create a dictionary that contain the parameter as key and the tensor as value.
        model_dict = {}
        for name, param in net.named_parameters():
            if param.requires_grad:
                model_dict[name]=str(param.data)
        
        # save the json
        json_data = json.dumps(model_dict)
        with open(os.path.join(saving_path, self.params.model_name + ".json"), "w") as file:
            json.dump(model_dict, file, indent=1)

    def eval_and_save_best_model(self):

        if self.step > 0 and self.params.eval_and_save_best_model_every > 0 \
            and self.step % self.params.eval_and_save_best_model_every == 0:

            logger.info('Evaluating and saving the new best model (if founded)...')

            # Calculate the evaluation metrics
            self.evaluate()

            # Have we found a better model? (Better in validation metric).
            if self.validation_eval_metric > self.best_model_validation_eval_metric:

                logger.info('We found a better model!')

               # Update best model evaluation metrics
                self.best_model_train_loss = self.train_loss
                self.best_model_training_eval_metric = self.training_eval_metric
                self.best_model_validation_eval_metric = self.validation_eval_metric

                logger.info(f"Best model train loss: {self.best_model_train_loss:.3f}")
                logger.info(f"Best model train evaluation metric: {self.best_model_training_eval_metric:.3f}")
                logger.info(f"Best model validation evaluation metric: {self.best_model_validation_eval_metric:.3f}")

                self.save_model() 

                # Since we found and improvement, validations_without_improvement and validations_without_improvement_or_opt_update are reseted.
                self.validations_without_improvement = 0
                self.validations_without_improvement_or_opt_update = 0
            
            else:
                # In this case the search didn't improved the model
                # We are one validation closer to do early stopping
                self.validations_without_improvement = self.validations_without_improvement + 1
                self.validations_without_improvement_or_opt_update = self.validations_without_improvement_or_opt_update + 1
                

            logger.info(f"Consecutive validations without improvement: {self.validations_without_improvement}")
            logger.info(f"Consecutive validations without improvement or optimizer update: {self.validations_without_improvement_or_opt_update}")
            logger.info('Evaluating and saving done.')
            self.info_mem(self.step, logger_level = "DEBUG")
        #endregion

    #region artifacts
    def delete_version_artifacts(self):

        logger.info(f'Starting to delete not latest checkpoint version artifacts...')

        # We want to keep only the latest checkpoint because of wandb memory storage limit

        api = wandb.Api()
        actual_run = api.run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
        
        # We need to finish the run and let wandb upload all files
        wandb.run.finish()

        for artifact_version in actual_run.logged_artifacts():
            
            if 'latest' in artifact_version.aliases:
                latest_version = True
            else:
                latest_version = False

            if latest_version == False:
                logger.info(f'Deleting not latest artifact {artifact_version.name} from wandb...')
                artifact_version.delete(delete_aliases=True)
                logger.info(f'Deleted.')

        logger.info(f'All not latest artifacts deleted.')


    def save_model_artifact(self):

        # Save checkpoint as a wandb artifact

        logger.info(f'Starting to save checkpoint as wandb artifact...')

        # Define the artifact
        trained_model_artifact = wandb.Artifact(
            name = self.params.model_name,
            type = "trained_model",
            description = self.params.model_architecture_name,
            metadata = self.wandb_config,
        )

        # Add folder directory
        checkpoint_folder = os.path.join(self.params.model_output_folder, self.params.model_name)
        logger.info(f'checkpoint_folder {checkpoint_folder}')
        trained_model_artifact.add_dir(checkpoint_folder)

        # Log the artifact
        wandb.run.log_artifact(trained_model_artifact)

        logger.info(f'Artifact saved.')
    #endregion 


    def main(self):
        self.train(self.starting_epoch, self.params.max_epochs)
        if self.params.use_weights_and_biases: self.save_model_artifact()
        if self.params.use_weights_and_biases: self.delete_version_artifacts()        



def main():
    args_parser = ArgsParser()
    args_parser()
    trainer_params = args_parser.arguments

    trainer = Trainer(trainer_params)
    trainer.main()

if __name__=="__main__":
    main()