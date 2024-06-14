import torch
from torch import nn
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

class NoneAdapter(torch.nn.Module):

    def __init__(self):
        super().__init__()
        logger.info("NoneAdapter initialized")
    

    def forward(self, input_tensor):

        return input_tensor
    

class LinearAdapter(torch.nn.Module):

    def __init__(self, input_vectors_dimension, output_vectors_dimension):
        super().__init__()

        logger.info(f"LinearAdapter initialized with input_vectors_dimension: {input_vectors_dimension} and output_vectors_dimension: {output_vectors_dimension}")
        self.input_vectors_dimension = input_vectors_dimension
        self.output_vectors_dimension = output_vectors_dimension
        self.adapter_layer =  nn.Sequential(
            nn.Linear(self.input_vectors_dimension, self.output_vectors_dimension),
            nn.LayerNorm(self.output_vectors_dimension),
        )
    
    
    def forward(self, input_tensor):

        output_tensor = self.adapter_layer(input_tensor)

        return output_tensor
    

class NonLinearAdapter(torch.nn.Module):

    def __init__(self, input_vectors_dimension, output_vectors_dimension):
        super().__init__()

        logger.info(f"NonLinearAdapter initialized with input_vectors_dimension: {input_vectors_dimension} and output_vectors_dimension: {output_vectors_dimension}")
        self.input_vectors_dimension = input_vectors_dimension
        self.output_vectors_dimension = output_vectors_dimension
        self.adapter_layer =  nn.Sequential(
            nn.Linear(self.input_vectors_dimension, self.output_vectors_dimension),
            nn.LayerNorm(self.output_vectors_dimension),
            nn.ReLU(),
        )
    
    
    def forward(self, input_tensor):

        output_tensor = self.adapter_layer(input_tensor)

        return output_tensor