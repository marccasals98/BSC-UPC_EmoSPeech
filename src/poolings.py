import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
import math


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


class NoneSeqToSeq(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        logger.info("NoneSeqToSeq initialized")


    def forward(self, input_tensor):

        return input_tensor
    

class SelfAttention(nn.Module):

    """
    Sequence to sequence component, the input dimension is the same than the output dimension.
    Sequence length is not fixed.
    Self-attention without trainable parameters.
    """

    def __init__(self) -> None:
        super().__init__()
        logger.info("SelfAttention initialized")
        

    def forward(self, input_tensor):

        # Perform the matrix multiplication between the input tensor and its transpose.
        raw_weights = torch.bmm(input_tensor, input_tensor.transpose(1, 2))

        # TODO If we want to analyze the attention weights, we should analyze weights
        weights = F.softmax(raw_weights, dim = 2)

        output = torch.bmm(weights, input_tensor)

        return output 

class MultiHeadAttention(nn.Module):

    """
        Sequence to sequence component, the input dimension is the same than the output dimension.
        Sequence length is not fixed.

        Arguments:
        ----------

        emb_in : int
            Dimension of every input vector (embedding). 
        heads : int
            Number of heads to use in the Multi-Head Attention.
    """

    def __init__(self, emb_in, heads) -> None:
        super().__init__()
       
        logger.info("MultiHeadAttention initialized")
        
        self.emb_in = emb_in
        # HACK emb_out is set as emb_in because we want the same dimension in the input and in the output. 
        self.emb_out = emb_in
        self.heads = heads
 
        
        self.init_matrix_transformations()

    def init_matrix_transformations(self):
        
        if self.heads is None:
            raise Exception("The number of heads must be defined. Define it through `seq_to_seq_heads_number` variable")
        # Matrix transformations to stack every head keys, queries and values.
        # shape: (emb_in, emb_out*heads)
        self.to_keys = nn.Linear(self.emb_in, self.emb_out * self.heads, bias = False)
        self.to_queries = nn.Linear(self.emb_in, self.emb_out * self.heads, bias = False)
        self.to_values = nn.Linear(self.emb_in, self.emb_out * self.heads, bias = False)

        # For each input vector we get self.heads heads, so we need to unify them.
        # To do so, we project them into one single vector. 
        # TODO: do we make a projection instead of just concatenating?
        self.unify_heads = nn.Linear(self.emb_out * self.heads, self.emb_in, bias = False)

    def forward(self, input_tensors):
        
        # Get the batch size, sequence length and the dimension of the input vectors.
        b, t, e = input_tensors.size()
        assert e == self.emb_in, f"Input tensor dimension {e} is different than the expected {self.emb_in}"

        # Project the input vectors into keys, queries and values.
        # shape: (b, t, heads, emb_out)
        # TODO: Question: b*t=emb_in/e? 
        keys = self.to_keys(input_tensors).view(b, t, self.heads, self.emb_out)
        queries = self.to_queries(input_tensors).view(b, t, self.heads, self.emb_out)
        values = self.to_values(input_tensors).view(b, t, self.heads, self.emb_out)

        #region 1 - Compute scaled dot-product self-attention 

        # fold heads into the batch dimension.
        # - Swap dimensions 1 and 2
        # - Contiguous???? quilombo
        # - Collapse batch and heads dimensions into one.
        keys = keys.transpose(1, 2).contiguous().view(b * self.heads, t, self.emb_out)  
        queries = queries.transpose(1, 2).contiguous().view(b * self.heads, t, self.emb_out)
        values = values.transpose(1, 2).contiguous().view(b * self.heads, t, self.emb_out)

        # Instead of dividing the dot products by sqrt(e), we scale the queries and the keys
        # this should be more memmory efficient.
        # We already make de division of sqrt(d_k) here. Because we divide it two times, elevate it at 1/4, not 1/2.
        queries = queries / (self.emb_out ** (1/4))
        keys = keys / (self.emb_out ** (1/4))
        
        # Do the product QK^T
        dot = torch.bmm(queries, keys.transpose(1,2))
        
        # let t be the sequence length:
        assert dot.size() == (b * self.heads, t, t), f"Matrix has size {dot.size()}m expected {(b * self.heads, t, t)}"

        # Calculate row-wise self-attention probabilities:
        softmax = F.softmax(dot, dim=2)

        #region 2 - Apply the self attention to the values
        attention = torch.bmm(softmax, values).view(b, self.heads, t, self.emb_out)

        # swap h, t back
        attention = attention.transpose(1, 2).contiguous().view(b, t, self.heads * self.emb_out)

        # unify heads
        multi_head_attention = self.unify_heads(attention)
        #endregion

        return multi_head_attention

class StatisticalPooling(nn.Module):

    """
        Sequence to one component, the input dimension is the same than the output dimension.
        Sequence length is not fixed.
        Given n vectors, takes their average as output.
        emb_in is the dimension of every input vector (embedding).
    """

    def __init__(self, emb_in):

        super().__init__()
        
        self.emb_in = emb_in 


    def forward(self, input_tensors):

        logger.debug(f"input_tensors.size(): {input_tensors.size()}")

        b, t, e = input_tensors.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        # Get the average of the input vectors (dim = 0 is the batch dimension)
        output = input_tensors.mean(dim = 1)

        return output


class AttentionPooling(nn.Module):

    """
        Sequence to one component, the input dimension is the same than the output dimension.
        Sequence length is not fixed.
        Given n vectors, takes their weighted average as output. These weights comes from an attention mechanism.
        It can be seen as a One Head Self-Attention, where a unique query is used and input vectors are the values and keys.   
        emb_in is the dimension of every input vector (embedding).
    """

    def __init__(self, emb_in):

        super().__init__()

        self.emb_in = emb_in
        self.init_query()

        
    def init_query(self):

        # Init the unique trainable query.
        self.query = torch.nn.Parameter(torch.FloatTensor(self.emb_in, 1))
        torch.nn.init.xavier_normal_(self.query)


    def forward(self, input_tensors):

        #logger.debug(f"input_tensors.size(): {input_tensors.size()}")

        #logger.debug(f"self.query[0]: {self.query[0]}")

        b, t, e = input_tensors.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        attention_scores = torch.matmul(input_tensors, self.query)
        #logger.debug(f"attention_scores.size(): {attention_scores.size()}")
        #logger.debug(f"self.query.size(): {self.query.size()}")
        attention_scores = attention_scores.squeeze(dim = -1)
        #logger.debug(f"attention_scores.size(): {attention_scores.size()}")
        attention_scores = F.softmax(attention_scores, dim = 1)
        #logger.debug(f"attention_scores.size(): {attention_scores.size()}")
        attention_scores = attention_scores.unsqueeze(dim = -1)
        #logger.debug(f"attention_scores.size(): {attention_scores.size()}")

        output = torch.bmm(attention_scores.transpose(1, 2), input_tensors)
        #logger.debug(f"output.size(): {output.size()}")
        output = output.view(output.size()[0], output.size()[1] * output.size()[2])
        #logger.debug(f"output.size(): {output.size()}")
        
        return output