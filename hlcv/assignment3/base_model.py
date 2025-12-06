import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod # To be implemented by child classes.
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """

        ret_str = super().__str__()
    
        #### TODO #######################################
        # Print the number of **trainable** parameters  #
        # by appending them to ret_str                  #
        #################################################  
    
        all_parameter_count = 0
        count = 1

        for layer in (self.layers):
            layer_parameter_count = 0
            for param in layer.parameters():
                parameter_dim_count = param.numel()
                if param.requires_grad:
                    layer_parameter_count = layer_parameter_count + parameter_dim_count
                    all_parameter_count = all_parameter_count + parameter_dim_count
                    
            ret_str += "Layer: " + str(count) + " Trainable Parameter Count: " + str(layer_parameter_count) + "\n"
            count = count + 1
            
        ret_str += "Total trainable parameters: " + str(all_parameter_count)
        print("Total number of trainable parameters: ", all_parameter_count)
        
        return ret_str