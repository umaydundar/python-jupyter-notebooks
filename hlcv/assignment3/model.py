import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import torch.nn as nn
from ..base_model import BaseModel


class ConvNet(BaseModel):
    def __init__(self, input_size, hidden_layers, num_classes, activation, norm_layer, drop_prob=0.0):
        
        super(ConvNet, self).__init__()

        ############## TODO ###############################################
        # Initialize the different model parameters from the config file  #
        # (basically store them in self)                                  #
        ###################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.activation = activation
        self.norm_layer = norm_layer
        self.drop_prob = drop_prob
        self.layers = None
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._build_model()

    def _build_model(self):

        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module if drop_prob > 0          #
        # Do NOT add any softmax layers.                                                #
        #################################################################################
        layers = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
     
         
        for i in range(len(self.hidden_layers)):
            if i == 0:
                layers.append(nn.Conv2d(self.input_size, self.hidden_layers[i], kernel_size=3, stride=1, padding=1))
                layers.append(self.norm_layer(self.hidden_layers[i]))
                layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
                layers.append(self.activation())
                if self.drop_prob > 0:
                    layers.append(nn.Dropout(self.drop_prob))
            elif i == len(self.hidden_layers) - 1:
                layers.append(nn.Flatten())
                layers.append(nn.Linear(self.hidden_layers[i], self.num_classes))
            else:
                layers.append(nn.Conv2d(self.hidden_layers[i-1], self.hidden_layers[i], kernel_size=3, stride=1, padding=1))
                layers.append(self.norm_layer(self.hidden_layers[i]))
                layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
                layers.append(self.activation())
                if self.drop_prob > 0:
                    layers.append(nn.Dropout(self.drop_prob))
                
        self.layers = nn.Sequential(*layers)
            
                
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def _normalize(self, img):
        """
        Helper method to be used for VisualizeFilter. 
        This is not given to be used for Forward pass! The normalization of Input for forward pass
        must be done in the transform presets.
        """
        max = np.max(img)
        min = np.min(img)
        return (img-min)/(max-min)    
    
    def VisualizeFilter(self):
        ################################################################################
        # TODO: Implement the functiont to visualize the weights in the first conv layer#
        # in the model. Visualize them as a single image fo stacked filters.            #
        # You can use matlplotlib.imshow to visualize an image in python                #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        weights = self.layers[0].weight.cpu().data.clone().numpy()

        columns = 16
        rows = int(weights.shape[0] / columns)
        size = columns * 0.8, rows * 0.8
        fig, axs = plt.subplots(rows, columns, figsize = size)

        for i in range(weights.shape[0]):
            filter_img = weights[i][:3]
            filter_img = np.transpose(filter_img, (1, 2, 0)) 
            filter_img = self._normalize(filter_img)

            row_index = int(math.floor(i / columns))
            column_index = i % columns
        
            sub_plt = axs[row_index, column_index]
            sub_plt.imshow(filter_img)
            sub_plt.axis('off')

        fig.set_facecolor("black")
        plt.tight_layout()
        plt.show()


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #############################################################################
        # TODO: Implement the forward pass computations                             #
        # This can be as simple as one line :)
        # Do not apply any softmax on the logits.                                   #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****        

        out = self.layers(x)
    
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out
