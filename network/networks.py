"""
Base network modules

@muhammad huzaifah 01/11/2019 
"""

import torch 
import torch.nn as nn

class RnnBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        """
        A stack of RNN layers with dense layers at input and output
        cond_size: size of conditional vector
        input_size: if input layer - no. of input audio channels+conditional vectors, 
                    for one-hot audio=mu-law channels + cond vector size
        hidden_size: no. of hidden nodes for each GRU layer
        output_size: size of output, normally=256 for 8-bit mu-law if final layer
        n_layers: no of stacked GRU layers
        """
        super(RnnBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.i2h = nn.Linear(self.input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
        self.h2o = nn.Linear(hidden_size, output_size)
    

    # input and cv are each one sequence element 
    def forward(self, input, hidden, batch_size=1):       
        h1 = self.i2h(input)    
        h_out, hidden = self.gru(h1.view(batch_size,1,-1), hidden)  #hidden shape = (n_layers*n_directions,batch_size,hidden_size)
        output = self.h2o(h_out.view(batch_size,-1))                #h_out shape = (timestep,batch_size,hidden_size*n_directions)
        return output, hidden

    # initialize hiddens for each minibatch
    def init_hidden(self,batch_size=1):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size, dtype=torch.float)#, device=device)


class RnnStack(nn.Module):
    def __init__(self, stack_size):
        super(RnnStack, self).__init__()
        self.stack_size = stack_size