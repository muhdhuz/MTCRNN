"""
credits to Asha Anoosheh: https://gist.github.com/AAnoosheh/c6877583ee5c46d6adf81b5bc355379d
for the following code. Slightly modified to suite this repo.
"""

import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F


class PhasedLSTMCell(nn.Module):
    """Phased LSTM recurrent network cell.
    https://arxiv.org/pdf/1610.09513v1.pdf
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        leak=0.001,
        ratio_on=0.2,
        period_init_min=1.0,
        period_init_max=3.0,
    ):
        """
        Args:
            input_size: int, The length of the input dimension.
            hidden_size: int, The number of units in the Phased LSTM cell.
            leak: float or scalar float Tensor with value in [0, 1]. Leak applied
                during training.
            ratio_on: float or scalar float Tensor with value in [0, 1]. Ratio of the
                period during which the gates are open.
            period_init_min: float or scalar float Tensor. With value > 0.
                Minimum value of the initialized period.
                The period values are initialized by drawing from the distribution:
                e^U(log(period_init_min), log(period_init_max))
                Where U(.,.) is the uniform distribution.
            period_init_max: float or scalar float Tensor.
                With value > period_init_min. Maximum value of the initialized period.
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear_ih = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.linear_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.bias = nn.Parameter(torch.Tensor(4 * hidden_size))
        #fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(4 * hidden_size)
        init.uniform_(self.bias, -bound, bound)

        # initialize standard LSTM parameters
        #stdv = 1.0 / math.sqrt(hidden_size)
        #for weight in self.parameters():
        #    init.uniform_(weight, -stdv, stdv)

        # initialize time-gating parameters
        #period = torch.exp(
        #    torch.Tensor(hidden_size).uniform_(
        #        math.log(period_init_min), math.log(period_init_max)
        #    )
        #)

        period = torch.exp(
            torch.Tensor(hidden_size).uniform_(period_init_min, period_init_max)
            )

        phase = torch.Tensor(hidden_size).uniform_() * period
        ratio_on = torch.Tensor(hidden_size).fill_(ratio_on)

        self.period = nn.Parameter(period) #these are learnable params in phasedlstm so need gradients for backprop
        self.phase = nn.Parameter(phase)
        self.ratio_on = nn.Parameter(ratio_on)
        self.leak = leak

    def _mod(self, x, y):
        """Modulo function that propagates x gradients."""
        return x + (torch.fmod(x, y) - x).detach()

    def forward(self, inputs, state):
        """Takes input as (x, time) tuple.
           Returns output, (h, c) state tuples."""
        h_prev, c_prev = state
        #print("H",h_prev)
        #print("C",c_prev)
        x, time = inputs
        #print("IN",x.shape)

        # Regular LSTM equations
        chunks = self.linear_ih(x) + self.linear_hh(h_prev) + self.bias
        #print("self.linear_ih(x)",self.linear_ih(x))
        #print("self.linear_hh(h_prev)",self.linear_hh(h_prev))
        #print("self.bias",self.bias)
        #print("chunk",chunks)
        xh_ifo, xh_c = chunks.split([3 * self.hidden_size, self.hidden_size], dim=1)
        #print("xh_ifo",xh_ifo)
        #print("xh_c",xh_c)

        input_gate, forget_gate, output_gate = torch.sigmoid(xh_ifo).split(
            self.hidden_size, dim=1
        )

        new_c = forget_gate * c_prev + input_gate * torch.tanh(xh_c)
        #print("new_c",new_c)
        new_h = output_gate * torch.tanh(new_c)
        #print("new_h",new_h)
        # Phase-related augmentations
        tt = time.view(-1, 1).repeat(1, self.hidden_size)
        #print("tt",tt)
        shifted_time = tt - self.phase #eq6
        #print("shifted_time",shifted_time)
        cycle_ratio = self._mod(shifted_time, self.period) / self.period
        #print("cycle_ratio",cycle_ratio)

        k_up = 2 * cycle_ratio / self.ratio_on 
        k_down = 2 - k_up
        k_closed = self.leak * cycle_ratio
        #print("k_up",k_up)
        #print("k_down",k_down)
        #print("k_closed",k_closed)

        #torch.where(condition,if cond met,else)
        k = torch.where(cycle_ratio < self.ratio_on, k_down, k_closed)
        k = torch.where(cycle_ratio < 0.5 * self.ratio_on, k_up, k)
        #print("k",k)
        #print("1-k",(1-k))
        #print("c_prev",c_prev)
        a = k * new_c
        b = (1 - k) * c_prev
        new_c = a + b
        #new_c = k * new_c + (1 - k) * c_prev #eq8
        #print("k * new_c",a)
        #print("(1 - k) * c_prev",b)
        #print("new_c",new_c)
        #print("k * new_h",k * new_h)
        #print("(1 - k) * h_prev",(1 - k) * h_prev)
        new_h = k * new_h + (1 - k) * h_prev
        #print("new_h",new_h)

        return new_h, (new_h, new_c)


class PhasedLSTM(nn.Module):
    """Wrapper for multi-layer sequence forwarding via
       PhasedLSTMCell"""

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size

        self.cells = nn.ModuleList([
            PhasedLSTMCell(input_size, hidden_size)
        ])
        for _ in range(num_layers-1):
            self.cells.append( PhasedLSTMCell(hidden_size, hidden_size) )

        self.num_layers = num_layers

    """
    def _init_hidden_state(self, batch_size):
        zeros = torch.zeros(batch_size, self.hidden_size, dtype=torch.float)
        hidden_states = [
            (zeros, zeros) for _ in range(len(self.cells))
        ]
        return hidden_states
    """
    def forward(self, sequence, times, hidden_states):
        """
        Args:
            sequence: The input sequence data of shape (batch, time, N)
            times: The timestamps corresponding to the data of shape (batch, time)
        

        outputs = []
        for t in range(sequence.size(1)):
            cur_input = (sequence[:, t, :], times[:, t])

            for i, cell in enumerate(self.cells):
                output, hidden_states[i] = cell(cur_input, hidden_states[i])
                cur_input = (output, times[:, t])

            outputs.append(output)
        """
        outputs = []
        cur_input = (sequence, times)

        for i, cell in enumerate(self.cells):
            output, hidden_states[i] = cell(cur_input, hidden_states[i])
            cur_input = (output, times)

            outputs.append(output)
            #print(output)
        #print("O",outputs)
        y = torch.stack(outputs, dim=1)
        #print("Y",y.shape)
        #print(y)
        out = y[:,self.num_layers-1,:]
        #print(out.shape)
        #print(hidden_states)
        return out, hidden_states
