"""
Main model of WaveNet
Calculate loss and optimizing
"""
import os

import torch
import torch.optim

from network.networks import RnnBlock as RnnModule



class CondRNN:
    def __init__(self, cond_size, hidden_size, output_size, n_layers, device, lr=0.002):
        self.device = device
        self.net = RnnModule(cond_size, hidden_size, output_size, n_layers).to(self.device)
        print(self.net)

        self.lr = lr
        self.loss = self._loss()
        self.optimizer = self._optimizer()

    @staticmethod
    def _loss():
        loss = torch.nn.CrossEntropyLoss()
        return loss

    def _optimizer(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def _train_step(self, input, target, hidden):
        """
        Train 1 time
        :param inputs: Tensor[batch, timestep, channels]
        :param targets: Torch tensor [batch, channels, timestep]
        :return: float loss
        """
        outputs, hidden = self.net(input,hidden,input.shape[0])
        loss = self.loss(outputs,target)

        return loss, hidden

    def train(self,inputs,targets):
        hidden = self.net.init_hidden(inputs.shape[0]).to(self.device)
        self.optimizer.zero_grad()
        sequence_loss = 0.

        for timestep in range(inputs.shape[1]):
            loss, hidden = self._train_step(inputs[:,timestep,:], torch.squeeze(targets[:,timestep],1), hidden)
            sequence_loss += loss
        
        sequence_loss.backward()
        self.optimizer.step()
        return sequence_loss/inputs.shape[1] #return average sample loss 

    def generate(self, inputs):
        """
        Generate 1 time
        :param inputs: Tensor[batch, timestep, channels]
        :return: Tensor[batch, timestep, channels]
        """
        outputs = self.net(inputs)

        return outputs

    @staticmethod
    def get_model_path(model_dir, step=0):
        basename = 'model'

        if step:
            return os.path.join(model_dir, '{0}_{1}.pkl'.format(basename, step))
        else:
            return os.path.join(model_dir, '{0}.pkl'.format(basename))

    def load(self, model_dir, step=0):
        """
        Load pre-trained model
        :param model_dir:
        :param step:
        :return:
        """
        print("Loading model from {0}".format(model_dir))

        model_path = self.get_model_path(model_dir, step)

        self.net.load_state_dict(torch.load(model_path, map_location=self.device))

    def save(self, model_dir, step=0):
        print("Saving model into {0}".format(model_dir))

        model_path = self.get_model_path(model_dir, step)

        torch.save(self.net.state_dict(), model_path)

