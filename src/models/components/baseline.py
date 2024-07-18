import torch
from torch import nn

class LinearLayer(nn.Module):
    """A linear layer.
    In the case of the base line. There are 2 linear layers:
    one with 600 input features and 200 output features 
    the other with 800 input features and 300 output features.
    """

    def __init__(
            self, 
            in_features: int = 784, 
            out_features: int = 10,
            activation = nn.ReLU,
        ) -> None:
        """Initialize a `LinearLayer` module.

        :param in_features: The number of input features.
        :param out_features: The number of output features.
        :param activation: The activation function.
        """
        super().__init__()
        self.activation = activation()
        self.linear = nn.Linear(in_features, out_features)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: The output tensor.
        """
        return self.activation(self.linear(x))


class BiLSTMLayer(nn.Module):
    """A BiLSTM layer.
    In the case of the base line: 
    There are BiLSTM layers with 300 hidden units.
    Dropout rate is not mentioned in the base line.
    """

    def __init__(
            self, 
            input_size: int = 769, 
            hidden_size: int = 300,
            num_layers: int = 1,
            dropout: float = 0.5,
        ) -> None:
        """Initialize a `BiLSTMLayer` module.

        :param input_size: The number of input features.
        :param hidden_size: The number of hidden units.
        :param num_layers: The number of LSTM layers.
        :param dropout: The dropout rate.
        """
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: The output tensor.
        """
        # hiden state and cell state are not used
        # x, (_, _) = self.bilstm(x)
        # the base line uses the hidden layer as the output
        return self.bilstm(x)


if __name__ == "__main__":
    # _ = SimpleDenseNet()
    a = torch.rand((5,769))
    model = BiLSTMLayer() 
    output, (hd, _) = model(a)
    print(output.shape)
    print(hd.shape)
