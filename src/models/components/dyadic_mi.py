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

    dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1
    """

    def __init__(
            self, 
            input_size: int = 769, 
            hidden_size: int = 300,
            num_layers: int = 1,
            dropout: float = 0.,
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
        _, (x, _) = self.bilstm(x)
        return x
    

class ClassiferLayer(nn.Module):
    """A classifier layer.
    In the case of the base line: 
    There is a classifier layer with 300 input features and 2 output classes.
    """

    def __init__(
            self, 
            in_features: int = 300, 
            out_features: int = 2,
        ) -> None:
        """Initialize a `ClassiferLayer` module.

        :param in_features: The number of input features.
        :param out_features: The number of output features.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: The output tensor.
        """
        return self.softmax(self.linear(x))
    


class BaseLineModel(nn.Module):
    """The base line model. 
    There are three inputs for the base line model: BiLSTM_Lang+Face
    1. 6 * (768 RoBERTa feature + 1 speaker ID)
    2. 6 * 674 openface feature for client 
    3. 6 * 674 openface feature for conselor

    Parameters sizes are hard coded for baseline model.
    """

    def __init__(self) -> None:
        """Initialize a `BaseLineModel` module.

        :param num_classes: The number of classes.
        :param lstm_input_size: The number of input features for the BiLSTM layer.
        :param lstm_hidden_size: The number of hidden units for the BiLSTM layer.
        :param lstm_num_layers: The number of LSTM layers.
        :param lstm_dropout: The dropout rate for the BiLSTM layer.
        """
        super().__init__()
        self.linear_client = LinearLayer(600, 200)
        self.linear_counselor = LinearLayer(600, 200)
        self.linear_preclasifier = LinearLayer(1000, 300)
        self.bilstm_RoBERTa = BiLSTMLayer(
            input_size=769,
            hidden_size=300,
            num_layers=1,
        )
        self.bilstm_client = BiLSTMLayer(
            input_size=674,
            hidden_size=300,
            num_layers=1,
        )
        self.bilstm_counselor = BiLSTMLayer(
            input_size=674,
            hidden_size=300,
            num_layers=1,
        )

        self.classifier = ClassiferLayer(300, 2)

    def forward(self, x1, x2, x3: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: The output tensor.
        """
        # unpack inputs or write forward function accepting 3 inputs
        x_RoBERTa, x_client, x_counselor = x1, x2, x3
        # RoBERTa
        x_RoBERTa = self.bilstm_RoBERTa(x_RoBERTa)
        x_RoBERTa = x_RoBERTa.permute(1,0,2)
        x_RoBERTa = x_RoBERTa.flatten(start_dim=1)
        x_client = self.bilstm_client(x_client)
        x_client = x_client.permute(1,0,2)
        x_client = x_client.flatten(start_dim=1)
        x_counselor = self.bilstm_counselor(x_counselor)
        x_counselor = x_counselor.permute(1,0,2)
        x_counselor = x_counselor.flatten(start_dim=1)
        # linear layers
        x_client = self.linear_client(x_client)
        print(x_client.shape)
        x_counselor = self.linear_counselor(x_counselor)
        # concantenate and linear layer
        x = torch.cat((x_RoBERTa, x_client, x_counselor), dim=1)
        x = self.linear_preclasifier(x)
        # classifier
        return self.classifier(x)


if __name__ == "__main__":
    # _ = SimpleDenseNet()
    # a = torch.rand((5,769))
    # model = BiLSTMLayer() 
    # output, (hd, _) = model(a)
    # print(output.shape)
    # print(hd.shape)
    model = BaseLineModel()
    a = torch.rand((5,6,769))
    b = torch.rand((5,6,674))
    c = torch.rand((5,6,674))
    output = model(a,b,c)
    print(output.shape)
