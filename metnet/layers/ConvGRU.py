"""Implementation of Conv GRU and cell module."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    """The Conv GRU Cell."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size=(3, 3),
        bias=True,
        activation=torch.tanh,
        batchnorm=False,
    ):
        """
        Initialize ConvGRU cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = (
            kernel_size if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 2
        )
        self.padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        self.bias = bias
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv_zr = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=2 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv_h1 = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv_h2 = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.reset_parameters()

    def forward(self, input: torch.Tensor, h_prev=None):
        """Get the current hidden layer of the input layer."""
        # init hidden on forward
        if h_prev is None:
            h_prev = self.init_hidden(input)

        combined = torch.cat((input, h_prev), dim=1)  # concatenate along channel axis

        combined_conv = torch.sigmoid(self.conv_zr(combined))

        z, r = torch.split(combined_conv, self.hidden_dim, dim=1)

        h_ = self.activation(self.conv_h1(input) + r * self.conv_h2(h_prev))

        h_cur = (1 - z) * h_ + z * h_prev

        return h_cur

    def init_hidden(self, input: torch.Tensor):
        """Create and return a hidden layer."""
        bs, ch, h, w = input.shape
        return one_param(self).new_zeros(bs, self.hidden_dim, h, w)

    def reset_parameters(self):
        """Reset the weights and bias of the ConvGRU cell."""
        # self.conv.reset_parameters()
        nn.init.xavier_uniform_(self.conv_zr.weight, gain=nn.init.calculate_gain("tanh"))
        self.conv_zr.bias.data.zero_()
        nn.init.xavier_uniform_(self.conv_h1.weight, gain=nn.init.calculate_gain("tanh"))
        self.conv_h1.bias.data.zero_()
        nn.init.xavier_uniform_(self.conv_h2.weight, gain=nn.init.calculate_gain("tanh"))
        self.conv_h2.bias.data.zero_()

        if self.batchnorm:
            self.bn1.reset_parameters()
            self.bn2.reset_parameters()


def one_param(m):
    """First parameter in `m`."""
    return next(m.parameters())


def dropout_mask(x, sz, p):
    """
    Get the dropout mask of x.

    Return a dropout mask of the same type as `x`, size `sz`, with probability
    `p` to cancel an element.
    """
    return x.new_empty(*sz).bernoulli_(1 - p).div_(1 - p)


class RNNDropout(nn.Module):
    """Dropout with probability `p` that is consistent on the seq_len dimension."""

    def __init__(self, p=0.5):
        """Initialize the RNN dropout layer."""
        super().__init__()
        self.p = p

    def forward(self, x):
        """Calculate the dropout mask."""
        if not self.training or self.p == 0.0:
            return x
        return x * dropout_mask(x.data, (x.size(0), 1, *x.shape[2:]), self.p)


class ConvGRU(nn.Module):
    """Conv GRU."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        n_layers,
        batch_first=True,
        bias=True,
        activation=F.tanh,
        input_p=0.2,
        hidden_p=0.1,
        batchnorm=False,
    ):
        """Initialize the configurations of the conv GRU."""
        super(ConvGRU, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, n_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, n_layers)
        activation = self._extend_for_multilayer(activation, n_layers)

        if not len(kernel_size) == len(hidden_dim) == len(activation) == n_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.bias = bias
        self.input_p = input_p
        self.hidden_p = hidden_p

        cell_list = []
        for i in range(self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvGRUCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                    activation=activation[i],
                    batchnorm=batchnorm,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([nn.Dropout(hidden_p) for layer_index in range(n_layers)])
        self.reset_parameters()

    def __repr__(self):
        """Return a string representation of the configuration options of the conv gru."""
        s = f"ConvGru(in={self.input_dim}, out={self.hidden_dim[0]}, ks={self.kernel_size[0]}, "
        s += f"n_layers={self.n_layers}, input_p={self.input_p}, hidden_p={self.hidden_p})"
        return s

    def forward(self, input, hidden_state=None):
        """

        Pass the input tensor into a sequence of models.

        Parameters
        ----------
        input_tensor:
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state:
        Returns
        -------
        last_state_list, layer_output
        """
        input = self.input_dp(input)
        cur_layer_input = torch.unbind(input, dim=int(self.batch_first))

        if hidden_state is None:
            hidden_state = self.get_init_states(cur_layer_input[0])

        seq_len = len(cur_layer_input)

        last_state_list = []

        for layer_index, (gru_cell, hid_dp) in enumerate(zip(self.cell_list, self.hidden_dps)):
            h = hidden_state[layer_index]
            output_inner = []
            for t in range(seq_len):
                h = gru_cell(input=cur_layer_input[t], h_prev=h)
                output_inner.append(h)

            cur_layer_input = torch.stack(output_inner)  # list to array
            if layer_index != self.n_layers:
                cur_layer_input = hid_dp(cur_layer_input)
            last_state_list.append(h)

        layer_output = torch.stack(output_inner, dim=int(self.batch_first))
        last_state_list = torch.stack(last_state_list, dim=0)
        return layer_output, last_state_list

    def reset_parameters(self):
        """Reset the parameters of each of the conv gru cells in the list."""
        for c in self.cell_list:
            c.reset_parameters()

    def get_init_states(self, input):
        """Collect the init states from the cell list."""
        init_states = []
        for gru_cell in self.cell_list:
            init_states.append(gru_cell.init_hidden(input))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """Check if kernel size is a tuple or is a list of tuples."""
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers: int):
        """Convert the param into a list."""
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
