class LIFLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise recurrent connections (LIF).
    ---------
    input_size :    int,   Number of features in the input tensors.
    hidden_size :   int,   Number of output neurons.
    batch_size :    int,   Batch size of the input tensors.
    """

    def __init__(self, input_size, hidden_size,):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=config.use_bias)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])

        self.normalize = False
        if config.normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif config.normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True
        self.drop = nn.Dropout(p=config.pdrop)

    def forward(self, x):
        # Concatenate flipped sequence on batch dim
        if config.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)
        Wx = self.W(x)

        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        s = self.mem_update(Wx)

        # Concatenate forward and backward sequences on feat dim
        if config.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)
        s = self.drop(s)
        return s

    def mem_update(self, Wx):
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(config.device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(config.device)
        s = []
        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
        for t in range(Wx.shape[1]):
            ut = alpha * (ut - st) + (1 - alpha) * Wx[:, t, :]
            st = self.spike_fct(ut - config.threshold)
            s.append(st)
        return torch.stack(s, dim=1)

class adLIFLayer(nn.Module):
    """
    A single layer of adaptive Leaky Integrate-and-Fire neurons without layer-wise recurrent connections (adLIF).
    ---------
    input_size :    int,   Number of features in the input tensors.
    hidden_size :   int,   Number of output neurons.
    batch_size :    int,   Batch size of the input tensors.
    """
    def __init__(self, input_size, hidden_size,):
        super().__init__()
        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=config.use_bias)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
        self.a = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])

        self.normalize = False
        if config.normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif config.normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        self.drop = nn.Dropout(p=config.pdrop)

    def forward(self, x, mask):
        # Concatenate flipped sequence on batch dim
        if config.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        s = self.mem_update(Wx)
        # Concatenate forward and backward sequences on feat dim
        if config.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        s = self.drop(s)
        return s

    def mem_update(self, Wx):
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(config.device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(config.device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(config.device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
        beta = torch.clamp(self.beta, min=self.beta_lim[0], max=self.beta_lim[1])
        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])

        for t in range(Wx.shape[1]):
            wt = beta * wt + a * ut + b * st
            ut = alpha * (ut - st) + (1 - alpha) * (Wx[:, t, :] - wt)
            st = self.spike_fct(ut - config.threshold)
            s.append(st)
        return torch.stack(s, dim=1)

class RLIFLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons with layer-wise recurrent connections (RLIF).
    ---------
    input_size :  int, Number of features in the input tensors.
    hidden_size : int, Number of output neurons.
    batch_size :  int, Batch size of the input tensors.
    """
    def __init__(self, input_size, hidden_size,):
        super().__init__()
        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=config.use_bias)
        self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.orthogonal_(self.V.weight)

        # Initialize normalization
        self.normalize = False
        if config.normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif config.normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True
        
        self.drop = nn.Dropout(p=config.pdrop)

    def forward(self, x, mask):
        # Concatenate flipped sequence on batch dim
        if config.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)
        Wx = self.W(x)

        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])
        s = self.mem_update(Wx)

        # Concatenate forward and backward sequences on feat dim
        if config.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)
        s = self.drop(s)
        return s

    def mem_update(self, Wx):
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(config.device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(config.device)
        s = []
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1]) # Bound values of the neuron parameters to plausible ranges
        V = self.V.weight.clone().fill_diagonal_(0) # Set diagonal elements of recurrent matrix to zero

        for t in range(Wx.shape[1]):
            ut = alpha * (ut - st) + (1 - alpha) * (Wx[:, t, :] + torch.matmul(st, V))
            st = self.spike_fct(ut - config.threshold)
            s.append(st)
        return torch.stack(s, dim=1)