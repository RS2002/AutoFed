import torch
import torch.nn as nn
from utils.AGCRNCell import AGCRNCell

class MLP(nn.Module):
    def __init__(self, layer_sizes=[64,64,1], arl=False, dropout=0.0, bias = True, norm = False):
        super().__init__()
        self.arl = arl
        if self.arl:
            self.attention = nn.Sequential(
                nn.Linear(layer_sizes[0],layer_sizes[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(layer_sizes[0],layer_sizes[0])
            )

        self.norm = norm
        if self.norm:
            self.batch_norm = nn.ModuleList()
            for i in range(1, len(layer_sizes) - 1):
                self.batch_norm.append(nn.BatchNorm1d(layer_sizes[i]))

        self.layer_sizes = layer_sizes
        if len(layer_sizes) < 2:
            raise ValueError()
        self.layers = nn.ModuleList()
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(dropout)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias = bias))

    def forward(self, x):
        if self.arl:
            x = x * self.attention(x)
        for i in range(len(self.layers)-1):
            if self.norm:
                x = self.dropout(self.act(self.batch_norm[i](self.layers[i](x))))
            else:
                x = self.dropout(self.act(self.layers[i](x)))
        x = self.layers[-1](x)
        return x

class AGCRN_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, layer=2):
        super().__init__()
        self.layer = layer
        self.dim_out = dim_out
        self.model = nn.ModuleList()
        self.model.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, self.layer):
            self.model.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, node_embeddings):
        # x: batch, time, node, input_dim
        # node_embeddings: node, hidden_dim
        for i in range(self.layer):
            y = []
            state = torch.zeros([x.shape[0], x.shape[2], self.dim_out]).to(x.device)
            for t in range(x.shape[1]):
                state = self.model[i](x[:,t,...], state, node_embeddings)
                y.append(state)
            y = torch.stack(y,dim=1)
            x = y
        return state # return the last hidden state in the last layer (batch, node, output_dim)


class AGCRN_Decoder(nn.Module):
    def __init__(self, node_num, horizon, dim_in, dim_out, dim_hidden, cheb_k, embed_dim, layer=2):
        super().__init__()
        self.horizon = horizon
        self.layer = layer
        self.dim_out = dim_out

        self.model = nn.ModuleList()
        self.model.append(AGCRNCell(node_num, dim_in + dim_out, dim_hidden, cheb_k, embed_dim))
        for _ in range(1, self.layer):
            self.model.append(AGCRNCell(node_num, dim_hidden, dim_hidden, cheb_k, embed_dim))

        self.predictor = MLP([dim_hidden,dim_hidden,dim_out])

    def forward(self, x_dec, state, node_embeddings, gt=None):
        # x_dec: batch, time, node, input_dim1
        # state: batch, layer, node, input_dim2 (input_dim1+input_dim2=dim_in)
        # node_embeddings: node, hidden_dim
        y = []
        y_prev = torch.zeros([x_dec.shape[0], x_dec.shape[2], self.dim_out]).to(x_dec.device)
        for t in range(self.horizon):
            new_state = []
            x = torch.concatenate([x_dec[:, t, ...], y_prev],dim=-1)  # auto-regressive: current input & last prediction
            for i in range(self.layer):
                x = self.model[i](x, state[:,i,...], node_embeddings)
                new_state.append(x)
            state = torch.stack(new_state,dim = 1)
            y_prev = self.predictor(x)
            y.append(y_prev)

            if gt is not None:
                y_prev = gt[:,t,...]

        return torch.stack(y, dim = 1)

class AGCRN(nn.Module):
    def __init__(self, node_num, horizon, dim_in, dim_in_dec, dim_out, dim_hidden, cheb_k, embed_dim, layer=2):
        super().__init__()
        self.layer = layer

        self.W = nn.Parameter(torch.randn([node_num,embed_dim]), requires_grad=True) # self-adaptive adj
        nn.init.xavier_normal_(self.W)

        self.encoder = AGCRN_Encoder(node_num, dim_in, dim_hidden, cheb_k, embed_dim, layer)
        self.decoder = AGCRN_Decoder(node_num, horizon, dim_in_dec, dim_out, dim_hidden, cheb_k, embed_dim, layer)

    def forward(self, x, x_dec, gt=None):
        hidden_state = self.encoder(x, self.W)
        hidden_state = hidden_state.unsqueeze(1).repeat(1,self.layer,1,1)
        y = self.decoder(x_dec, hidden_state, self.W, gt)
        return y

class AutoFed(nn.Module):
    def __init__(self, node_num, history, horizon, dim_in, dim_in_dec, dim_out, dim_hidden, cheb_k, embed_dim, layer=2):
        super().__init__()
        self.layer = layer
        self.node_num = node_num
        self.dim_hidden = dim_hidden

        self.pattern_mlp_e = MLP([dim_in*history, dim_hidden*history], norm=False)
        self.pattern_mlp_d = MLP([dim_hidden*history, dim_in*history], norm=False)

        self.pattern_mlp = MLP([dim_hidden, dim_hidden, dim_hidden], norm=True)

        self.W = nn.Parameter(torch.randn([node_num,embed_dim]), requires_grad=True) # self-adaptive adj
        nn.init.xavier_normal_(self.W)

        self.encoder = AGCRN_Encoder(node_num, dim_in, dim_hidden, cheb_k, embed_dim, layer)
        self.encoder_low = AGCRN_Encoder(node_num, dim_hidden, dim_hidden, cheb_k, embed_dim, layer)
        # self.encoder_low = self.encoder
        self.decoder = AGCRN_Decoder(node_num, horizon, dim_in_dec, dim_out, dim_hidden + dim_hidden, cheb_k, embed_dim, layer)

        # self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x, x_dec, gt=None):
        # AE filter
        x_low = x.permute(0, 2, 1, 3) # batch * node * time * dim
        x_low = x_low.reshape(x.shape[0], x.shape[2], -1)
        x_low_e = self.pattern_mlp_e(x_low)
        x_low = self.pattern_mlp_d(x_low_e)
        x_low = x_low.reshape(x.shape[0], x.shape[2], x.shape[1], x.shape[3])
        x_low = x_low.permute(0, 2, 1, 3) # batch * time * node * dim

        # x_low = self.act(x_low)
        loss_ae = torch.mean(torch.abs(x_low-x))

        x_low_e = x_low_e.reshape(x.shape[0], x.shape[2], x.shape[1], -1)
        x_low_e = x_low_e.permute(0, 2, 1, 3)
        hidden_state_low = self.encoder_low(x_low_e, self.W)  # batch * node_num * hidden_dim

        # construct pattern
        hidden_state_low = hidden_state_low.reshape(-1,self.dim_hidden)
        hidden_state_low = self.pattern_mlp(hidden_state_low)
        hidden_state_low = hidden_state_low.reshape(-1,self.node_num,self.dim_hidden) # batch * node_num * hidden_dim

        hidden_state = self.encoder(x, self.W)
        hidden_state = torch.concatenate([hidden_state, hidden_state_low], dim=-1)

        hidden_state = hidden_state.unsqueeze(1).repeat(1,self.layer,1,1)
        y = self.decoder(x_dec, hidden_state, self.W, gt)
        # y = self.act(y)

        return y, loss_ae
