import torch.nn as nn

class Block(nn.Module):
    """ implementation of the decoder block in our GPT model. """
    def __init__(self, embed_size, dropout_rate):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_size)
        # ToDo: adapt to self defined self attention
        self.attention = CausalSelfAttention()
        self.layer_norm2 = nn.LayerNorm(embed_size)

        # Layers for the MLP
        self.linear1 = nn.Linear(embed_size, 4*embed_size)
        # ToDo: adapt to self defined Gelu
        self.activation = GeLu()
        self.linear2 = nn.Linear(4*embed_size, embed_size)
        self.dropout = nn.Dropout(dropout_rate)

    def mlp_forward(self, x):
        """ forward pass through the MLP with Dropout"""
        x = self.dropout(self.linear2(self.activation(self.linear1(x))))
        return x


    def forward(self, x):
        """forward step for the decoder block"""

        # residual connections
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.mlp_forward(x)

        return x

