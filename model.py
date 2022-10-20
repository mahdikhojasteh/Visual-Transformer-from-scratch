import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_patches(img, patch_size):
    """Given input image extract patches in shape (#N, num_patches, patch_size**2)

    Args:
        img (torch.Tensor): input image of shape (#N, channels, height, width)
        patch_size (int): size of the patches
    
    Returns:
        torch.Tensor: patches in shape (#N, num_patches, patch_size**2)
    """
    pathes = F.unfold(input=img, kernel_size=patch_size, stride=patch_size, padding=0)\
        .permute(0, 2, 1)
    
    return pathes


def positional_embedding(tokens):
    """Adds positional embeddings to the input tokens

    Args:
        tokens (torch.Tensor): input tokens in the shape of (#batch_size, #number_of_sequences, #embedding_size)

    Returns:
        torch.Tensor: tokens with positional embeddings
    """
    embeddings = torch.ones(size=(tokens.size(-2), tokens.size(-1)), device=tokens.device)
    d = tokens.size(-1)
     
    for i in range(len(embeddings)):
        for j in range(d):
            embeddings[i][j] = torch.sin(torch.tensor(i, dtype=torch.float32) / (10000 ** (j / d))) \
                if j % 2 == 0 else torch.cos(torch.tensor(i, dtype=torch.float32) / (10000 ** ((j - 1) / d)))

    embeddings = embeddings.unsqueeze(dim=0)
    tokens = tokens + embeddings
    return tokens


class MSA(nn.Module):
    """Multi-Head Self Attention

    Args:
        tokens_shape (List): shape of tokens
        num_heads (int): number of Multi-Head self Attention heads
    """
    def __init__(self, embed_size, num_heads=2):
        super().__init__()

        self.num_heads = num_heads
        self.embed_size_of_each_head = embed_size // num_heads

        self.Q_linear = nn.ModuleList([nn.Linear(embed_size, self.embed_size_of_each_head) for _ in range(num_heads)])
        self.K_linear = nn.ModuleList([nn.Linear(embed_size, self.embed_size_of_each_head) for _ in range(num_heads)])
        self.V_linear = nn.ModuleList([nn.Linear(embed_size, self.embed_size_of_each_head) for _ in range(num_heads)])

        self.softmax = nn.Softmax(dim=-1)

        self.last_linear = nn.Linear(embed_size, embed_size)

    def forward(self, tokens):
        attention_list = []
        for head in range(self.num_heads):
            Q = self.Q_linear[head](tokens)
            K = self.K_linear[head](tokens)
            V = self.V_linear[head](tokens)

            Q_K = self.softmax((Q @ K.mT) / self.embed_size_of_each_head)

            attention = Q_K @ V
            attention_list.append(attention)
        
        tokens = torch.concat(tensors=attention_list, dim=-1)
        tokens = self.last_linear(tokens)

        return tokens

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_msa_heads, mlp_ratio):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mhsa = MSA(embed_dim, num_heads=num_msa_heads)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, (embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear((embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x):
        # layer normalization 1 and Multi-Head Self Attention
        x = x + self.mhsa(self.layer_norm(x))
        # layer normalization 2
        x = self.layer_norm(x)
        # mlp and residual
        x = x + self.mlp(x)

        return x


class ViT(nn.Module):
    def __init__(
        self, 
        in_channels=1, 
        patch_size=4, 
        linear_Embed_size=8, 
        num_tr_encoders=2, 
        num_msa_heads=2, 
        mlp_ratio=4,
        out_features=10 # MNIST
    ):
        super().__init__()
        # 0) patch-size & enbedding-dim    
        self.patch_size = patch_size
        self.linear_Embed_size = linear_Embed_size
        # 1) Linear mapper
        self.linear_embedding = nn.Linear((patch_size**2)*in_channels, linear_Embed_size)
        # 2) Learnable classification token
        self.cls_token = nn.Parameter(torch.rand(size=(1,self.linear_Embed_size), requires_grad=True))
        # 3) Transformer encoder blocks
        self.transformer_encoder = nn.ModuleList([TransformerEncoder(linear_Embed_size, num_msa_heads, mlp_ratio) \
            for _ in range(num_tr_encoders)])
        # 4) Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(linear_Embed_size, out_features),
            nn.Softmax(dim=-1)
        )
        

    def forward(self, x):
        assert x.ndim == 4 # shape of (#N, channels, height, width)
        patches = extract_patches(x, patch_size=self.patch_size)
        tokens = self.linear_embedding(patches)
        # append CLS token
        tokens = torch.stack([torch.concat([self.cls_token, tokens[i]], dim=0) for i in range(len(tokens))], dim=0)
        # positional embedding
        tokens = positional_embedding(tokens)
        # Transformer Blocks
        for tr_encoder in self.transformer_encoder:
            tokens = tr_encoder(tokens)

        # CLS token
        out = tokens[:, 0]

        return self.mlp(out)

if __name__ == "__main__":
    image = torch.rand(size=(32, 1, 28, 28))
    model = ViT(
        in_channels=1, 
        patch_size=4, 
        linear_Embed_size=8, 
        num_tr_encoders=2,
        num_msa_heads=2, 
        mlp_ratio=4,
        out_features=10 # MNIST classification
    )
    preds = model(image)
    print("image shape:", image.shape)
    print("preds shape:", preds.shape)

    tokens = torch.rand(size=(1, 100, 300))
    tokens = positional_embedding(tokens)
    print("(positional embedding test) tokens shape:", tokens.shape)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7))
    plt.imshow(tokens[0].detach(), cmap="hot", interpolation="nearest")
    plt.show()

    tokens = torch.rand(size=(32, 50, 8))
    msa = MSA(tokens.shape[-1], 2)
    preds = msa(tokens)
    print(f"tokens shape is {tokens.shape}")
    print(f"msa output shape is {preds.shape}")
