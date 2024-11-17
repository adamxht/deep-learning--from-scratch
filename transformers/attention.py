import torch
import torch.nn as nn
from torch.nn import functional as F

# # hyperparameters
# batch_size = 32 # numberindepedent sequences to process in parallel?
# block_size = 8 # max context legnth for predictions?
# max_iters = 3000 # max number of weight updates
# eval_interval = 300 # number of iterations before running evaluation
# learning_rate = 1e-3 # self attention cant tolerate high learning rate
# device = 'cuda' if torch.cuda.is_available() else 'cpu' # Both data and model need to be on the same device.
# eval_iters = 200
# n_embed = 32 # Embedding dimension
# n_head = 8 # Number of attention heads
# n_layer = 4
# dropout = 0.5

# hyperparameters - scaled up
batch_size = 32 # numberindepedent sequences to process in parallel?
block_size = 256 # max context legnth for predictions?
max_iters = 5000 # max number of weight updates
eval_interval = 500 # number of iterations before running evaluation
learning_rate = 3e-4 # self attention cant tolerate high learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Both data and model need to be on the same device.
eval_iters = 200
n_embed = 384 # Embedding dimension
n_head = 6 # Number of attention heads
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# Data loader
def get_batch(split, batch_size):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # Random sampling which generates batch_size number of values in between [0, n-block_size)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # Y is always one offset from x
    # Load to GPU
    x, y = x.to(device), y.to(device)
    return x, y

# no_grad skips backpropagation, to be more memory efficient
@torch.no_grad()
def estimate_loss():
    # Averages loss of multiple batches, which is less noisy
    out = {}
    # Set model to inference mode for eval
    model.eval() # Remove layers for training like dropout, batch norm, etc. model in training mode will behave differently from inference mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean() 
    model.train() # Set model back to training mode.
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores ("Affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T), decoder block
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, T)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # Will be run in parallel

        # projection layer for residual connection
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # return torch.cat([h(x) for h in self.heads], dim = -1) # concatenate the output of the forward pass of each heads 

        # Projection back into the residual pathway
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ 
    a simple linear layer followed by non-linearity (ReLU) 
    - It can be added to allow the model to think "longer" for each forward pass.
    - Self attention is the data storing the tokens' communication, this layer allows them to think on the data individually.
    """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), # This is done on per token level (individual tokens). 4 * is just to reproduce the numbers in the paper.
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), # Projection back into the residual pathway
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ 
    Transformer block: communication (attention) followed by computation (feed forward)
    - block contains the attention layers followed by the feed forward layers
    - used to replicate the process multiple times in a full lm architecture.
    """

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # Without residual connections
        # x = self.sa(x)
        # x = self.ffwd(x)

        # With residual connections - This allows the gradients computed from the supervision to propagate all the way to the inputs, solving the vanishing gradient problem.
        x = self.ln1(x) # layernorm
        x = x + self.sa(x)
        x = self.ln2(x) # layernorm
        x = x + self.ffwd(x)
        return x


# Super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embed) # Use embedding dimension instead of vocab_sizes
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # self.sa_head = Head(n_embed)
        # self.sa_heads = MultiHeadAttention(4, n_embed//4) # i.e. with n_embed of 32, 4 heads of 8-dimensional self-attention, concatenated into 32
        # self.ffwd = FeedForward(n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) - Batch, Time/Step/Token, Channel (n_emb)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C) -> 0 until T-1 positions
        x = tok_emb + pos_emb # (B, T, C)
        # x = self.sa_heads(x)
        # x = self.ffwd(x)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, C) - Batch, Time/Step, Channel (vocab size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # Pytorch expects C to be the second dimension
            targets = targets.view(B*T) # Pytorch can guess the shape, we can pass in -1 or explicitly pass in B*T
            loss = F.cross_entropy(logits, targets) # SUM(p(x) log q(x)) ; p(x) - True probability distribution ; q(x) predicted probability distribution

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx (contexts) to the last block_size tokens - to avoid running out of scope in the position embeddings
            idx_cond = idx[:, -block_size:]
            # get the predictions through forward() pass
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C) - (1, 65) ; Softmax - turn logits into probabilities, where Sum of all values = 1
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) ; Selects the highest probabily token 
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1) - concatenate to list of token/indices instead of streaming
        return idx
    
# Validation results - val loss
# Training with single attention head converged to 2.43 instead of 2.5 previously. self attention improved it a bit
# Training with multi head attention converged to 2.34.
# Training with transformer blocks converged to 2.44. As we add more blocks, the nn gets very deep causing issues in optimization. We add residual/skip connections
# Training with residual blocks converged to 2.19. Some English words were generated!
# Training with layernorm converged to 2.14. The only deviation from the original transformers paper is the layer norm, We add layer norm before instead of after the transformation now.
# Training with bigger network (large n_embed, more iters, large block size/context, more layers) converged to 1.51
if __name__ == "__main__":
    # read it in to inspect it
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    model = BigramLanguageModel(vocab_size)
    m = model.to(device)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses["train"]:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train', batch_size)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1,1), dtype=torch.long, device=device) # Contexts needs to be on the same device as the model
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

