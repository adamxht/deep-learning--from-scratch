import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # numberindepedent sequences to process in parallel?
block_size = 8 # max context legnth for predictions?
max_iters = 3000 # max number of weight updates
eval_interval = 300 # number of iterations before running evaluation
learning_rate = 1e-2 
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Both data and model need to be on the same device.
eval_iters = 200

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

# Super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C) - Batch, Time/Step, Channel (vocab_size) - 4, 8, 65

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
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C) - (1, 65) ; Softmax - turn logits into probabilities, where Sum of all values = 1
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) ; Selects the highest probabily token 
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1) - concatenate to list of token/indices instead of streaming
        return idx

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

