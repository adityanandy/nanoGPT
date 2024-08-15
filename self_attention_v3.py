import torch
import torch.nn as nn 
from torch.nn import functional as F
torch.manual_seed(1337)

# parameters here
batch_size = 32 # independent sequences in parallel
block_size = 8 # max context
max_iters = 3000
eval_interval = 500
learning_rate = 1e-3
eval_iters = 200
n_embd = 32

with open('input.txt','r',encoding='utf-8') as f:
    ts = f.read()


chars = sorted(list(set(ts))) # all the characters in tinyshakespear
vocab_size = len(chars)

# next, we need to turn the strings into integers

s_to_i = {ch:i for i, ch in enumerate(chars)} # this is an encoding of string to int on the sorted list
i_to_s = {i:ch for i, ch in enumerate(chars)} # this is an encoding of string to int on the sorted list

encode = lambda s: [s_to_i[c] for c in s] # turns characters into ints
decode = lambda x: ''.join([i_to_s[i] for i in x]) # turns list of ints into characters

data = torch.tensor(encode(ts), dtype=torch.long)
print(data.shape,data.dtype)
print(data[:1000])

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)- block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad() # we dont call .backward on anything here, more efficient in memory
def estimate_loss():
    out = {}
    model.eval() # model in evaluation phase
    for split in ['train','val']:
        losses = torch.zeros(eval_iters) # compute loss over eval iter times
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # model in training phase
    return out


train_data[:block_size+1] # this is technically the time dimension 

# the block size takes a subset of the data for training. In a chunk of 9 chars --> 8 examples
# in the context of each character, we have the next examples

x = train_data[:block_size] # input to transformer
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f'when input is {context} the target is {target}')

xb, yb = get_batch('train')
print('inputs',xb.shape,xb)
print('targets',yb.shape,yb)

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b,:t+1]
        target = yb[b,t]
        print(f'when input is {context.tolist()} the target is {target}')

class Head(nn.Module):
    # one head of self attention
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)
        # compute attention scores "affinities"
        wei = q @ k.transpose(-2,-1) * C **(-0.5) # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei,dim=-1)
        v = self.value(x)
        out = wei @ v
        return out 

class MultiHeadAttention(nn.Module):
    # multiple heads of self attention in parallel. Its like a group self attention
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,n_embd),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.net(x)
    
# lets start with the bigram language model first, since it is the most basic model
class BigramLanguageModel(nn.Module):
    # bigram is translation invariant

    def __init__(self,vocab_size):
        super().__init__()
        # each token reads off of the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd) # 32 embedding table
        self.position_embedding_table = nn.Embedding(block_size,n_embd) # positional embedding
        self.sa_heads = MultiHeadAttention(4,n_embd//4)
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size) # get logits from the n_embd
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers where B / T are batch and time
        tok_emb = self.token_embedding_table(idx) # returns B, T, C, where C is channel (vocab_size dimensions)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C) --> gets broadcasted across B
        x = self.sa_heads(x)
        x = self.ffwd(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # negative log likelihood loss (same as cross entropy) --> evaluate this loss for what
            # comes next for quality of predictions. Cross entropy wants (B, C, T). Reshape logits.

            B, T, C = logits.shape
            logits = logits.view(B*T,C) # turn this into 2D to make it conform to pytorch dimensions

            # Targets are currently (B, T) but must match logits.
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets) # measures quality of logits with respect to targets

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) array of indices in current context
        # take the input idx --> current context of characters in some batch.
        # take this input and then make it ot be B, T to extend it to be B, T+1, etc
        # extend all batches in time dimension by max_new_tokens (which extends T+1, T+2, etc)
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-block_size:]
            logits, loss = self(idx_cond) # get predictions from current indices. This goes to forward function
            logits = logits[:,-1,:] #becomes (B, C) 
            probs = F.softmax(logits, dim=-1) #(B, C)
            idx_next = torch.multinomial(probs,num_samples=1) #(B, 1), sample 1 example from probabilities
            idx = torch.cat((idx,idx_next),dim=1) # (B, T+1)
        return idx



# make a bigram language model
model = BigramLanguageModel(vocab_size)
# pass inputs and targets
logits, loss = model(xb,yb)
print(logits.shape)
print(loss)

# At this point, the model is garbage and will generate nonsense
# right now, the history is not used because we have a bigram model, but eventually we'll use a history
context = torch.zeros((1,1),dtype=torch.long) # data type is integer 0
print(decode(model.generate(context,max_new_tokens=500)[0].tolist())) # generate works on the level of batches, so we go to 0th row

# Lets train the model now.
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3) # we used AdamW with a higher learning rate because it is a small model

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss() # every once in a while evaluate the loss
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    # eval the loss
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# for steps in range(10000):
#     xb, yb = get_batch('train')
#     logits, loss = m(xb,yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()
# print(loss.item())

# Now that its trained, lets repeat the generation. It looks better. That was the simplest possible bigram model
# The tokens are not talking to each other in this case
# Now we say the tokens have to talk to each other and have to have context
context = torch.zeros((1,1),dtype=torch.long) # data type is integer 0
print(decode(model.generate(context,max_new_tokens=500)[0].tolist())) # generate works on the level of batches, so we go to 0th row
