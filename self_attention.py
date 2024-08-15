import torch 
import torch.nn as nn 
from torch.nn import functional as F
torch.manual_seed(1337)
B,T,C = 4,8,2 # 8 tokens are currently not talking to each other. We want to couple them
x = torch.randn(B,T,C)
print(x.shape)

# the token in the 5th location should only talk to 4, 3, 2, 1. Information should only
# flow from previous contexts. Future contexts should not be used here. We only communicate with past

# version 1
#we want x[n.t] = mean{i<=t} x[b,i] --> add history
xbow = torch.zeros((B,T,C)) #bow is bag of words
# this is the dumb example. lets do this with matrices below instead
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # t, C
        xbow[b,t] = torch.mean(xprev,0)

# matrix math examples below
# a = torch.ones(3,3)
# b = torch.randint(0,10,(3,2)).float()
# c = a @ b 
# print('a=')
# print(a)
# print('--')
# print('b=')
# print(b)
# print('--')
# print('c=')
# print(c)

# a = torch.tril(torch.ones(3,3)) #lower triangular matrix, used to pluck out the rows and multiply
# a = a /torch.sum(a,1,keepdim=True) # make sure rows sum to 1
# b = torch.randint(0,10,(3,2)).float()
# c = a @ b 
# print('a=')
# print(a)
# print('--')
# print('b=')
# print(b)
# print('--')
# print('c=')
# print(c)

# version 2
wei = torch.tril(torch.ones(T,T)) #weighted aggregation
wei = wei / wei.sum(1,keepdim=True)
print(wei) # allows taking average much easier

xbow2 = wei @ x # (T, T) @ (B, T, C) --> (B, T, T) @ (B, T, C) --> (B, T, C)
# print(xbow)
# print('==')
# print(xbow2)
print(torch.allclose(xbow,xbow2)) # checks if the two are equal to each other. Returns True if so.

# version 3: softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf')) # for all the elements where tril is 0, make it negative inf
wei = F.softmax(wei,dim=-1) #exp everything and divide by sum -- gives same result
xbow3 = wei @ x
print(torch.allclose(xbow,xbow3))

# Now on self attention
B,T,C = 4,8,32
x = torch.randn(B,T,C)
tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril==0,float('-inf'))
wei = F.softmax(wei,dim=-1)
out = wei @ x
print(out.shape)

'''
in self attention, instead of having a weighted average that has even waiting as in above,
you want the weighting to be dependent on the data.

For instance, you probably want vowels to more closely look for consonants.

Self attention solves this.

For every token at each position, we emit two vectors. We emit a query and we emit a key.

Query --> what am I looking for?
Key --> what do I contain?

Dot product between keys and queries gives affinity. The dot product is wei. If key and 
query are aligned, they will learn more.

Attention is a communication mechanism. Can be seen as nodes in a directed graph looking
at each other and aggregating the weighted sum of data pointing to it.

Attention works on graph --> no notion of space. Need to encode position and anchor that.

Self attention --> keys, queries, and values are all from the same source
Cross attention --> queries produced from x, keys and values from different source 

Imagine a scenario in a classroom where a student (Query) is seeking help from various classmates (Keys) who each have different knowledge areas (Values):

Query: Represents the student looking for help on a specific topic.
Key: Represents each classmate's area of expertise or subject knowledge (e.g., one knows math, another knows science).
Value: Represents the actual knowledge or information each classmate can provide (e.g., solving a math problem, explaining a scientific concept).
Here’s how it works:

The student (Query) evaluates each classmate’s subject knowledge (Key) to decide whom to ask for help.
Based on the relevance of each classmate's expertise (Key) to the student’s need (Query), attention scores are assigned.
Each classmate’s response (Value) is weighted by the attention scores, reflecting how much the student (Query) should consider each response.
'''

# we implement a single attention head below
# now, the wei is dependent on the context
head_size = 16
key = nn.Linear(C, head_size, bias = False) # bias = false means matrix multiplication without bases
query = nn.Linear(C, head_size, bias = False)
value = nn.Linear(C, head_size, bias = False)

k = key(x)   # (B, T, 16)
q = query(x) # (B, T, 16)
wei = q @ k.transpose(-2,-1) * head_size **(-0.5) # (B, T, 16) @ (B, 16, T) --> (B, T, T)
print(wei.var(),'weivar')

# when the query and the key have a high dot product, they have a high affinity 

tril = torch.tril(torch.ones(T,T))
wei = wei.masked_fill(tril==0,float('-inf')) # this is the part that limits directionality. Causes nodes from future to not talk to past
wei = F.softmax(wei,dim=-1)
v = value(x) # vectors that we aggregate, instead of the raw x. x is private info.
out = wei @ v

