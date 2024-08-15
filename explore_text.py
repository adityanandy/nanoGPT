with open('input.txt','r',encoding='utf-8') as f:
    ts = f.read()

print(len(ts),'# characters in tinyshakespeare')

chars = sorted(list(set(ts))) # all the characters in tinyshakespear
vocab = len(chars)
print("".join(chars),'chars')
print(vocab,'length of vocab')

# next, we need to turn the strings into integers

s_to_i = {ch:i for i, ch in enumerate(chars)} # this is an encoding of string to int on the sorted list
i_to_s = {i:ch for i, ch in enumerate(chars)} # this is an encoding of string to int on the sorted list

encode = lambda s: [s_to_i[c] for c in s] # turns characters into ints
decode = lambda x: ''.join([i_to_s[i] for i in x]) # turns list of ints into characters

print(encode("hii"))
print(decode(encode("hii")))

# sentence piece is a tokenizer for subwords --> done by google. You can have tiktoken too