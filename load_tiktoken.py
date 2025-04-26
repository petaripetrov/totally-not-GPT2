import tiktoken
import pickle

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(enc, f)


