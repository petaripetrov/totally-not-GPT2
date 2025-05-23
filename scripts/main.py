import torch
import time
from data_loader import DataLoader
from model import GPT, GPTConfig

RNG = 1337
DATA_DIR = "data"

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

torch.manual_seed(RNG)
torch.cuda.manual_seed_all(RNG)

torch.set_float32_matmul_precision('high')

train_loader = DataLoader(B=4, T=512)

model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)

# optimize! the model
# TODO read into AdamW and other optimizers
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=18e-4, device=device)

for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    
    optimizer.zero_grad()
    logits, loss = model(x, y)
    
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)

    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    
    dt = (t1 - t0) * 1000 # time diff in miliseconds
    
    print(f"step {i} | loss: {loss.item()} | {dt:.2f}ms")