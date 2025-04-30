import os
import time
import math

import torch
# import torch.distributed as dist
from data_loader import DataLoader
from model import GPT, GPTConfig
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group

torch.cuda.empty_cache()
RNG = 1337
DATA_DIR = "data"

# ___________________________________Set up DDP_____________________________________
# ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
ddp = False
if ddp:
    assert torch.cuda.is_available(), "for now we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla process
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process= True

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

print(f"using device: {device}")
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(RNG)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RNG)
    
total_batch_size = 524288 # 2**19, roughly 0.5M we see (roughly) half a shard per batch
B = 8
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", data_root=DATA_DIR)
val_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", data_root=DATA_DIR)

torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig(warmup_steps = 715), ddp_rank)
model.to(device)
model = torch.compile(model) # TODO explain this better ?

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_steps = 19073 # TODO bump this up as it only covers half the dataset right now
# max_steps = 1000 # defo not enough but cant do more due to GPU compute constraints

# optimize! the model
# optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), eps=1e-8) # TODO read about AdamW
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=18e-4, device=device)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

scaler = torch.cuda.amp.GradScaler()
val_step = 25

for step in range(max_steps):
    start = time.time()
    last_step = (step == max_steps -1)
    
    if step % val_step == -1 or last_step: #abstract this into a property on the model config or other constant
        model.eval()
        val_loader.reset()
        
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    logits, loss = model(x, y)
                
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
                
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 500 == 0 or last_step):
                    # optionally write model checkpoints
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item()
                    }
                    # you might also want to add optimizer.state_dict() and
                    # rng seeds etc., if you want to resume training
                    torch.save(checkpoint, checkpoint_path)
    
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # switch to the DDP context manager
            
        with torch.autocast(device_type=device_type, dtype=torch.float16): 
            # TODO look into how to get a gradient scaler working
            logits, loss = model(x, y)
        
        # we have to scale the loss to account for gradient accumulation
        # because the gradients just add on each successive backward()
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        scaler.scale(loss).backward()
        
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = model.module.get_lr(step, max_steps) if ddp else model.get_lr(step, max_steps) # try out the scheduler provided by pytorch (if there are any)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    scaler.step(optimizer)
    scaler.update()
    
    if device_type == "cuda":
        torch.cuda.synchronize()
    
    end = time.time()
    total_time = end - start
    miliseconds = total_time*1000
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / total_time
    
    if master_process:
        log_str = f"step: {step:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | time: {miliseconds:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
        print(log_str)
        with open(log_file, "a") as f:
            f.write(f"{log_str}\n")
    
if ddp:
    destroy_process_group()
