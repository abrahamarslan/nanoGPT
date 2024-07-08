"""
This training script can be run both on a single GPU in debug mode,
and also in a larger training run with distributed data parallel (DDP).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 GPUs on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 GPUs across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# Default configuration values
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'

# Wandb logging
wandb_log = False  # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2'

# Data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# Model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?

# AdamW optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# Learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.

# System
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32', 'bfloat16', or 'float16'
compile = True  # use PyTorch 2.0 to compile the model to be faster

# Collect config keys
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

# -----------------------------------------------------------------------------

# Initialize DDP if needed
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

# Tokens per iteration
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

# Set up output directory
if master_process:
    os.makedirs(out_dir, exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Initialize iteration variables
iter_num = 0
best_val_loss = 1e9

# Attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Model initialization
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=None, dropout=dropout)

if init_from == 'scratch':
    # Init a new model from scratch
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    # Resume training from a checkpoint
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    # Initialize from OpenAI GPT-2 weights
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
    gptconf = GPTConfig(**model_args)

# Model to GPU
model.to(device)

# Compile model
if compile:
    print("Compiling the model...")
    model = torch.compile(model)

# Wrap model into DDP if needed
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Setup optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# Setup Wandb logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    wandb.watch(model, log="all")

# Learning rate schedule
def get_lr(iteration):
    if iteration < warmup_iters:
        return learning_rate * iteration / warmup_iters
    if iteration > lr_decay_iters:
        return min_lr
    decay_ratio = (iteration - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Evaluation function
@torch.no_grad()
def evaluate(model):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch('val')
        with ctx:
            logits, loss = model(X, Y)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out

# Training loop
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0  # number of iterations in this epoch
running_mfu = -1.0

while True:
    # Batch acquisition
    if local_iter_num % eval_interval == 0 and master_process:
        val_loss = evaluate(model)
        if val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % log_interval == 0 and master_process:
        print(f"iter {iter_num}: loss {loss.item():.4f}, time {(time.time()-t0)*1000:.2f}ms")
        t0 = time.time()

    # Forward pass
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    iter_num += 1
    local_iter_num += 1
    if iter_num > max_iters:
        break

    # Fetch new batch
    X, Y = get_batch('train')

    # Calculate throughput (MFU)
    if iter_num % log_interval == 0 and iter_num > 0:
        if iter_num < 5:
            continue  # Skip initial iterations to avoid warming-up impact
        else:
            # compute the tokens per second
            mfu = tokens_per_iter / (time.time() - t0) / 1e12
            running_mfu = mfu if running_mfu == -1 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {loss.item():.4f}, time {(time.time()-t0)*1000:.2f}ms, MFU {running_mfu:.2f}")
            t0 = time.time()

if ddp:
    destroy_process_group()
