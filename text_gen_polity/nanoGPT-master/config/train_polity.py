# Train a lightweight GPT model on custom data
out_dir = 'out-polity'
eval_interval = 100    # Evaluates and saves out-polity/ckpt.pt every 100 iterations
eval_iters = 20        # Quick evaluation steps to avoid extra memory usage
log_interval = 10

always_save_checkpoint = True

dataset = 'polity'
batch_size = 8
gradient_accumulation_steps = 4
block_size = 256

# Model architecture (~30M parameters)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1

learning_rate = 1e-3
max_iters = 1000       # Allows the loss to drop closer to ~2.0
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99

device = 'cuda'
compile = False
dtype = 'float16'