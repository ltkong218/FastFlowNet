import os
import time
import numpy as np
import torch
from models.FastFlowNet import FastFlowNet

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

input_t = torch.randn(1, 6, 448, 1024).cuda()
print(input_t.shape)

model = FastFlowNet().cuda().eval()
model.load_state_dict(torch.load('./checkpoints/fastflownet_ft_mix.pth'))

output_t = model(input_t)
print(output_t.shape)

start = time.time()
for x in range(1000):
    output_t = model(input_t)    
end = time.time()
print('Time elapsed: {:.3f} ms'.format(end-start))

model = model.train()
print('Number of parameters: {:.2f} M'.format(count_parameters(model) / 1e6))
