import torch
from torchvision.models import resnet50

import time

model = resnet50(pretrained=True).cuda()
print(model)

n = 100
start = time.time()
for _ in range(n):
    model(torch.randn(1, 3, 224, 224).cuda())
print((time.time() - start)/n)