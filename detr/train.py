from torch import nn
from torch import optim
import torch
from typing import Iterable
from detr import Detr, SetCriterion

def train_fn(model: nn.Module,criterion: nn.Module, data_loader: Iterable, 
    optimizer: optim.Optimizer ,device: torch.device, epoch: int, scheduler):
    
    model.train()
    criterion.train()
    
    summary_loss = AverageMeter()
    
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for step, (images, targets, image_ids) in enumerate(tk0):
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        

        output = model(images)
        
        loss_dict = criterion(output, targets)
        weight_dict = criterion.weight_dict
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        optimizer.zero_grad()

        losses.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        summary_loss.update(losses.item(),BATCH_SIZE)
        tk0.set_postfix(loss=summary_loss.avg)
        
    return summary_loss

if __name__ == "__main__":
    
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # configuration
    num_classes = 10
    null_class_coef = 0.5
    learning_rate = 2e-5
    epoch = 1

    # model
    model = Detr(num_classes=num_classes)
    model.to(device)

    # loss
    weight_dict = {"loss_ce": 1, "loss_bbox": 1, "loss_giou": 1}
    losses = ["labels", "boxes", "cardinality"]
    criterion = SetCriterion(
        num_classes - 1,
        model.matcher,
        weight_dict,
        eos_coef=null_class_coef,
        losses=losses,
    )
    criterion = criterion.to(device)

    # data loader
    data_loader = None

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_fn(model, criterion, data_loader, 
    optimizer, device, epoch, scheduler=None):