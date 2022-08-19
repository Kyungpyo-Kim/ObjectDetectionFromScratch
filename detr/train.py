from torch import nn
from torch import optim
import torch
from typing import Iterable
from detr import Detr, SetCriterion
from data.dataset import TrafficLightDataset, get_train_transforms
import pandas as pd
from torch.utils.data import DataLoader
from matcher import HungarianMatcher
from tqdm import tqdm


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scheduler,
):

    model.train()
    criterion.train()

    summary_loss = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    for step, (images, targets, image_ids) in enumerate(tk0):

        images = torch.stack(list(image.to(device) for image in images), dim=0)
        targets = tuple({k: v.to(device) for k, v in t.items()} for t in targets)

        print("output")
        output = model(images)

        print("loss")
        loss_dict = criterion(output, targets)
        weight_dict = criterion.weight_dict

        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        optimizer.zero_grad()

        losses.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        summary_loss.update(losses.item(), BATCH_SIZE)
        tk0.set_postfix(loss=summary_loss.avg)

    return summary_loss


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    # Load the data
    BATCH_SIZE = 4
    EPOCHS = 1
    num_classes = 4
    base_path = r"C:\Users\kyung\Downloads\images"
    dataframe = pd.read_csv(f"{base_path}/train.csv")
    transforms = None
    # transforms = get_train_transforms()
    train_dataset = TrafficLightDataset(base_path, "train", dataframe, transforms)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # model
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Detr(num_classes=num_classes)
    model.to(device)

    # loss
    null_class_coef = 0.5
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

    # optimizer
    learning_rate = 2e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_loss = 10**5
    for epoch in range(EPOCHS):
        train_loss = train_fn(
            model,
            criterion,
            train_dataloader,
            optimizer,
            device,
            epoch=epoch,
            scheduler=None,
        )
        print(f"|EPOCH {epoch+1}| TRAIN_LOSS {train_loss.avg}| VALID_LOSS {0}|")
        # if valid_loss.avg < best_loss:
        #     best_loss = valid_loss.avg
        #     print('Best model found for Fold {} in Epoch {}........Saving Model'.format(fold,epoch+1))
        #     torch.save(model.state_dict(), f'detr_best_{fold}.pth')


if __name__ == "__main__":
    main()

    # train_fn(model, criterion, data_loader,
    # optimizer, device, epoch, scheduler=None):
