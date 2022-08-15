import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def normalize_bbox(bbox, width, height):
    return [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]


def normalize_bboxes(bboxes, width, height):
    return [normalize_bbox(bbox, width, height) for bbox in bboxes]


class TrafficLightDataset(Dataset):
    def __init__(self, base_path, trainvaltest, dataframe, transforms=None):
        self.base_path = base_path
        self.trainvaltest = trainvaltest
        self.dataframe = dataframe[dataframe["trainvaltest"] == self.trainvaltest]
        self.image_ids = list(self.dataframe.image_id.unique())
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index):
        target = None
        image_id = self.image_ids[index]
        records = self.dataframe[self.dataframe["image_id"] == image_id]

        assert len(records.image_name.unique()) == 1
        image_name = records.image_name.unique()[0]

        image = cv2.imread(f"{self.base_path}/{image_name}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # bounding boxes in coco format
        labels = records["category_id"].values
        bboxes = records["bbox"].values
        assert len(labels) == len(bboxes)

        target = {}
        if labels[0] >= 0:
            bboxes = [
                list(map(int, bbox.lstrip("[").rstrip("]").split(",")))
                for bbox in bboxes
            ]

            if self.transforms is not None:
                transformed = self.transforms(
                    image=image, bboxes=bboxes, class_labels=labels
                )
                image = transformed["image"]
                bboxes = transformed["bboxes"]
                labels = transformed["class_labels"]

            height, width, _ = image.shape
            bboxes = normalize_bboxes(bboxes, width, height)

            target["boxes"] = torch.as_tensor(bboxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.long)
            target["image_id"] = torch.tensor([index])

            return image, target, image_id
        else:
            return image, target, image_id


def get_train_transforms():
    return A.Compose(
        [
            A.OneOf(
                [
                    A.HueSaturationValue(
                        hue_shift_limit=0.2,
                        sat_shift_limit=0.2,
                        val_shift_limit=0.2,
                        p=0.9,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.9
                    ),
                ],
                p=0.9,
            ),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="coco", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def get_valid_transforms():
    return A.Compose(
        [A.Resize(height=512, width=512, p=1.0), ToTensorV2(p=1.0)],
        p=1.0,
        bbox_params=A.BboxParams(
            format="coco", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def test():
    import pandas as pd
    import random

    base_path = r"C:\Users\kyung\Downloads\images"
    dataframe = pd.read_csv(f"{base_path}/train.csv")
    transforms = A.Compose(
        [], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"])
    )

    dataset = TrafficLightDataset(base_path, "train", dataframe, transforms)
    for idx in random.sample(range(len(dataset)), 10):
        _ = dataset[idx]
    dataset = TrafficLightDataset(base_path, "val", dataframe, transforms)
    for idx in random.sample(range(len(dataset)), 10):
        _ = dataset[idx]
    dataset = TrafficLightDataset(base_path, "test", dataframe, transforms)
    for idx in random.sample(range(len(dataset)), 10):
        _ = dataset[idx]


if __name__ == "__main__":
    test()
