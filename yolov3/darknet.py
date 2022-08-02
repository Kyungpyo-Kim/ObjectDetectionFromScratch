""" darknet.py """
import torch
from torch.nn import ModuleList
from torch.nn import Sequential
from torch.nn import Conv2d
from torch.nn import BatchNorm2d
from torch.nn import LeakyReLU
from torch.nn import Upsample
from torch.nn import Module
from torch.autograd import Variable

import numpy as np
import cv2

from util import predict_transform

CHANNEL_RGB = 3


class EmptyLayer(Module):  # pylint: disable=abstract-method
    """EmptyLayer is an empty layer that doesn't do anything"""


class DetectionLayer(Module):  # pylint: disable=abstract-method
    """DetectionLayer is a YOLOv3 detection layer"""

    def __init__(self, anchors):
        super().__init__()
        self.anchors = anchors


def parse_cfg(cfg_file):
    """
    Takes a configuration file
    Returns a list of blocks
    Each block describes a block in the neural network
    """
    file = open(cfg_file, "r")
    lines = file.read().split("\n")
    lines = [line for line in lines if len(line) > 0]
    lines = [line for line in lines if line[0] != "#"]
    lines = [line.rstrip().lstrip() for line in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def test_parse_cfg():
    """test parse_cfg"""
    blocks = parse_cfg("cfg/yolov3.cfg")
    print("blocks:")
    print(blocks)
    print("len(blocks):")
    print(len(blocks))


def create_modules(
    blocks,
):  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    """create_modules"""
    net_info = blocks[0]
    module_list = ModuleList()
    prev_filters = CHANNEL_RGB
    output_filters = []

    for idx, block in enumerate(blocks[1:]):
        module = Sequential()

        if block.get("type") == "convolutional":
            activation = block.get("activation")
            if block.get("batch_normalize"):
                batch_normalize = int(block["batch_normalize"])
                bias = False
            else:
                batch_normalize = 0
                bias = True

            filters = int(block.get("filters"))
            padding = int(block.get("pad"))
            kernel_size = int(block.get("size"))
            stride = int(block.get("stride"))

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # add convolutional layer
            conv = Conv2d(prev_filters, filters, kernel_size, stride, pad, bias)
            module.add_module(f"conv_{idx:0d}", conv)

            # add batch normalization layer
            if batch_normalize:
                batch_norm = BatchNorm2d(filters)
                module.add_module(f"batch_norm_{idx:0d}", batch_norm)

            # add activation layer
            if activation == "leaky":
                _activation = LeakyReLU(0.1, inplace=True)
                module.add_module(f"leaky_{idx:0d}", _activation)

        elif block.get("type") == "upsample":  # bilinear interpolation
            stride = int(block.get("stride"))
            upsample = Upsample(scale_factor=stride, mode="bilinear")
            module.add_module(f"upsample_{idx:0d}", upsample)

        elif block.get("type") == "route":
            layers = block.get("layers").split(",")
            layers = [int(i) for i in layers]
            start = int(layers[0])
            end = int(layers[1]) if len(layers) > 1 else 0

            # positive indices indicate forward connections
            start = start - idx if start > 0 else start
            end = end - idx if end > 0 else end

            route = Sequential()
            module.add_module(f"route_{idx:0d}".format(idx), route)

            if end < 0:
                filters = output_filters[idx + start] + output_filters[idx + end]
            else:
                filters = output_filters[idx + start]

        elif block.get("type") == "shortcut":  # skip connection
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(idx), shortcut)

        elif block.get("type") == "yolo":
            mask = block.get("mask").split(",")
            mask = [int(x) for x in mask]

            anchors = block.get("anchors").split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module(f"Detection_{idx:0d}", detection)

        else:
            raise ValueError(f'Unknown block type {block.get("type")}')

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)


def test_create_modules():
    """test create_modules"""
    blocks = parse_cfg("cfg/yolov3.cfg")
    print(create_modules(blocks))


class Darknet(Module):
    """Darknet is a YOLOv3 neural network"""

    def __init__(self, cfg_file):
        super().__init__()
        self.blocks = parse_cfg(cfg_file)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, device):
        modules = self.blocks[1:]
        outputs = {}  # We cache the outputs for the route layer

        write = 0  # This is explained a bit later
        for idx, module in enumerate(modules):
            module_type = module["type"]

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[idx](x)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - idx

                if len(layers) == 1:
                    x = outputs[idx + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - idx

                    map1 = outputs[idx + layers[0]]
                    map2 = outputs[idx + layers[1]]

                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[idx - 1] + outputs[idx + from_]

            elif module_type == "yolo":

                anchors = self.module_list[idx][0].anchors
                # Get the input dimensions
                inp_dim = int(self.net_info["height"])

                # Get the number of classes
                num_classes = int(module["classes"])

                # Transform
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, device)
                if not write:  # if no collector has been intialised.
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections, x), 1)

            outputs[idx] = x

        return detections


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))  # Resize to the input dimension
    img_ = img[:, :, ::-1].transpose((2, 0, 1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = (
        img_[np.newaxis, :, :, :] / 255.0
    )  # Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()  # Convert to float
    img_ = Variable(img_)  # Convert to Variable
    return img_


if __name__ == "__main__":
    model = Darknet("cfg/yolov3.cfg")
    inp = get_test_input()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred = model(inp, device)
    print(pred)
