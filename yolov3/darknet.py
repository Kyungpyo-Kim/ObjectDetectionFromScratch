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
    with open(cfg_file, "r") as file:
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
            conv = Conv2d(
                in_channels=prev_filters,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                bias=bias,
            )
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

    def __init__(self, cfg_file, device="cpu"):
        super().__init__()
        self.blocks = parse_cfg(cfg_file)
        self.net_info, self.module_list = create_modules(self.blocks)
        self.device = device
        self.header = None
        self.seen = None

    def forward(self, x):
        """forward"""
        modules = self.blocks[1:]
        outputs = {}  # We cache the outputs for the route layer

        write = 0  # This is explained a bit later
        for idx, module in enumerate(modules):
            module_type = module["type"]

            if module_type in "convolutional" or module_type in "upsample":
                x = self.module_list[idx](x)

            elif module_type in "route":
                layers = module["layers"]
                layers = [int(layer) for layer in layers.split(",")]

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

            elif module_type in "shortcut":
                from_ = int(module["from"])
                x = outputs[idx - 1] + outputs[idx + from_]

            elif module_type in "yolo":

                anchors = self.module_list[idx][0].anchors
                # Get the input dimensions
                inp_dim = int(self.net_info["height"])

                # Get the number of classes
                num_classes = int(module["classes"])

                # Transform
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, self.device)
                if not write:  # if no collector has been intialised.
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections, x), 1)

            outputs[idx] = x

        return detections

    def load_weights(self, weightfile):
        """load_weights"""
        with open(weightfile, "rb") as file:
            # The first 5 values are header information
            # 1. Major version number
            # 2. Minor Version Number
            # 3. Subversion number
            # 4,5. Images seen by the network (during training)
            header = np.fromfile(file, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]

            weights = np.fromfile(file, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            # If module_type is convolutional load weights
            # Otherwise ignore.
            if module_type in "convolutional":
                model = self.module_list[i]

                if int(self.blocks[i + 1].get("batch_normalize", "0")):
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                else:
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize:
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr : ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr : ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(
                        weights[ptr : ptr + num_bn_biases]
                    )
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(
                        weights[ptr : ptr + num_bn_biases]
                    )
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr : ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the
                    # model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr : ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


def get_test_input():
    """get_test_input"""
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (608, 608))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()  # Convert to float
    img_ = Variable(img_)  # Convert to Variable
    return img_


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Darknet("cfg/yolov3.cfg", device_)
    model.to(device_)
    model.load_weights("cfg/yolov3.weights")

    inp = get_test_input()
    inp = inp.to(device_)

    pred = model(inp)
    print(pred.shape)
