# import torch
# from torch.nn import ModuleList
# from torch.nn import Sequential
# import torch.nn.functional as F
# from torch.autograd import Variable
import numpy as np

CHANNEL_RGB= 3

def parse_cfg(cfg_file):
    """
    Takes a configuration file
    Returns a list of blocks
    Each block describes a block in the neural network
    """
    file = open(cfg_file, 'r')
    lines = file.read().split('\n')
    lines = [l for l in lines if len(l) > 0]
    lines = [l for l in lines if l[0] != '#']
    lines = [l.rstrip().lstrip() for l in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

def test_parse_cfg():
    blocks = parse_cfg('cfg/yolov3.cfg')
    print('blocks:')
    print(blocks)
    print('len(blocks):')
    print(len(blocks))

def create_modules(blocks):
    net_info = blocks[0]
    # module_list = ModuleList()
    prev_filters = CHANNEL_RGB
    output_filters = []

    for idx, block in enumerate(blocks[1:]):
        # module = Sequential()

        if block.get("type") == 'convolutional':
            activation = block.get('activation')
            
        elif block.get("type") == 'upsample': # bilinear interpolation
            pass
        elif block.get("type") == 'route':
            pass
        elif block.get("type") == 'shortcut': # skip connection
            pass
        else:
            raise ValueError('Unknown block type')


if __name__ == "__main__":
    test_parse_cfg()