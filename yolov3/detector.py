"""detector.py"""
import os
import os.path as osp
import time
import argparse
import pickle as pkl
import random
import sys
import torch
import cv2
from util import load_classes, write_results, prep_image
from yolov3 import Yolov3
import pandas as pd


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description="YOLO v3 Detection Module")

    parser.add_argument(
        "--images",
        dest="images",
        help="Image / Directory containing images to perform detection upon",
        default="imgs",
        type=str,
    )
    parser.add_argument(
        "--det",
        dest="det",
        help="Image / Directory to store detections to",
        default="det",
        type=str,
    )
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument(
        "--confidence",
        dest="confidence",
        help="Object Confidence to filter predictions",
        default=0.5,
    )
    parser.add_argument(
        "--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4
    )
    parser.add_argument(
        "--cfg", dest="cfgfile", help="Config file", default="cfg/yolov3.cfg", type=str
    )
    parser.add_argument(
        "--weights",
        dest="weightsfile",
        help="weightsfile",
        default="cfg/yolov3.weights",
        type=str,
    )
    parser.add_argument(
        "--reso",
        dest="reso",
        help="Input resolution of the network. Increase to increase accuracy."
        " Decrease to increase speed",
        default="416",
        type=str,
    )

    return parser.parse_args()


args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classes = load_classes("data/coco.names")
num_classes = len(classes)  # For COCO

# Set up the neural network
print("Loading network.....")
model = Yolov3(args.cfgfile, device)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

model.eval()

read_dir = time.time()
# Detection phase
try:
    imlist = [osp.join(osp.realpath("."), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath("."), images))
except FileNotFoundError:
    print(f"No file or directory with the name {images}")
    sys.exit()

if not os.path.exists(args.det):
    os.makedirs(args.det)

load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]

# PyTorch Variables for images
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

# List containing dimensions of original images
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2).to(device)
leftover = 0
if len(im_dim_list) % batch_size:
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover
    im_batches = [
        torch.cat(
            (im_batches[i * batch_size : min((i + 1) * batch_size, len(im_batches))])
        )
        for i in range(num_batches)
    ]

write = 0
model.to(device)
start_det_loop = time.time()
with torch.no_grad():
    for i, batch in enumerate(im_batches):
        # load the image
        start = time.time()
        batch = batch.to(device)
        prediction = model(batch)

        prediction = write_results(
            prediction, confidence, num_classes, nms_conf=nms_thesh
        )

        end = time.time()

        if isinstance(prediction, int):

            for im_num, image in enumerate(
                imlist[i * batch_size : min((i + 1) * batch_size, len(imlist))]
            ):
                im_id = i * batch_size + im_num
                print(
                    f"{image.split('/')[-1]:20s} predicted in {(end - start) / batch_size:6.3f} seconds"
                )
                print(f"{'Objects Detected:':20s}")
                print("----------------------------------------------------------")
            continue

        prediction[:, 0] += (
            i * batch_size
        )  # transform the atribute from index in batch to index in imlist

        if not write:  # If we have't initialised output
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        for im_num, image in enumerate(
            imlist[i * batch_size : min((i + 1) * batch_size, len(imlist))]
        ):
            im_id = i * batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print(
                f"{image.split('/')[-1]:20s} predicted in {(end - start) / batch_size:6.3f} seconds"
            )
            print(f"{'Objects Detected:':20s} {' '.join(objs):s}")
            print("----------------------------------------------------------")

        # if "cuda" in device:
        torch.cuda.synchronize()

try:
    output
except NameError:
    print("No detections were made")
    exit()

im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

scaling_factor = torch.min(416 / im_dim_list, 1)[0].view(-1, 1)


output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2


output[:, 1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])


output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("pallete", "rb"))

draw = time.time()


def write(x, results):
    c1 = tuple(x[1:3].int().cpu().numpy())
    c2 = tuple(x[3:5].int().cpu().numpy())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    print(c1, c2, color)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(
        img,
        label,
        (c1[0], c1[1] + t_size[1] + 4),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        [225, 255, 255],
        1,
    )
    return img


list(map(lambda x: write(x, loaded_ims), output))
det_names = pd.Series(imlist).apply(
    lambda x: "{}/det_{}".format(args.det, x.split("/")[-1])
)
print("det_names:", det_names)
for idx, name in enumerate(det_names):
    print(name)
    cv2.imwrite("test.png", loaded_ims[idx])


end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print(f"{'Task':25s}: {'Time Taken (in seconds)'}")
print()
print(f"{'Reading addresses':25s}: {load_batch - read_dir:2.3f}")
print(f"{'Loading batch':25s}: {start_det_loop - load_batch:2.3f}")
print(
    f"{'Detection (' + str(len(imlist)) + ' images)':25s}: {output_recast - start_det_loop:2.3f}"
)
print(f"{'Output Processing':25s}: {class_load - output_recast:2.3f}")
print(f"{'Drawing Boxes':25s}: {end - draw:2.3f}")
print(f"{'Average time_per_img':25s}: {(end - load_batch) / len(imlist):2.3f}")
print("----------------------------------------------------------")

torch.cuda.empty_cache()
