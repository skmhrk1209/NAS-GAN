import torch
from torchvision.transforms import functional
import random


class AnnotateVOC(object):

    # NOTE: Background is always index 0
    classes = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    def __call__(self, image, target):

        def generator():

            if not isinstance(target['annotation']['object'], list):
                target['annotation']['object'] = [target['annotation']['object']]

            for obj in target['annotation']['object']:
                # NOTE: Make pixel indexes 0-based
                box = [float(obj['bndbox'][key]) - 1 for key in ['xmin', 'ymin', 'xmax', 'ymax']]
                label = AnnotateVOC.classes.index(obj['name'])
                difficult = int(obj['difficult'])
                yield box, label, difficult

        boxes, labels, difficults = map(torch.tensor, zip(*generator()))
        target.update(dict(boxes=boxes, labels=labels, difficults=difficults))

        return image, target


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = functional.hflip(image)
            target['boxes'][:, [0, 2]] = image.width - target['boxes'][:, [2, 0]]
        return image, target


class ToTensor(object):

    def __call__(self, image, target):
        image = functional.to_tensor(image)
        return image, target


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target
