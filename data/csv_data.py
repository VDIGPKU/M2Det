import os
import csv
import numpy as np
from PIL import Image

from torch.utils import data


def _transform(img, target):
    pass


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise ValueError(fmt.format(e))


class CustomDataset(data.Dataset):
    """
    CSV-template for Object Detection

    Arguments:

    """
    def __init__(self, anno_path, class_mapping_path, root=None, preproc=None):
        self.preproc = preproc
        self.annopath = anno_path
        self.class_mapping_path = class_mapping_path
        if root is None:
            root = os.path.dirname(self.annopath)
        self.classes = self._load_classes()
        self.image_data = self._load_annotations()
        self.ids = list(self.image_data.keys())

    def __getitem__(self, index):
        idx = self.ids[index]
        img = self.pull_image(idx)
        target = self.pull_anno(idx)
        assert target is not None and img is not None, idx

        if self.preproc is not None:
            img, target = self.preproc(img, target)
        return img, target

    def __len__(self):
        return len(self.ids)

    def pull_image(self, _id):
        return np.array(Image.open(_id))

    def pull_anno(self, _id):
        return self.image_data[_id]

    def _load_classes(self):
        classes = {}
        with open(self.class_mapping_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for line, row in enumerate(reader):
                _cls, index = row
                index = _parse(index, int, 'line {}: malformed: {{}}'.format(line))
                classes[_cls] = index
        return classes

    def _load_annotations(self):
        image_data = {}
        with open(self.annopath, 'r', newline='\n') as f:
            reader = csv.reader(f, delimiter=',')
            for line, row in enumerate(reader):
                try:
                    img_file, x1, y1, x2, y2, class_name = row[:6]
                except ValueError:
                    raise ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line))
                if img_file not in image_data.keys(): # recognize new path
                    if image_data: # already have keys, not empty
                        image_data[prev_path] = np.array(temp, dtype=np.float32)
                    temp = []
                    image_data[img_file] = None
                x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
                y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
                x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
                y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

                # Check that the bounding box is valid.
                if x2 <= x1:
                    raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
                if y2 <= y1:
                    raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

                # check if the current class name is correctly present
                if class_name not in self.classes.keys():
                    raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, self.classes.keys()))

                temp.append([x1, y1, x2, y2, self.classes[class_name]])
                prev_path = img_file
            # collect the last path
            image_data[prev_path] = np.array(temp, dtype=np.float32)
        return image_data

if __name__ == "__main__":
    ds = CustomDataset('/mnt/WORK/Projects/Research_AI/M2Det/train.csv', '/mnt/WORK/Projects/Research_AI/M2Det/class.csv')
    # for i in range(len(ds)):
    #     targets = ds.pull_anno(ds.ids[i])
    #     if targets is None:
    #         print(ds.ids[i])
    #         print(ds.ids[i - 1])
    #         print(i)
