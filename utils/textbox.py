from .configs import DATA_FOLDER
from skimage import img_as_float32, img_as_ubyte
from skimage.io import imread
from matplotlib import patches, pyplot as plt
from itertools import cycle
import numpy as np
import json
import os

__all__ = ['Anchor', 'BoundingBox', 'parse_ocr_json']
REMOVAL_CHAR = ",;:.*-'\"i#"
JOIN_CHAR = {'SPACE': ' ', 'SURE_SPACE': ' ', 'EOL_SURE_SPACE': '\t', 'LINE_BREAK': '\n', 'HYPHEN': '\t', 'EMPTY': ''}
BOX_COLOR = cycle('bgrcmy')

class Anchor(object):
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    @classmethod
    def make(cls, *vertices):
        x = min(c['x'] for c in vertices)
        y = min(c['y'] for c in vertices)
        w = max(c['x'] for c in vertices) - x
        h = max(c['y'] for c in vertices) - y
        return Anchor(x=x, y=y, w=w, h=h)
    @property
    def xx(self):
        return self.x + self.w
    @property
    def yy(self):
        return self.y + self.h

    def rotate(self, angle):
        if angle in [-90, 270]:
            x = 1 - self.y
            y = self.x
            w = self.h
            h = self.w
            self.x = x - w
            self.y = y
            self.w = w
            self.h = h
        elif angle in [-180, 180]:
            x = 1 - self.x
            y = 1 - self.y
            w = self.w
            h = self.h
            self.x = x - w
            self.y = y - h
        elif angle in [-270, 90]:
            x = self.y
            y = 1 - self.x
            w = self.h
            h = self.w
            self.x = x
            self.y = y - h
            self.w = w
            self.h = h
        else:
            raise ValueError('Only multiples of 90 are supported')

    def __lt__(self, anchor):
        return (self.y < anchor.y) or ((self.y == anchor.y) and self.x < anchor.x)
    def __le__(self, anchor):
        return (self.y <= anchor.y) or ((self.y == anchor.y) and self.x <= anchor.x)
    def __gt__(self, anchor):
        return (self.y > anchor.y) or ((self.y == anchor.y) and self.x > anchor.x)
    def __ge__(self, anchor):
        return (self.y >= anchor.y) or ((self.y == anchor.y) and self.x >= anchor.x)
    def __repr__(self):
        return '({:.6f}, {:.6f}), width={:.6f}, height={:.6f}'.format(self.x, self.y, self.w, self.h)

class BoundingBox(object):
    def __init__(self, *args, **kwargs):
        self.super_box = None
        self.sub_boxes = None
        self.image = kwargs.get('image')
        self.text = kwargs.get('text')
        self.anchor = Anchor.make(*args)

    def __lt__(self, box):
        return self.anchor < box.anchor
    def __gt__(self, box):
        return self.anchor > box.anchor
    def __le__(self, box):
        return self.anchor <= box.anchor
    def __ge__(self, box):
        return self.anchor >= box.anchor
    def __len__(self):
        return len(self.sub_boxes)
    def __getitem__(self, index):
        if self.sub_boxes is None:
            raise IndexError('No sub boxes available')
        return self.sub_boxes[index]
    def __repr__(self):
        return '{} at {}\ntext: {}'.format(self.__class__.__name__, self.anchor, self.text)

    @classmethod
    def aggregate(cls, *boxes, vertices=None):
        image = None
        text = ''
        vertices = []
        for box in boxes:
            if not ((image is None) or (box.image is image)):
                raise ValueError('The given boxes do not belong to the same document')
            if image is None:
                image = box.image
            text += box.text
            vertices.extend([{'x':box.anchor.x, 'y':box.anchor.y}, {'x':box.anchor.xx, 'y':box.anchor.yy}])
        super_box = BoundingBox(*vertices, image=image, text=text)
        super_box.sub_boxes = sorted(boxes)
        for box in boxes:
            box.super_box = super_box
        return super_box

    def visualize(self, padding=1e-2, show_sub_boxes=False):
        h, w = self.image.shape
        padding_pixels = int(min(h, w)*padding)
        upper = max(0, int(h*self.anchor.y)-padding_pixels)
        lower = min(h, int(h*self.anchor.yy)+padding_pixels)
        left = max(0, int(w*self.anchor.x)-padding_pixels)
        right = min(w, int(w*self.anchor.xx)+padding_pixels)
        sub_image = self.image[upper:lower, left:right]
        fig, ax = plt.subplots(1, figsize=(20,20))
        ax.imshow(sub_image, cmap='gray')
        if (show_sub_boxes) and (self.sub_boxes is not None):
            for box, c in zip(self.sub_boxes, BOX_COLOR):
                box_upper = int(h*box.anchor.y) - upper
                box_left = int(w*box.anchor.x) - left
                height = int(h*box.anchor.h)
                width = int(w*box.anchor.w)
                patch = patches.Rectangle((box_left, box_upper), width, height, linewidth=2, edgecolor=c, fill=False,
                        label=box.text[:20])
                ax.add_patch(patch)
        return fig

    def _recursive_rotate(self, angle):
        self.anchor.rotate(angle)
        if self.sub_boxes is not None:
            for box in self.sub_boxes:
                box._recursive_rotate(angle)

    def rotate(self, angle):
        if self.super_box is None:
            self._recursive_rotate(angle)
            new_image = np.rot90(self.image, (360+angle)%360//90)
            self.image.resize(new_image.shape)
            np.copyto(self.image, new_image)
        else:
            self.super_box.rotate(angle)

def parse_ocr_json(guid):
    with open(os.path.join(DATA_FOLDER, 'ocr', '{}_output-1-to-1.json'.format(guid))) as f:
        parsed = json.load(f)
    image = img_as_float32(imread(os.path.join(DATA_FOLDER, 'img', 'train', '{}.png'.format(guid))))
    for page in parsed['responses'][0]['fullTextAnnotation']['pages']:
        block_boxes = []
        for block in page['blocks']:
            paragraph_boxes = []
            for paragraph in block['paragraphs']:
                word_boxes = []
                for word in paragraph['words']:
                    word_text = ''.join(symbol['text'] for symbol in word['symbols'])
                    detected_break = JOIN_CHAR[word['symbols'][-1].get('property',{}).get('detectedBreak',{}).get('type', 'EMPTY')]
                    word_text += detected_break
                    try:
                        word_boxes.append(BoundingBox(*word['boundingBox']['normalizedVertices'], image=image, text=word_text))
                    except KeyError:
                        continue
                if len(word_boxes) > 0:
                    paragraph_box = BoundingBox.aggregate(*word_boxes)
                    purified = ''.join(c for c in paragraph_box.text if not ((c in REMOVAL_CHAR) or (c in JOIN_CHAR.values())))
                    if len(purified) == 0:
                        continue
                    else:
                        paragraph_boxes.append(BoundingBox.aggregate(*word_boxes))
                else:
                    continue
            if len(paragraph_boxes) > 0:
                block_boxes.append(BoundingBox.aggregate(*paragraph_boxes))
            else:
                continue
        if len(block_boxes) > 0:
            page_box = BoundingBox.aggregate(*block_boxes)
        else:
            page_box = None
    return page_box

