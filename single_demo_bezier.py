from PIL import Image, ImageOps
import numpy as np
import torch
from torch import nn
from adet.layers.bezier_align import BezierAlign

class testModel(nn.Module):
    def __init__(self, input_size, output_size, scale):
        super(testModel, self).__init__()
        self.bezier_align = BezierAlign(output_size, scale, 1)
        self.masks = nn.Parameter(torch.ones(input_size, dtype=torch.float32))

    def forward(self, input, rois):
        x = input * self.masks
        rois = self.convert_to_roi_format(rois)
        return self.bezier_align(x, rois)

    def convert_to_roi_format(self, beziers):
        concat_boxes = cat([b for b in beziers], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(beziers)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

def get_size(image_size, w, h):
    w_ratio = w / image_size[1]
    h_ratio = h / image_size[0]
    down_scale = max(w_ratio, h_ratio)
    if down_scale > 1:
        return down_scale
    else:
        return 1

def cat(tensors, dim=0):
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def single_image_bezier(img,control_points,output_size):
    # if type(img).__module__ == np.__name__:
    #     img = Image.fromarray(img)
    h, w = img.shape[:2]
    image_size = (w, h)
    m = testModel(image_size, output_size, 1)

    beziers = [[]]
    im_arrs = [img]
    # down_scales = []
    # down_scale = get_size(image_size, w, h)
    # down_scales.append(down_scale)
    # if down_scale > 1:
    #     img = img.resize((int(w / down_scale), int(h / down_scale)), Image.ANTIALIAS)
    #     w, h = img.size

    # padding = (0, 0, image_size[1] - w, image_size[0] - h)
    # img = ImageOps.expand(img, padding)
    # img = img.resize((image_size[1], image_size[0]), Image.ANTIALIAS)
    # img.show()
    # im_arrs.append(np.array(img))

    beziers[0].append(control_points)
    beziers = [torch.from_numpy(np.stack(b)).float() for b in beziers]
    # beziers = [b / d for b, d in zip(beziers, down_scales)]
    a = beziers[0]
    b = a[0,:,:]
    beziers[0] = b
    # print('::::bezier shape::::', beziers[0].shape)

    im_arrs = np.stack(im_arrs)
    x = torch.from_numpy(im_arrs).permute(0, 3, 1, 2).float()

    x = m(x, beziers)
    imgs = []
    for i, roi in enumerate(x):
        roi = roi.cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8)
        img = Image.fromarray(roi, "RGB")
        imgs.append(img)
    return imgs

        