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

if __name__ == '__main__':
    imgfile = 'rename_total_images_train/subimg_6227_hand.jpg'
    im = Image.open(imgfile)
    w, h = im.size
    image_size = (w, h)
    output_size = (320, 1000)

    input_size = (image_size[0],
                image_size[1])
    m = testModel(input_size, output_size, 1)

    beziers = [[]]
    im_arrs = []
    down_scales = []
    down_scale = get_size(image_size, w, h)
    down_scales.append(down_scale)
    if down_scale > 1:
        im = im.resize((int(w / down_scale), int(h / down_scale)), Image.ANTIALIAS)
        w, h = im.size

    padding = (0, 0, image_size[1] - w, image_size[0] - h)
    im = ImageOps.expand(im, padding)
    im = im.resize((input_size[1], input_size[0]), Image.ANTIALIAS)
    im.show()
    im_arrs.append(np.array(im))

    cps = [[4.0,115.0,122.71996808204982,-103.38141266822475,490.4138845002305,2.162317411197261,531.0,226.0,461.0,253.0,374.32406953057955,135.2859313132704,229.83366308391152,99.89611424545215,92.0,152.0]]
    # cps = [[383.0,556.0,553.1065518592696,440.0130263558785,822.3420395385314,475.7618499943446,934.0,657.0,864.0,723.0,752.1262214272937,628.3219191019219,576.2736508447849,567.9415511648621,447.0,672.0]]


    beziers[0].append(cps)
    beziers = [torch.from_numpy(np.stack(b)).float() for b in beziers]
    beziers = [b / d for b, d in zip(beziers, down_scales)]
    a = beziers[0]
    b = a[0,:,:]
    beziers[0] = b
    print('::::bezier shape::::', beziers[0].shape)

    im_arrs = np.stack(im_arrs)
    x = torch.from_numpy(im_arrs).permute(0, 3, 1, 2).float()

    x = m(x, beziers)
    for i, roi in enumerate(x):
        roi = roi.cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8)
        im = Image.fromarray(roi, "RGB")
        im.save('roi_' + str(i).zfill(3) + '.png')