import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from PIL import Image
import os
from skimage import img_as_float
from skimage.segmentation import slic, find_boundaries

def calculate_metric_percase(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        if pred.sum() == 0 and gt.sum() == 0:
            return 1.0, 0.0 
        else:
            return 0.0, 0.0


def test_single_volume(image, label, model, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        model.eval()
        with torch.no_grad():
            output = model(input)

            if isinstance(output, (list, tuple)):
                logits = output[0] 
            else:
                logits = output

            out = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list

def test_single_volume_diffusion(image, label, model, classes, test_save_path, iter_num, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    superpixel = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        label_slice = label[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        label_slice = zoom(label_slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        label_slice = torch.from_numpy(label_slice).unsqueeze(0).unsqueeze(0).float().cuda()
        model.eval()

        y1, segments = test_single_case_2d(
            model,
            input, 
            label_slice,
            patch_size,
            num_classes=classes
        )
        if y1.dtype == torch.float32 or y1.dtype == torch.float64:
            y1 = (y1 * 255).to(torch.uint8)
        y1 = y1.argmax(dim=0)
        pred_np = y1.byte().cpu().numpy()
        segment = segments[0][0].byte().cpu().numpy()
        pred = zoom(pred_np, (x / patch_size[0], y / patch_size[1]), order=0)
        superpixel1 = zoom(segment, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction[ind] = pred
        superpixel[ind] = superpixel1
    return prediction, superpixel

def test_single_case_2d(net, image, label, patch_size, num_classes):

    segments, contours = slic_segmentation_batch(image, num_classes)
    segments = segments.cuda()
    x_t, t, noise = net(x=[segments, label, contours], pred_type="q_sample")
    p_l_xi = net(x=x_t, step=t, image=image, pred_type="D_xi_l")

    if isinstance(p_l_xi, (tuple, list)):
        p_l_xi = p_l_xi[0] 
    y1 =torch.softmax(p_l_xi, dim=1) 
    y1 = y1[0, ...]

    return y1, segments


def slic_segmentation_batch(tensor_imgs, n_segments=50, compactness=0.5):
    segments_list = []
    contours_list = []

    for img in tensor_imgs:
        img_np = img[0].detach().cpu().numpy()
        img_float = img_as_float(img_np)

        segments = slic(img_float, n_segments=n_segments, 
                       compactness=compactness, channel_axis=None)
        segments_list.append(segments)
        
        contour = find_boundaries(segments, mode='thick').astype(np.uint8)
        contours_list.append(contour)

    superpixels = np.stack(segments_list, axis=0)
    superpixels = np.expand_dims(superpixels, axis=1)
    superpixels = torch.from_numpy(superpixels).cuda()

    contours = np.stack(contours_list, axis=0)
    contours = np.expand_dims(contours, axis=1)
    contours = torch.from_numpy(contours).cuda()

    return superpixels, contours

if __name__ == "__main__":
    from ..DiffVNet.diff_vnet_2d import DiffVNet
    from torch.utils.data import DataLoader
    from dataloaders.dataset_2D import *

    def make_model_all(num_classes):
        model = DiffVNet(
            n_channels=1,
            n_classes=num_classes,
            n_filters=32,
            normalization='batchnorm',
            has_dropout=True
        ).cuda()

        return model
    
    def colorize_mask(mask):
        palette = np.random.randint(0, 256, (50, 3), dtype=np.uint8)  
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

        for cls_id in range(50):
            color_mask[mask == cls_id] = palette[cls_id]
        
        return Image.fromarray(color_mask)
    
    def save(y, segments, images, labels, test_save_path, iter, ind):
        for channel in range(y.shape[0]):
            color_img = colorize_mask(y[channel])
            if not os.path.exists(f"{test_save_path}/img/"):
                os.makedirs(f"{test_save_path}/img/")
            color_img.save(f"{test_save_path}/img//output_{ind}_channel_{channel}.png")

    superpixel_block = 50
    root_path = "../../data/ACDC"
    snapshot_path = "../../model/diff10_seg1_30000_.pth"
    save_path = "../../model/test_result/diff10_seg1_30000_"
    diff_model = make_model_all(superpixel_block)

    db_val = BaseDataSets(base_dir=root_path, split="val")
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=4)
    diff_model.eval()

    ind = 0
    for _, sampled_batch in enumerate(valloader):
        y, segments = test_single_volume_diffusion(sampled_batch["image"], sampled_batch["label"], diff_model, superpixel_block, snapshot_path)
        save(y, segments, sampled_batch["image"], sampled_batch["label"], save_path, ind)
        ind += 1