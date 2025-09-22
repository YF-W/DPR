import argparse
import os
import shutil
import logging
import sys
import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
from networks.net_factory_2D import net_factory
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='DD-Net', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=7, help='labeled data')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--label_ratio', type=str, default='10%', help='GPU to use')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--cfg', type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml",help='path to config file', )
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+', )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'], help='no: no cache, ''full: cache all data, ''part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
args = parser.parse_args()

if args.model == "swinunet":
    args.patch_size = [224, 224]

label_num_mapping = {
    "ACDC": {'5%': 3, '10%': 7, '15%': 21, '20%': 14, '100%': 140},
    "Prostate": {'5%': 2, '10%': 4, '20%': 7, "25%": 9, '100%': 35},
    "Hippocampus": {'5%': 8, '10%': 16, '20%': 31, '100%': 156},
    "ATLAS":{'5%': 2, '10%': 4, '20%': 7, "25%": 9, '100%': 36},
    "BTCV":{'5%': 1, '10%': 2, '20%': 4, '100%': 18},
    "Vertebral":{'5%': 5, '10%': 10, '20%': 21, '100%':103},
    "HepaticVessel":{'5%': 9, '10%': 18, '20%': 36, '100%': 181},

}

if "ACDC" in args.root_path:
    args.labeled_num = label_num_mapping["ACDC"][args.label_ratio]
    args.num_classes = 4
elif "Prostate" in args.root_path:
    args.labeled_num = label_num_mapping["Prostate"][args.label_ratio]
    args.num_classes = 2
elif "Hippocampus" in args.root_path:
    args.labeled_num = label_num_mapping["Hippocampus"][args.label_ratio]
    args.num_classes = 3
elif "ATLAS" in args.root_path:
    args.labeled_num = label_num_mapping["ATLAS"][args.label_ratio]
    args.num_classes = 3
elif "BTCV" in args.root_path:
    args.labeled_num = label_num_mapping["BTCV"][args.label_ratio]
    args.num_classes = 14

elif "Vertebral" in args.root_path:
    args.labeled_num = label_num_mapping["Vertebral"][args.label_ratio]
    args.num_classes = 20
elif "HepaticVessel" in args.root_path:
    args.labeled_num = label_num_mapping["HepaticVessel"][args.label_ratio]
    args.num_classes = 3
else:
    print("Error")
    sys.exit(1)

def calculate_metric_percase(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    if gt.sum() == 0: 
        return None, None, None, None

    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        asd = metric.binary.asd(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, jc, hd95, asd
    else:
        if pred.sum() == 0 and gt.sum() == 0:
            return 1.0, 1.0, 0.0, 0.0
        else:
            return 0.0, 0.0, None, None

def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    patch_x, patch_y = FLAGS.patch_size
    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_x / x, patch_y / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_main = net(input)
            if len(out_main) > 1:
                out_main = out_main[0]
            else:
                out_main = out_main
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_x, y / patch_y), order=0)
            prediction[ind] = pred

    per_class_metrics = []
    for cls in range(1, FLAGS.num_classes): 
        if np.sum(label == cls) == 0:
            per_class_metrics.append((None, None, None, None))
        else:
            metrics = calculate_metric_percase(prediction == cls, label == cls)
            per_class_metrics.append(metrics)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))

    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")

    return per_class_metrics


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    snapshot_path = "../model/{}_{}_{}_{}_labeled".format(FLAGS.root_path[8:], FLAGS.model, FLAGS.exp, FLAGS.labeled_num)
    test_save_path = "../model/{}_{}_{}_{}_labeled/predictions/".format(FLAGS.root_path[8:],FLAGS.model, FLAGS.exp, FLAGS.labeled_num)

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    logging.basicConfig(filename=snapshot_path + "/detail.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes, args=FLAGS)
    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_model_path), strict=False)
    net = net.cuda()
    print("init weight from {}".format(save_model_path))
    net.eval()

    total_metric_per_class = np.zeros((FLAGS.num_classes - 1, 4), dtype=np.float64) 
    all_metrics = []

    for case in tqdm(image_list):
        case_metrics = test_single_volume(case, net, test_save_path, FLAGS)
        case_metrics = np.asarray(case_metrics, dtype=np.float64)  
        total_metric_per_class += case_metrics 

        print(f"\n{case} results:")
        row_metrics = [case]  
        for cls in range(1, FLAGS.num_classes):
            dice, jc, hd95, asd = case_metrics[cls - 1]
            print(f"Class {cls:2d}: Dice: {dice:.4f}, JC: {jc:.4f}, HD95: {hd95:.2f}, ASD: {asd:.2f}")
            row_metrics.extend([dice, jc, hd95, asd])  
        all_metrics.append(row_metrics)

    avg_metric_per_class = total_metric_per_class / len(image_list)

    header = ['Sample']  
    for cls in range(1, FLAGS.num_classes):
        header.extend([f'Class {cls} Dice', f'Class {cls} JC', f'Class {cls} HD95', f'Class {cls} ASD'])
    
    csv_file = os.path.join(test_save_path, 'metrics_per_sample.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header) 
        writer.writerows(all_metrics)

    return avg_metric_per_class, test_save_path




if __name__ == '__main__':
    metric, test_save_path = Inference(args)

    for cls in range(1, args.num_classes):
        print(f"Class {cls:2d} => Dice: {metric[cls - 1, 0]:.6f}, JC: {metric[cls - 1, 1]:.6f}, HD95: {metric[cls - 1, 2]:.6f}, ASD: {metric[cls - 1, 3]:.6f}")

    mean_avg = np.nanmean(metric, axis=0)
    print(f"\nOverall Average => Dice: {mean_avg[0]:.6f}, JC: {mean_avg[1]:.6f}, HD95: {mean_avg[2]:.6f}, ASD: {mean_avg[3]:.6f}")

    with open(test_save_path + '../performance.txt', 'w') as f:
        f.writelines("Per-class metrics:\n")
        for cls in range(1, args.num_classes):
            f.writelines(f"Class {cls:2d} => Dice: {metric[cls - 1, 0]:.6f}, JC: {metric[cls - 1, 1]:.6f}, HD95: {metric[cls - 1, 2]:.6f}, ASD: {metric[cls - 1, 3]:.6f}\n")
        f.writelines(f"\nOverall Average => Dice: {mean_avg[0]:.6f}, JC: {mean_avg[1]:.6f}, HD95: {mean_avg[2]:.6f}, ASD: {mean_avg[3]:.6f}\n")
