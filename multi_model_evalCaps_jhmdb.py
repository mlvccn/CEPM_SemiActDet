import os
import csv
import glob
import torch
import argparse
import numpy as np
import os.path as osp

from pathlib import Path
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader
from utils.commons import init_seeds
from models.capsules_jhmdb_semi_sup_pa import CapsNet
import re

def get_specific_format_folders(save_path):
    pattern = re.compile(r"^\d{2}-\d{2}-\d{2}-\d{2}$")
    folder_names = []
    for item in os.listdir(save_path):
        item_path = os.path.join(save_path, item)
        if os.path.isdir(item_path) and pattern.match(item):
            folder_names.append(item)
    return folder_names

def iou():
    """
    Calculates the accuracy, f-mAP, and v-mAP over the test set for each exp_time folder
    """
    from datasets.jhmdb_dataloader_eval import JHMDB21Dataset
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--ckpt', type=str, help='experiment name')
    parser.add_argument('--seed', type=int, default=47, help='seed for initializing training.')
    parser.add_argument('--jhmdb_dataset_path', type=str, default='path/to/JHMDB/data/Videos/ReCompress_Videos', help='jhmdb_dataset_path')
    parser.add_argument('--jhmdb_exp_save_path', type=str, default=None, help='jhmdb_exp_save_path')  
    args = parser.parse_args()

    init_seeds(args.seed)
    save_path = osp.join(args.jhmdb_exp_save_path, args.ckpt)

    folders = get_specific_format_folders(save_path)

    for exp_time in folders:
        print(f"\n>>> Processing exp_time: {exp_time}")

        eval_scores = osp.join(save_path, "scores", exp_time)
        Path(eval_scores).mkdir(parents=True, exist_ok=True)

        v_map_thresh = open(osp.join(eval_scores, "v_map_thresh.csv"), "w+")
        f_map_thresh = open(osp.join(eval_scores, "f_map_thresh.csv"), "w+")
        v_map_class = open(osp.join(eval_scores, "v_map_classes.csv"), "w+")
        f_map_class = open(osp.join(eval_scores, "f_map_classes.csv"), "w+")

        iou_threshs = np.arange(0, 20, dtype=np.float32)/20
        num_classes = np.arange(21)

        csv.writer(v_map_thresh).writerow(["epoch_num"] + [str(i) for i in iou_threshs])
        csv.writer(f_map_thresh).writerow(["epoch_num"] + [str(i) for i in iou_threshs])
        csv.writer(v_map_class).writerow(["epoch_num"] + [str(i) for i in num_classes])
        csv.writer(f_map_class).writerow(["epoch_num"] + [str(i) for i in num_classes])

        model = CapsNet().cuda()
        clip_batch_size = 14
        n_classes = 21

        model_names = []
        fmap_best = []
        vmap_best = []

        # model_paths = sorted(glob.glob(osp.join(save_path, exp_time, 'best_model_train_' + '*.pth')))
        model_paths = sorted(
            glob.glob(osp.join(save_path, exp_time, 'best_model_train_*.pth')),
            key=lambda x: float(re.search(r'loss_(\d+(?:\.\d+)?)', osp.basename(x)).group(1)),
            reverse=True  # Sort in descending epoch order.
        )
        for saved_wts in model_paths:
            model.load_previous_weights(saved_wts)
            model.eval()
            model.training = False
            model_names.append(saved_wts)

            with torch.no_grad():
                validationset = JHMDB21Dataset('test', [224, 224], 8,file_id="testlist.txt",jhmdb_dataset_path=args.jhmdb_dataset_path)
                val_data_loader = DataLoader(validationset, batch_size=1, num_workers=0, shuffle=False)

                n_correct = 0
                n_vids = np.zeros((n_classes, 1))
                n_tot_frames = np.zeros((n_classes, 1))
                frame_ious = np.zeros((n_classes, 20))
                video_ious = np.zeros((n_classes, 20))
                for sample in tqdm(val_data_loader, total=len(val_data_loader), desc="testing..."):
                    video, bbox, label = sample
                    video, bbox, label = video[0], bbox[0], label[0]

                    f_skip = 2
                    clips = []
                    n_frames = video.shape[0]
                    for i in range(0, n_frames, 8 * f_skip):
                        for j in range(f_skip):
                            b_vid, b_bbox = [], []
                            valid_mask = []
                            for k in range(8):
                                ind = i + j + k * f_skip
                                if ind >= n_frames:
                                    if n_frames == 0:
                                        b_vid.append(np.zeros((1, 224, 224, 3), dtype=np.float32))
                                        b_bbox.append(np.zeros((1, 224, 224, 1), dtype=np.float32))
                                    else:
                                        b_vid.append(video[n_frames - 1:n_frames])
                                        b_bbox.append(bbox[n_frames - 1:n_frames])
                                    valid_mask.append(False)
                                else:
                                    b_vid.append(video[ind:ind + 1])
                                    b_bbox.append(bbox[ind:ind + 1])
                                    valid_mask.append(True)
                            clip = (
                                np.concatenate(b_vid, axis=0),
                                np.concatenate(b_bbox, axis=0),
                                label,
                                np.array(valid_mask, dtype=bool),
                            )
                            if np.sum(clip[1][clip[3]]) != 0:
                                clips.append(clip)

                    if len(clips) == 0:
                        print('Video has no bounding boxes')
                        continue

                    batches = []
                    gt_segmentations = []
                    valid_masks = []
                    for i in range(0, len(clips), clip_batch_size):
                        x_batch, bb_batch, y_batch, valid_batch = [], [], [], []
                        for j in range(i, min(i + clip_batch_size, len(clips))):
                            x, bb, y, valid = clips[j]
                            x_batch.append(x)
                            bb_batch.append(bb)
                            y_batch.append(y)
                            valid_batch.append(valid)
                        batches.append((x_batch, bb_batch, y_batch))
                        gt_segmentations.append(np.stack(bb_batch))
                        valid_masks.append(np.stack(valid_batch))

                    gt_segmentations = np.concatenate(gt_segmentations, axis=0).reshape((-1, 224, 224, 1))
                    valid_masks = np.concatenate(valid_masks, axis=0).reshape((-1,))

                    segmentations, predictions, frames = [], [], []
                    for x_batch, bb_batch, y_batch in batches:
                        data = np.transpose(np.array(x_batch), [0, 4, 1, 2, 3])
                        data = torch.from_numpy(data).type(torch.cuda.FloatTensor)
                        empty_action = torch.ones((len(x_batch),1), dtype=torch.int64).cuda() * 500

                        segmentation, pred, _ = model(data, empty_action, empty_action, 0, 0)
                        segmentation = torch.sigmoid(segmentation).cpu().numpy()
                        segmentation = np.transpose(segmentation, [0, 2, 3, 4, 1])
                        save_clip = np.transpose(data.cpu().numpy(), [0, 2, 3, 4, 1])

                        segmentations.append(segmentation)
                        predictions.append(pred.cpu().numpy())
                        frames.append(save_clip)

                    predictions = np.concatenate(predictions, axis=0)
                    fin_pred = np.argmax(np.mean(predictions, axis=0))
                    if fin_pred == label:
                        n_correct += 1

                    pred_segmentations = np.concatenate(segmentations, axis=0).reshape((-1, 224, 224, 1))
                    pred_segmentations = (pred_segmentations >= 0.5).astype(np.int64)
                    seg_plus_gt = pred_segmentations + gt_segmentations

                    vid_inter = vid_union = 0
                    for i in range(gt_segmentations.shape[0]):
                        if not valid_masks[i]:
                            continue
                        frame_gt = gt_segmentations[i]
                        if np.sum(frame_gt) == 0:
                            continue
                        n_tot_frames[label] += 1
                        inter = np.count_nonzero(seg_plus_gt[i] == 2)
                        union = np.count_nonzero(seg_plus_gt[i])
                        vid_inter += inter
                        vid_union += union
                        iou = inter / union
                        for k in range(20):
                            if iou >= iou_threshs[k]:
                                frame_ious[label, k] += 1

                    n_vids[label] += 1
                    vid_iou = vid_inter / vid_union if vid_union > 0 else 0
                    for k in range(20):
                        if vid_iou >= iou_threshs[k]:
                            video_ious[label, k] += 1

                fAP = frame_ious / n_tot_frames
                fmAP = np.mean(fAP, axis=0)
                vAP = video_ious / n_vids
                vmAP = np.mean(vAP, axis=0)

                print(f'Accuracy: {n_correct / np.sum(n_vids):.3f}, fmap/vmap@0.5: {fmAP[10]:.3f}/{vmAP[10]:.3f}, fmap/vmap@avg: {np.mean(fmAP[10:]):.3f}/{np.mean(vmAP[10:]):.3f}')

                epoch_num = "epoch_" + saved_wts.split('.')[0].split("_")[-1]
                csv.writer(v_map_thresh).writerow([epoch_num] + ['{:.3f}'.format(i) for i in vmAP])
                csv.writer(f_map_thresh).writerow([epoch_num] + ['{:.3f}'.format(i) for i in fmAP])
                csv.writer(v_map_class).writerow([epoch_num] + ['{:.3f}'.format(i) for i in vAP[:, 10]])
                csv.writer(f_map_class).writerow([epoch_num] + ['{:.3f}'.format(i) for i in fAP[:, 10]])

                fmap_best.append(fmAP[10])
                vmap_best.append(vmAP[10])

        # best_fmap_model = model_names[fmap_best.index(max(fmap_best))]
        # best_vmap_model = model_names[vmap_best.index(max(vmap_best))]
        # print(f"Best fmap model: {best_fmap_model}")
        # print(f"Best vmap model: {best_vmap_model}")

        # Select the best checkpoint by f-mAP and v-mAP.
        best_fmap_idx = fmap_best.index(max(fmap_best))
        best_vmap_idx = vmap_best.index(max(vmap_best))
        
        # Retrieve the checkpoint paths and their corresponding scores.
        best_fmap_model = model_names[best_fmap_idx]
        best_vmap_model = model_names[best_vmap_idx]
        best_fmap_value = fmap_best[best_fmap_idx]
        best_vmap_value = vmap_best[best_vmap_idx]
        
        print(f"Best fmap model: {best_fmap_model} with fmap@0.5: {best_fmap_value:.3f}")
        print(f"Best vmap model: {best_vmap_model} with vmap@0.5: {best_vmap_value:.3f}")

iou()

