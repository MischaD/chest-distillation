from sklearn.metrics import jaccard_score
import numpy as np
import torch
from log import logger
from einops import reduce, rearrange
import os
import cv2


def compute_prediction_from_binary_mask(binary_prediction):
    binary_prediction = binary_prediction.to(torch.bool).numpy()
    horizontal_indicies = np.where(np.any(binary_prediction, axis=0))[0]
    vertical_indicies = np.where(np.any(binary_prediction, axis=1))[0]
    x1, x2 = horizontal_indicies[[0, -1]]
    y1, y2 = vertical_indicies[[0, -1]]
    prediction = np.zeros_like(binary_prediction)
    prediction[y1:y2, x1:x2] = 1
    center_of_mass = [x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2]
    return prediction, center_of_mass, (x1, x2, y1, y2)


def compute_metrics(x, y, h, w, binary_prediction):
    ground_truth_bbox_img = torch.zeros_like(binary_prediction)
    ground_truth_bbox_img[x:(x + w), y: (y + h)] = 1

    prediction, center_of_mass_prediction, bbox_pred = compute_prediction_from_binary_mask(binary_prediction)

    iou = torch.tensor(jaccard_score(ground_truth_bbox_img.flatten(), prediction.flatten()))
    iou_rev = torch.tensor(jaccard_score(1 - ground_truth_bbox_img.flatten(), 1 - prediction.flatten()))

    center_of_mass = [x + w / 2,
                      y + h / 2]
    miou = (iou + iou_rev)/2

    distance = np.sqrt((center_of_mass[0] - center_of_mass_prediction[0])**2 +
                       (center_of_mass[1] - center_of_mass_prediction[1])**2
                      )
    return iou, miou, distance, bbox_pred


def get_latent_slice(batch, opt):
    ds_slice = []
    for slice_ in batch["slice"]:
        if slice_.start is None:
            ds_slice.append(slice(None, None, None))
        else:
            ds_slice.append(slice(slice_.start // opt.f, slice_.stop // opt.f, None))
    return tuple(ds_slice)

def apply_rect(img, x, y, h, w, color="red"):
    img = (img * 255).to(torch.uint8).numpy()
    img = rearrange(img, "c h w -> h w c")
    if color == "red":
        color = (255, 0, 0)
    elif color == "blue":
        color = (0, 0, 255)

    img = cv2.rectangle(img.copy(), [x, y], [x + h, y + w], color, 3)
    img = rearrange(img, "h w c -> c h w") / 255.
    return torch.tensor(img)


def add_viz_of_data_and_pred(images, batch, x_samples_ddim, opt):
    # append input
    x0_norm = torch.clamp((batch["x"] + 1.0) / 2.0, min=0.0, max=1.0).cpu()
    x0_norm = reduce(x0_norm, 'b c (h h2) (w w2) -> b c h w', 'mean', h2=opt.f, w2=opt.f)
    images.append(x0_norm)

    # append model output

    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()
    images.append(
        reduce(x_samples_ddim, 'b c (h h2) (w w2) -> b c h w', 'mean', h2=opt.f, w2=opt.f))

    # append gt mask
    images.append(
        reduce(batch["segmentation_x"], 'b c (h h2) (w w2) -> b c h w', 'max', h2=opt.f, w2=opt.f))


def contrast_to_noise_ratio(ground_truth_img, prelim_mask_large):
    gt_mask = ground_truth_img.flatten()
    pr_mask = prelim_mask_large.flatten()

    roi_values = pr_mask[gt_mask == 1.0]
    not_roi_values = pr_mask[gt_mask != 1.0]

    contrast = roi_values.mean() - not_roi_values.mean()
    noise = torch.sqrt(
        roi_values.var() + not_roi_values.var()
    )
    cnr = contrast / noise
    return cnr


def check_mask_exists(mask_dir, samples):
    for i in range(len(samples["rel_path"])):
        path = os.path.join(mask_dir, samples["rel_path"][i] + ".pt")
        if not os.path.exists(path):
            return False
    return True



def get_latent_slice(batch, opt):
    ds_slice = []
    for slice_ in batch["slice"]:
        if slice_.start is None:
            ds_slice.append(slice(None, None, None))
        else:
            ds_slice.append(slice(slice_.start // opt.f, slice_.stop // opt.f, None))
    return tuple(ds_slice)

def apply_rect(img, x, y, h, w, color="red"):
    img = (img * 255).to(torch.uint8).numpy()
    img = rearrange(img, "c h w -> h w c")
    if color == "red":
        color = (255, 0, 0)
    elif color == "blue":
        color = (0, 0, 255)

    img = cv2.rectangle(img.copy(), [x, y], [x + h, y + w], color, 3)
    img = rearrange(img, "h w c -> c h w") / 255.
    return torch.tensor(img)


def add_viz_of_data_and_pred(images, batch, x_samples_ddim, opt):
    # append input
    x0_norm = torch.clamp((batch["x"] + 1.0) / 2.0, min=0.0, max=1.0).cpu()
    x0_norm = reduce(x0_norm, 'b c (h h2) (w w2) -> b c h w', 'mean', h2=opt.f, w2=opt.f)
    images.append(x0_norm)

    # append model output

    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()
    images.append(
        reduce(x_samples_ddim, 'b c (h h2) (w w2) -> b c h w', 'mean', h2=opt.f, w2=opt.f))

    # append gt mask
    images.append(
        reduce(batch["segmentation_x"], 'b c (h h2) (w w2) -> b c h w', 'max', h2=opt.f, w2=opt.f))


def contrast_to_noise_ratio(ground_truth_img, prelim_mask_large):
    gt_mask = ground_truth_img.flatten()
    pr_mask = prelim_mask_large.flatten()

    roi_values = pr_mask[gt_mask == 1.0]
    not_roi_values = pr_mask[gt_mask != 1.0]

    contrast = roi_values.mean() - not_roi_values.mean()
    noise = torch.sqrt(
        roi_values.var() + not_roi_values.var()
    )
    cnr = contrast / noise
    return cnr


def check_mask_exists(mask_dir, samples):
    for i in range(len(samples["rel_path"])):
        path = os.path.join(mask_dir, samples["rel_path"][i] + ".pt")
        if not os.path.exists(path):
            return False
    return True


def check_mask_exists_multiquery(mask_dir, samples):
    for i in range(len(samples["rel_path"])):
        query_classes = samples["query_classes"][i]
        for query_class in query_classes:
            path = samples_to_path_multiquery(mask_dir, samples, i, query=query_class)
            if not os.path.exists(path):
                return False
    return True


def samples_to_path(mask_dir, samples, j):
    sample_path = samples["rel_path"][j]
    label = samples["finding_labels"][j]
    impr = samples["impression"][j].replace(" ", "_")[:100]
    path = os.path.join(mask_dir, sample_path + label + impr) + ".pt"
    logger.info(f"StoPath: {path}")
    return path


def samples_to_path_multiquery(mask_dir, samples, j, query=""):
    sample_path = samples["rel_path"][j]
    captions = samples["captions"][j].replace(" ", "_")[:228]
    path = os.path.join(mask_dir, sample_path + captions + query) + ".pt"
    #logger.info(f"StoPath: {path}")
    return path
