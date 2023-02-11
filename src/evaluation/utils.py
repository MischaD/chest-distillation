from sklearn.metrics import jaccard_score
import numpy as np

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


def compute_metrics(bbox, binary_prediction):
    x, y, h, w = bbox
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

