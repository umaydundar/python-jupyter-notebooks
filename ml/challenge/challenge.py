# Load important packages

import numpy as np

from util import load_dataset
from util import store_predictions

from util import*


######### Training dataset

# Load training dataset
images_train, scrib_train, gt_train, fnames_train, palette = load_dataset(
    "C:/Users/User/OneDrive/Desktop/python_notebooks/ml/Challenge/dataset/train/", "images", "scribbles", "ground_truth"
)

# Inference
# Create a numpy array of size num_train x 375 x 500, a stack of all the
# segmented images. 1 = foreground, 0 = background.

print (len((images_train)))
pred_train = np.stack(
   [find_segments(image, scribble)
     for image, scribble in zip(images_train, scrib_train)],
    axis=0
)

# Storing Predictions
store_predictions(
    pred_train, "C:/Users/User/OneDrive/Desktop/python_notebooks/ml/Challenge/dataset/train/", "predictions", fnames_train, palette
)
print("Predictions stored successfully.")

# Visualizing model performance
vis_index = np.random.randint(images_train.shape[0])
visualize(
    images_train[vis_index], scrib_train[vis_index],
    gt_train[vis_index], pred_train[vis_index]
)

######### Test dataset

# Load test dataset
images_test, scrib_test, fnames_test = load_dataset(
    "C:/Users/User/OneDrive/Desktop/python_notebooks/ml/Challenge/dataset/test2/", "images", "Scribbles"
)
print("Test dataset loaded successfully.")


# Inference
# Create a numpy array of size num_test x 375 x 500, a stack of all the 
# segmented images. 1 = foreground, 0 = background.
pred_test = np.stack(
    [find_segments(image, scribble)
     for image, scribble in zip(images_test, scrib_test)],
    axis=0
)

# Storing segmented images for test dataset.
store_predictions(
    pred_test, "C:/Users/User/OneDrive/Desktop/python_notebooks/ml/Challenge/dataset/test2/", "predictions", fnames_test, palette
)

print("Test predictions stored successfully.")
print(pred_test)

# Evaluate mIoU on training set
iou_bg_total = 0.0
iou_obj_total = 0.0
miou_total = 0.0

for i in range(len(pred_train)):
    iou_bg, iou_obj, miou = compute_iou_per_class(pred_train[i], gt_train[i])
    iou_bg_total += iou_bg
    iou_obj_total += iou_obj
    miou_total += miou
    print(f"Image {i+1}/{len(pred_train)} - IoU BG: {iou_bg:.4f}, FG: {iou_obj:.4f}, mIoU: {miou:.4f}")

N = len(pred_train)
print("\n=== Overall Performance on Training Set ===")
print(f"Mean IoU:        {miou_total / N:.4f}")
print(f"Background IoU:  {iou_bg_total / N:.4f}")
print(f"Object IoU:      {iou_obj_total / N:.4f}")
