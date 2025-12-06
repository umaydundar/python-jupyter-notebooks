import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from typing import Any
import numpy as np
import cv2
import random
from skimage.measure import regionprops
from collections import defaultdict


######### Methods for loading dataset

def _open_image(path, convert_to):
    if convert_to == "RGB":
        return Image.open(path).convert("RGB")
    if convert_to == "grayscale":
        return Image.open(path).convert("L")
    return np.array(Image.open(path))

def _get_file_names(folder):
    return sorted(
        [file for file in os.listdir(folder) if not file.startswith('.')]
    )

def _load_images(folder_path, folder_name, convert_to):
    image_dir_path = os.path.join(folder_path, folder_name)
    filenames = _get_file_names(image_dir_path)
    filepaths = [
        os.path.join(image_dir_path, filename) for filename in filenames
    ]
    return np.stack([_open_image(file, convert_to) for file in filepaths])

def _get_palette(folder_path, ground_truth_dir, filename):
    gt_dir_path = os.path.join(folder_path, ground_truth_dir)
    filepath = os.path.join(gt_dir_path, filename)
    return Image.open(filepath).getpalette()

def _get_filenames(folder_path, scribbles_dir):
    sc_dir_path = os.path.join(folder_path, scribbles_dir)
    filenames = _get_file_names(sc_dir_path)
    return filenames

def load_dataset(
    folder_path: str,
    images_dir: str,
    scribbles_dir: str,
    ground_truth_dir: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:

    images = _load_images(folder_path, images_dir, "RGB")
    scribbles = _load_images(folder_path, scribbles_dir, "grayscale")
    filenames = _get_filenames(folder_path, scribbles_dir)
    if ground_truth_dir is None:
        return images, scribbles, filenames
    
    ground_truth = _load_images(folder_path, ground_truth_dir, None)
    palette = _get_palette(folder_path, ground_truth_dir, filenames[0])
    return images, scribbles, ground_truth, filenames, palette

def store_predictions(
    predictions: np.ndarray,
    folder_path: str,
    predictions_dir: str,
    filenames: list[str],
    palette: Any
):
    
    pred_dir_path = os.path.join(folder_path, predictions_dir)
    if not os.path.exists(pred_dir_path):
        os.makedirs(pred_dir_path)
    for filename, pred_array in zip(filenames, predictions):
        filepath = os.path.join(pred_dir_path, filename)
        pred_image = Image.fromarray(pred_array.astype(np.uint8), mode='P')
        pred_image.putpalette(palette)
        pred_image.save(filepath)


# Methods for baseline model

def baseline(
    image: np.ndarray,
    scribble: np.ndarray,
    k: int = 3
) -> np.ndarray:
    
    H, W, C = image.shape
    assert C == 3, "Image must be RGB."

    # Reshape image to (H*W, 3)
    image_flat = image.reshape(-1, 3)

    # Flatten scribble mask
    scribbles_flat = scribble.flatten()

    # Create mask for labeled and unlabeled pixels
    labeled_mask = (scribbles_flat != 255)
    unlabeled_mask = (scribbles_flat == 255)

    # Prepare training data
    X_train = image_flat[labeled_mask]
    y_train = scribbles_flat[labeled_mask]

    # Prepare test data
    X_test = image_flat[unlabeled_mask]

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict on unlabeled pixels
    y_pred = knn.predict(X_test)

    # Reconstruct full prediction mask
    predicted_mask = np.zeros_like(scribbles_flat)
    predicted_mask[labeled_mask] = y_train
    predicted_mask[unlabeled_mask] = y_pred

    # Reshape to (H, W)
    return predicted_mask.reshape(H, W)


######### Methods for visualization

def _overlay_scribbles(
    image, scribble, color_fg=(255, 0, 0), color_bg=(0, 0, 255), alpha=0.6
):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be RGB")
    if scribble.shape != image.shape[:2]:
        raise ValueError("Scribble must match image spatial size")
    
    overlaid = image.copy().astype(np.float32)
    
    mask_fg = scribble == 1
    mask_bg = scribble == 0
    
    for mask, color in [(mask_fg, color_fg), (mask_bg, color_bg)]:
        for c in range(3):
            overlaid[..., c][mask] = (
                alpha * color[c] + (1 - alpha) * overlaid[..., c][mask]
            )
    
    return overlaid.astype(np.uint8)

def visualize(
    image: np.ndarray,
    scribbles: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    alpha: float=0.6
):
    
    image_with_scribbles = _overlay_scribbles(image, scribbles, alpha=alpha)
    cmap = plt.get_cmap('bwr')
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_with_scribbles)
    axes[0].set_title("Image + Scribbles")
    axes[1].imshow(ground_truth, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(prediction, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title("Model Prediction")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


#-----------------------------------------

 
def compute_iou_per_class(prediction: np.ndarray, ground_truth: np.ndarray) -> tuple:
    assert prediction.shape == ground_truth.shape, "Shape mismatch"
    
    ious = []
    for cls in [0, 1]:
        pred_cls = (prediction == cls)
        gt_cls = (ground_truth == cls)

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()

        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union

        ious.append(iou)

    iou_background, iou_object = ious
    miou = np.mean(ious)
    return iou_background, iou_object, miou


GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD = 0, 1, 2, 3

def find_segments(image, scribble, iters=5):
    height, width, _ = image.shape
    
    # Create initial regions and merge
    num_initial_points = 100
    initial_regions_map = np.zeros((height, width), dtype=np.int32)
    next_region_id = 1
    
    points_to_seed = [(random.randint(0, height - 1), random.randint(0, width - 1)) for _ in range(num_initial_points)]
    
    for y_seed, x_seed in points_to_seed:
        if initial_regions_map[y_seed, x_seed] == 0:
            region_id = next_region_id
            next_region_id += 1
            
            queue = [(y_seed, x_seed)]
            initial_regions_map[y_seed, x_seed] = region_id
            seed_color = image[y_seed, x_seed]
            
            while queue:
                y, x = queue.pop(0)
                
                for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ny, nx = y + dy, x + dx
                    
                    if 0 <= ny < height and 0 <= nx < width and initial_regions_map[ny, nx] == 0:
                        neighbor_color = image[ny, nx]
                        color_diff = np.linalg.norm(neighbor_color.astype(float) - seed_color.astype(float))
                        if color_diff < 20:
                            initial_regions_map[ny, nx] = region_id
                            queue.append((ny, nx))
                            
    merged_regions_map = {i: i for i in np.unique(initial_regions_map) if i != 0}
    
    for _ in range(10): 
        merged_in_iter = False
        regions = regionprops(initial_regions_map)
        region_props_dict = {r.label: r for r in regions}
        
        for region in regions:
            current_id = merged_regions_map[region.label]
            mask = initial_regions_map == region.label
            adjacent_ids = np.unique(initial_regions_map[np.where(mask)])
            adjacent_ids = [merged_regions_map[adj_id] for adj_id in adjacent_ids if adj_id != 0 and merged_regions_map[adj_id] != current_id]

            for adj_id in adjacent_ids:
                if current_id != adj_id:
                    adj_region = next((r for r in regions if merged_regions_map[r.label] == adj_id), None)
                    if not adj_region: continue
                    
                    current_avg_color = region_props_dict[current_id].intensity_image.mean(axis=0)
                    adj_avg_color = region_props_dict[adj_id].intensity_image.mean(axis=0)
                    color_diff = np.linalg.norm(current_avg_color.astype(float) - adj_avg_color.astype(float))
                    
                    if color_diff < 30:
                        for r_id, r_label in merged_regions_map.items():
                            if r_label == adj_id:
                                merged_regions_map[r_id] = current_id
                        merged_in_iter = True
        if not merged_in_iter:
            break
            
    final_merged_image = np.zeros_like(initial_regions_map)
    for y in range(height):
        for x in range(width):
            original_id = initial_regions_map[y, x]
            if original_id in merged_regions_map:
                final_merged_image[y, x] = merged_regions_map[original_id]
                
    # Flood-fill refinement 
    temp_mask = np.zeros((height + 2, width + 2), np.uint8)
    temp_mask[1:-1, 1:-1] = np.where(final_merged_image > 0, 1, 0)
    
    flood_mask = scribble.copy()
    
    fg_seeds = np.argwhere(scribble == 1)
    bg_seeds = np.argwhere(scribble == 2)
    
    for y, x in fg_seeds:
        cv2.floodFill(flood_mask, temp_mask, (x, y), 1, loDiff=0, upDiff=0, flags=cv2.FLOODFILL_MASK_ONLY)
    
    for y, x in bg_seeds:
        cv2.floodFill(flood_mask, temp_mask, (x, y), 2, loDiff=0, upDiff=0, flags=cv2.FLOODFILL_MASK_ONLY)

    # Map the flood-filled regions to GrabCut labels
    gc_mask = np.full((height, width), GC_PR_BGD, dtype=np.uint8)
    
    gc_mask[flood_mask == 1] = GC_FGD
    gc_mask[flood_mask == 2] = GC_BGD
    
    gc_mask[scribble == 1] = GC_FGD
    gc_mask[scribble == 2] = GC_BGD
            
    if not np.any(gc_mask == GC_FGD) or not np.any(gc_mask == GC_BGD):
        gc_mask = np.full((height, width), GC_PR_BGD, dtype=np.uint8)
        gc_mask[scribble == 1] = GC_FGD
        gc_mask[scribble == 2] = GC_BGD

    # Run GrabCut algo with the refined mask
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    cv2.grabCut(bgr, gc_mask, None, bgdModel, fgdModel, iters, mode=cv2.GC_INIT_WITH_MASK)
    mask = np.where((gc_mask == GC_FGD) | (gc_mask == GC_PR_FGD), 1, 0).astype(np.uint8)
    
    # Final refinement based on connectivity and scribble location
    num_labels, labels_map = cv2.connectedComponents(mask)
    
    # Find the label of the component that contains the foreground scribble
    scribble_fg_label = 0
    fg_y, fg_x = np.where(scribble == 1)
    
    if fg_y.size > 0:
        # Find the label that most of the scribble pixels belong to
        scribble_labels = labels_map[fg_y, fg_x]
        
        # Use a voting system to find the primary scribble label
        counts = np.bincount(scribble_labels)
        scribble_fg_label = np.argmax(counts[1:]) + 1 if counts.size > 1 else 0

    refined_final_mask = np.zeros_like(mask)
    
    # If a primary foreground component was found
    if scribble_fg_label != 0:
        # Calculate the average color of the primary object
        primary_object_mask = (labels_map == scribble_fg_label).astype(np.uint8)
        primary_object_colors = image[primary_object_mask == 1]
        
        if primary_object_colors.size > 0:
            primary_object_avg_color = np.mean(primary_object_colors, axis=0)
            
            for label in range(1, num_labels):
                component_mask = (labels_map == label).astype(np.uint8)
            
                if label == scribble_fg_label:
                    refined_final_mask[component_mask == 1] = 1
                    continue

                component_colors = image[component_mask == 1]
                if component_colors.size > 0:
                    component_avg_color = np.mean(component_colors, axis=0)
                    color_diff = np.linalg.norm(component_avg_color.astype(float) - primary_object_avg_color.astype(float))
                    
                    # Keep the component only if it's large enough and has a similar color
                    if color_diff < 20 and np.sum(component_mask) > 30:
                         refined_final_mask[component_mask == 1] = 1
    return refined_final_mask