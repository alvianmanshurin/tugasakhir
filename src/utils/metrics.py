"""
Modul metrik untuk evaluasi model
"""
import numpy as np
from collections import defaultdict

def calculate_iou(box1, box2):
    """Hitung Intersection over Union (IoU)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def calculate_precision_recall(predictions, ground_truth, iou_threshold=0.5):
    """Hitung precision dan recall"""
    tp = 0
    fp = 0
    fn = 0
    
    matched_gt = set()
    
    for pred in predictions:
        best_iou = 0
        best_gt_idx = -1
        
        for idx, gt in enumerate(ground_truth):
            if idx in matched_gt:
                continue
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    
    fn = len(ground_truth) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall

def calculate_ap(precisions, recalls):
    """Hitung Average Precision (AP)"""
    sorted_indices = np.argsort(recalls)
    sorted_recalls = np.array(recalls)[sorted_indices]
    sorted_precisions = np.array(precisions)[sorted_indices]
    
    ap = 0
    for i in range(1, len(sorted_recalls)):
        ap += (sorted_recalls[i] - sorted_recalls[i-1]) * sorted_precisions[i]
    
    return ap

def main():
    print("=== MODUL METRIK ===")
    # Tambahkan kode sesuai kebutuhan

if __name__ == "__main__":
    main()
