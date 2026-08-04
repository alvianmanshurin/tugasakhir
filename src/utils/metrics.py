"""Metrics calculation utilities for Vehicle Detection evaluation"""

import numpy as np
from collections import defaultdict


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def calculate_precision_recall(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculate precision and recall for object detection.

    Args:
        predictions: List of dicts with 'bbox', 'class_id', 'confidence'
        ground_truths: List of dicts with 'bbox', 'class_id'
        iou_threshold: IoU threshold for matching

    Returns:
        precision, recall, true_positives, false_positives, false_negatives
    """
    # Sort predictions by confidence
    predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)

    tp = 0
    fp = 0
    fn = len(ground_truths)
    matched_gt = set()

    for pred in predictions:
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            if pred["class_id"] != gt["class_id"]:
                continue

            iou = calculate_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            fn -= 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def calculate_f1_score(precision, recall):
    """Calculate F1 score."""
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def calculate_map(predictions_list, ground_truths_list, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP).

    Args:
        predictions_list: List of prediction sets per image
        ground_truths_list: List of ground truth sets per image
        iou_threshold: IoU threshold

    Returns:
        mAP value
    """
    aps = []

    # Group by class
    class_ids = set()
    for gts in ground_truths_list:
        for gt in gts:
            class_ids.add(gt["class_id"])

    for class_id in class_ids:
        # Get predictions and GTs for this class
        class_preds = []
        class_gts = []

        for preds, gts in zip(predictions_list, ground_truths_list):
            class_preds.extend(
                [p for p in preds if p["class_id"] == class_id]
            )
            class_gts.extend(
                [g for g in gts if g["class_id"] == class_id]
            )

        if not class_gts:
            continue

        # Sort by confidence
        class_preds = sorted(
            class_preds, key=lambda x: x["confidence"], reverse=True
        )

        # Calculate precision-recall curve
        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []
        matched = set()

        for pred in class_preds:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(class_gts):
                if gt_idx in matched:
                    continue
                iou = calculate_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp_cumsum += 1
                matched.add(best_gt_idx)
            else:
                fp_cumsum += 1

            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / len(class_gts)

            precisions.append(precision)
            recalls.append(recall)

        # Calculate AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            prec_at_recall = [
                p for p, r in zip(precisions, recalls) if r >= t
            ]
            if prec_at_recall:
                ap += max(prec_at_recall)
        ap /= 11

        aps.append(ap)

    return np.mean(aps) if aps else 0


def calculate_fps(inference_times):
    """Calculate FPS from inference times."""
    if not inference_times:
        return 0
    avg_time = np.mean(inference_times)
    return 1.0 / avg_time if avg_time > 0 else 0


def generate_evaluation_report(metrics_dict):
    """Generate a formatted evaluation report."""
    report = []
    report.append("=" * 60)
    report.append("EVALUATION REPORT - Vehicle Detection System")
    report.append("=" * 60)
    report.append("")

    for metric_name, value in metrics_dict.items():
        if isinstance(value, float):
            report.append(f"  {metric_name}: {value:.4f}")
        else:
            report.append(f"  {metric_name}: {value}")

    report.append("")
    report.append("=" * 60)

    return "\n".join(report)
