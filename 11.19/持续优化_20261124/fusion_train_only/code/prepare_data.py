"""Create a group split using original-training imagery only."""
import random

import cv2
import numpy as np

from common import ROOT, TrainArchive, write_json


def main():
    data = TrainArchive()
    sift = cv2.SIFT_create(nfeatures=400)
    features = []
    for label in data.labels:
        image = data.image(label)
        image.thumbnail((384, 384))
        keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY), None)
        points = np.array([point.pt for point in keypoints], dtype=np.float32)
        features.append((points, descriptors))
    parent = list(range(len(features)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = []
    for i, (points, descriptors) in enumerate(features):
        for j in range(i):
            other_points, other_descriptors = features[j]
            if descriptors is None or other_descriptors is None:
                continue
            matches = matcher.knnMatch(descriptors, other_descriptors, k=2)
            good = [a for a, b in matches if a.distance < 0.68 * b.distance]
            if len(good) < 12:
                continue
            source = points[[m.queryIdx for m in good]]
            target = other_points[[m.trainIdx for m in good]]
            matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=3)
            if matrix is not None and int(inliers.sum()) >= 12:
                parent[find(i)] = find(j)
                pairs.append({"first": data.labels[i], "second": data.labels[j],
                              "affine_inliers": int(inliers.sum())})
    groups = {}
    for i, label in enumerate(data.labels):
        groups.setdefault(find(i), []).append(label)
    values = list(groups.values())
    random.Random(20261123).shuffle(values)
    development = []
    for group in values:
        if len(development) < 15 and len(development) + len(group) <= 20:
            development.extend(group)
    fitting = [label for label in data.labels if label not in development]
    if len(fitting) < 15 or len(development) < 5:
        raise ValueError("Overlap groups cannot form a useful internal development gallery")
    result = {"seed": 20261123, "source_split": "original train only", "fit": fitting, "dev": development,
              "description": "One internal development round, SIFT-overlap groups kept together",
              "overlap_groups": list(groups.values()), "flagged_pairs": pairs,
              "official_validation_or_test_used": False}
    write_json(ROOT / "results/split.json", result)
    data.audit(ROOT / "results/split_data_access.json")
    data.close()
    print(f"Grouped training-only split: {len(fitting)} fit images, {len(development)} dev images, "
          f"{len(groups)} overlap groups, {len(pairs)} overlapping pairs", flush=True)


if __name__ == "__main__":
    main()
