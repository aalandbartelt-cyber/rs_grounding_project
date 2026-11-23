"""Continuous image-edge coordinates; affine transforms for safe bbox augmentation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Affine:
    a: float = 1
    b: float = 0
    c: float = 0
    d: float = 0
    e: float = 1
    f: float = 0

    def point(self, x, y):
        return self.a * x + self.b * y + self.c, self.d * x + self.e * y + self.f

    def box(self, xywh):
        x, y, w, h = xywh
        points = [self.point(u, v) for u in (x, x + w) for v in (y, y + h)]
        xs, ys = zip(*points)
        return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

    def inverse(self):
        det = self.a * self.e - self.b * self.d
        if abs(det) < 1e-12:
            raise ValueError("Singular affine mapping")
        return Affine(self.e / det, -self.b / det, (self.b * self.f - self.e * self.c) / det,
                      -self.d / det, self.a / det, (self.d * self.c - self.a * self.f) / det)

    def values(self):
        return [self.a, self.b, self.c, self.d, self.e, self.f]


def transform(operation, width, height, crop=None):
    if operation == "flip_x":
        return Affine(-1, 0, width, 0, 1, 0), width, height
    if operation == "rotate90_ccw":
        return Affine(0, 1, 0, -1, 0, width), height, width
    if operation == "crop_keep_all":
        if crop is None:
            raise ValueError("Crop window is required")
        x, y, w, h = crop
        return Affine(1, 0, -x, 0, 1, -y), w, h
    if operation in ("original", "brightness_095", "resample_blur_085", "jpeg_80"):
        return Affine(), width, height
    raise ValueError(f"Unknown operation: {operation}")


def geometry_is_joint_safe(operation):
    # Flags cannot establish semantic invariance, so geometric transforms never auto-pass.
    return operation not in ("flip_x", "rotate90_ccw", "crop_keep_all")
