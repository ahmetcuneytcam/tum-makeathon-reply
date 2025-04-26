#!/usr/bin/env python3
"""
Ray-based 3D triangulation using camera images, precise floating-point operations,

This script:
 1. Converts image pixels to normalized camera rays (optionally undistorted).
 2. Builds rotation matrices from roll, pitch, yaw angles with double precision.
 3. Applies gimbal and drone-body rotations to bring rays into the world frame.
 4. Converts GPS positions to/from ECEF coordinates using high-precision transforms.
 5. Triangulates two skew rays to estimate a 3D point.

Usage:
    python ray_triangulation.py
    # Customize inputs in `main` as needed.
"""

import numpy as np
import cv2
from pyproj import Transformer
from typing import Optional, Tuple

# Ensure full double-precision output
np.set_printoptions(precision=15, suppress=False)

# ——— Global Coordinate Transformers —————————————————————————————————
transformer_to_ecef = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
transformer_to_lla  = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)


def pixel_to_normalized_ray(
    u: float,
    v: float,
    K: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert pixel coordinates into a unit-length 3D ray in the camera frame (float64).

    Args:
        u (float): Pixel x-coordinate.
        v (float): Pixel y-coordinate.
        K (np.ndarray): 3×3 camera intrinsic matrix (float64).
        dist_coeffs (Optional[np.ndarray]): Lens distortion coefficients (float64).

    Returns:
        np.ndarray: Unit ray (shape: [3,]) in camera frame, dtype float64.
    """
    pixel = np.array([[u, v]], dtype=np.float64)
    if dist_coeffs is not None:
        undist = cv2.undistortPoints(
            pixel.reshape(-1, 1, 2), K, dist_coeffs
        ).reshape(2)
        x, y = float(undist[0]), float(undist[1])
    else:
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        x = (u - cx) / fx
        y = (v - cy) / fy

    ray = np.array([x, y, 1.0], dtype=np.float64)
    return ray / np.linalg.norm(ray)


def rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float
) -> np.ndarray:
    """
    Create a 3×3 rotation matrix from roll, pitch, yaw (in degrees) with float64.

    Args:
        roll (float): Rotation around X (deg).
        pitch (float): Rotation around Y (deg).
        yaw (float): Rotation around Z (deg).

    Returns:
        np.ndarray: 3×3 rotation matrix (float64).
    """
    r, p, y = np.deg2rad([roll, pitch, yaw]).astype(np.float64)

    Rx = np.array([
        [1.0,      0.0,       0.0],
        [0.0,  np.cos(r), -np.sin(r)],
        [0.0,  np.sin(r),  np.cos(r)],
    ], dtype=np.float64)
    Ry = np.array([
        [ np.cos(p), 0.0, np.sin(p)],
        [       0.0, 1.0,       0.0],
        [-np.sin(p), 0.0, np.cos(p)],
    ], dtype=np.float64)
    Rz = np.array([
        [np.cos(y), -np.sin(y), 0.0],
        [np.sin(y),  np.cos(y), 0.0],
        [      0.0,        0.0, 1.0],
    ], dtype=np.float64)

    return Rz @ Ry @ Rx


def apply_orientation(
    ray_cam: np.ndarray,
    drone_angles: Tuple[float, float, float],
    gimbal_angles: Tuple[float, float, float]
) -> np.ndarray:
    """
    Rotate a camera-frame ray into world coordinates by combining gimbal and drone orientations.

    Args:
        ray_cam (np.ndarray): Unit ray in camera frame ([3,], float64).
        drone_angles (tuple): (roll, pitch, yaw) in degrees of drone body.
        gimbal_angles (tuple): (roll, pitch, yaw) in degrees of gimbal.

    Returns:
        np.ndarray: Unit ray in world frame ([3,], float64).
    """
    R_drone  = rotation_matrix(*drone_angles)
    R_gimbal = rotation_matrix(*gimbal_angles)

    # Apply gimbal then drone
    ray_world = R_drone @ (R_gimbal @ ray_cam)
    return ray_world / np.linalg.norm(ray_world)


def gps_to_ecef(
    lat: float,
    lon: float,
    alt: float
) -> np.ndarray:
    """
    Convert geodetic to ECEF using high-precision.
    """
    x, y, z = transformer_to_ecef.transform(lon, lat, alt)
    return np.array([x, y, z], dtype=np.float64)


def ecef_to_gps(
    x: float,
    y: float,
    z: float
) -> Tuple[float, float, float]:
    """
    Convert ECEF back to (lat, lon, alt) with high precision.
    """
    lon, lat, alt = transformer_to_lla.transform(x, y, z)
    return float(lat), float(lon), float(alt)


def triangulate(
    c1: np.ndarray,
    d1: np.ndarray,
    c2: np.ndarray,
    d2: np.ndarray
) -> np.ndarray:
    """
    Compute midpoint of shortest line between two skew rays in ECEF.
    """
    A = np.column_stack((d1, -d2)).astype(np.float64)
    b = (c2 - c1).astype(np.float64)
    s, t = np.linalg.lstsq(A, b, rcond=None)[0]

    p1 = c1 + s * d1
    p2 = c2 + t * d2
    return (p1 + p2) / 2.0


def main():
    # Inputs — adjust as needed
    pixel1 = (1800.0, 350.0)
    pixel2 = (1500.0, 500.0)

    K = np.array([
        [2804.051000000000,       0.000000000000, 2010.410000000000],
        [      0.000000000000, 2804.051000000000, 1512.734000000000],
        [      0.000000000000,       0.000000000000,    1.000000000000]
    ], dtype=np.float64)
    dist_coeffs = np.array([
        0.116413456000000,
       -0.202624237000000,
        0.000004293000000,
       -0.000216595000000,
        0.136982457000000
    ], dtype=np.float64)

    drone1_angles = (-4.4, -10.9, 0.1)
    gimbal1_angles = (0.0, -90.0, -1.1)
    drone2_angles = (-5.5, -5.4, -0.5)
    gimbal2_angles = (0.0, -90.0, -1.0)

    drone1_pos = (49.099383853000000, 12.181017494000000, 21.692000000000)
    drone2_pos = (49.099412333000000, 12.181017859000000, 21.524000000000)

    # Ray construction and transform
    ray1_cam = pixel_to_normalized_ray(*pixel1, K, dist_coeffs)
    ray2_cam = pixel_to_normalized_ray(*pixel2, K, dist_coeffs)

    ray1_world = apply_orientation(ray1_cam, drone1_angles, gimbal1_angles)
    ray2_world = apply_orientation(ray2_cam, drone2_angles, gimbal2_angles)

    # ECEF positions
    c1 = gps_to_ecef(*drone1_pos)
    c2 = gps_to_ecef(*drone2_pos)

    # Triangulate
    point_ecef = triangulate(c1, ray1_world, c2, ray2_world)
    lat, lon, alt = ecef_to_gps(*point_ecef)

    # Print with maximum precision
    print(f"Estimated object location:\n"
          f"  lat = {lat:.12f}\n"
          f"  lon = {lon:.12f}\n"
          f"  alt = {alt:.6f} m")

if __name__ == "__main__":
    main()
