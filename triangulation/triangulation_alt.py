#!/usr/bin/env python3
"""
Compute planar ENU x,y and geographic coordinates (lat, lon) of an object from two DJI images.
Uses only gimbal pitch (absolute) and combined yaw (flight + gimbal), ignores roll.
"""

import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image


def extract_metadata_from_single_image(img_path):
    """
    Parse DJI XMP metadata from a JPEG and extract relevant parameters.
    Returns dict with keys:
      'GpsLatitude', 'GpsLongitude', 'AbsoluteAltitude', 'RelativeAltitude',
      'FlightYawDegree', 'GimbalPitchDegree', 'GimbalYawDegree'
    """
    with open(img_path, encoding="latin-1") as f:
        data = f.read()

    start = data.find('<x:xmpmeta')
    end = data.find('</x:xmpmeta>') + len('</x:xmpmeta>')
    xmp_str = data[start:end]

    root = ET.fromstring(xmp_str)
    desc = root.find('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description')
    attrs = desc.attrib

    # URIs of needed tags
    keys = {
        'GpsLatitude':       '{http://www.uav.com/drone-dji/1.0/}GpsLatitude',
        'GpsLongitude':      '{http://www.uav.com/drone-dji/1.0/}GpsLongitude',
        'AbsoluteAltitude':  '{http://www.uav.com/drone-dji/1.0/}AbsoluteAltitude',
        'RelativeAltitude':  '{http://www.uav.com/drone-dji/1.0/}RelativeAltitude',
        'FlightYawDegree':   '{http://www.uav.com/drone-dji/1.0/}FlightYawDegree',
        'GimbalPitchDegree':'{http://www.uav.com/drone-dji/1.0/}GimbalPitchDegree',
        'GimbalYawDegree':   '{http://www.uav.com/drone-dji/1.0/}GimbalYawDegree'
    }

    meta = {}
    for k, uri in keys.items():
        v = attrs.get(uri)
        if v is None:
            print(f"Missing metadata '{k}' in {img_path}")
            continue
        meta[k] = float(v)

    return meta


def pixel_to_angle(offset_px, img_size_px, fov_deg):
    """Convert pixel offset to FoV angle in degrees."""
    return (offset_px / img_size_px) * fov_deg


def camera_direction_vector(x_px, y_px, img_w, img_h, fov_deg):
    """
    Build a unit ray in camera coords through pixel (x_px,y_px).
    Principal point at (img_w/2, img_h/2), symmetric FoV.
    """
    dx = x_px - img_w/2
    dy = y_px - img_h/2
    ang_x = np.deg2rad(pixel_to_angle(dx, img_w, fov_deg))
    ang_y = np.deg2rad(pixel_to_angle(dy, img_h, fov_deg))
    dir_cam = np.array([np.tan(ang_x), np.tan(ang_y), 1.0])
    return dir_cam / np.linalg.norm(dir_cam)


def rotation_matrix_from_yaw_pitch(yaw_deg, pitch_deg):
    """
    Build rotation matrix R = R_z(yaw) @ R_y(pitch).
    Ignores roll.
    """
    y = np.deg2rad(yaw_deg)
    p = np.deg2rad(pitch_deg)
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y),  np.cos(y), 0],
                   [       0,         0,    1]])
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [        0,  1,        0],
                   [-np.sin(p), 0, np.cos(p)]])
    return Rz @ Ry


def triangulate_rays(p0_1, d1, p0_2, d2):
    """
    Compute midpoint of shortest segment between two skew rays.
    Returns midpoint in 3D.
    """
    w0 = p0_1 - p0_2
    a = np.dot(d1, d1); b = np.dot(d1, d2); c = np.dot(d2, d2)
    d = np.dot(d1, w0); e = np.dot(d2, w0)
    denom = a*c - b*b
    if abs(denom) < 1e-6:
        return None
    sc = (b*e - c*d) / denom
    tc = (a*e - b*d) / denom
    pt1 = p0_1 + sc*d1
    pt2 = p0_2 + tc*d2
    return (pt1 + pt2) / 2


def triangulate_from_images(img1_path, img2_path, px1, py1, px2, py2, fov_deg=73.7):
    # 1) Extract metadata
    meta1 = extract_metadata_from_single_image(img1_path)
    meta2 = extract_metadata_from_single_image(img2_path)
    lat1, lon1 = meta1['GpsLatitude'], meta1['GpsLongitude']
    lat2, lon2 = meta2['GpsLatitude'], meta2['GpsLongitude']

    """ lat1 = 48.18933856
    lon1 = 11.499883

    lat2 = 48.18933856
    lon2 = 11.500094444

    alt1 = 13.308
    alt2 = 13.891 """

    alt1 = meta1['RelativeAltitude'] or meta1['AbsoluteAltitude']
    alt2 = meta2['RelativeAltitude'] or meta2['AbsoluteAltitude']

    # 2) Image sizes
    w1, h1 = Image.open(img1_path).size
    w2, h2 = Image.open(img2_path).size

    # 3) Build direction vectors in camera frame
    d_cam1 = camera_direction_vector(px1, py1, w1, h1, fov_deg)
    d_cam2 = camera_direction_vector(px2, py2, w2, h2, fov_deg)

    # ✨ New: Only keep the X, Y part — we project onto the EN plane
    d2d_1 = d_cam1[:2] / np.linalg.norm(d_cam1[:2])
    d2d_2 = d_cam2[:2] / np.linalg.norm(d_cam2[:2])

    # 4) Define ray origins in EN plane
    def latlon_to_enu(lat, lon, lat0, lon0):
        dlat = lat - lat0
        dlon = lon - lon0
        north = dlat * 111319.5
        east  = dlon * 111319.5 * np.cos(np.deg2rad(lat0))
        return np.array([east, north])

    p0_1 = np.array([0.0, 0.0])  # (east, north)
    enu2 = latlon_to_enu(lat2, lon2, lat1, lon1)
    p0_2 = enu2  # (east, north)

    # 5) Triangulate in 2D (solve for intersection of 2 lines)
    # Line 1: p0_1 + t1 * d2d_1
    # Line 2: p0_2 + t2 * d2d_2
    A = np.array([d2d_1, -d2d_2]).T
    b = p0_2 - p0_1
    if np.linalg.matrix_rank(A) < 2:
        raise RuntimeError("Lines are parallel or nearly so.")

    t = np.linalg.solve(A, b)
    intersection = p0_1 + t[0] * d2d_1  # 2D intersection point

    # 6) Average altitude (simple way)
    avg_alt = (alt1 + alt2) / 2

    # 7) Convert back to lat/lon
    east, north = intersection
    lat_obj = lat1 + north / 111319.5
    lon_obj = lon1 + east / (111319.5 * np.cos(np.deg2rad(lat1)))

    return lat_obj, lon_obj

# Example usage:
if __name__ == '__main__':
    ex = triangulate_from_images(
        './Images_with_barcodes/DJI_20250424192953_0006_V.jpeg',
        './Images_with_barcodes/DJI_20250424193037_0036_V.jpeg',
        1850, 215, 3000, 2000
    )
    print(f"lat={ex[0]:.8f}, lon={ex[1]:.8f}")

    """ ex = triangulate_from_images(
        './some_test_images/DJI_20250424162345_0123_V.jpeg',
        './some_test_images/DJI_20250424162350_0128_V.jpeg',
        1772, 246, 1621, 234
    )

    print(f"lat={ex[0]:.8f}, lon={ex[1]:.8f}") """
