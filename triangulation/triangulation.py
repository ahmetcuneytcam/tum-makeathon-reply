import numpy as np
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
from pyproj import Transformer
from scipy.optimize import least_squares

# High-precision transforms
to_ecef = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
to_lla = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)

# Camera intrinsics (adjust to your camera)
K = np.array([
    [3049.296, 0.0,      2016.0],
    [0.0,      3049.296, 1512.0],
    [0.0,      0.0,      1.0   ]
], dtype=np.float64)
# Distortion coefficients [k1, k2, p1, p2, k3]
dist_coeffs = np.array([0.092649264, -0.168504154, 0.126455254, -0.000198582, -0.000211353], dtype=np.float64)

# -----------------------------------------------------------------------------
# Metadata extraction
def extract_metadata(img_path: Path) -> dict:
    data = img_path.read_text(encoding="latin-1")
    # Locate XMP block robustly
    start_idx = data.find('<x:xmpmeta')
    end_idx = data.find('</x:xmpmeta>', start_idx)
    if start_idx == -1 or end_idx == -1:
        raise RuntimeError(f"XMP metadata block not found in {img_path}")
    xmp_block = data[start_idx:end_idx + len('</x:xmpmeta>')]
    root = ET.fromstring(xmp_block)
    desc = root.find('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description')
    if desc is None:
        raise RuntimeError(f"No XMP Description element in {img_path}")

    tags = {
        'lat':    '{http://www.uav.com/drone-dji/1.0/}GpsLatitude',
        'lon':    '{http://www.uav.com/drone-dji/1.0/}GpsLongitude',
        'alt':    '{http://www.uav.com/drone-dji/1.0/}AbsoluteAltitude',
        'f_roll': '{http://www.uav.com/drone-dji/1.0/}FlightRollDegree',
        'f_pitch':'{http://www.uav.com/drone-dji/1.0/}FlightPitchDegree',
        'f_yaw':  '{http://www.uav.com/drone-dji/1.0/}FlightYawDegree',
        'g_roll': '{http://www.uav.com/drone-dji/1.0/}GimbalRollDegree',
        'g_pitch':'{http://www.uav.com/drone-dji/1.0/}GimbalPitchDegree',
        'g_yaw':  '{http://www.uav.com/drone-dji/1.0/}GimbalYawDegree'
    }
    meta = {}
    for key, tag in tags.items():
        val = desc.attrib.get(tag)
        if val is None:
            raise KeyError(f"Missing metadata tag {tag} in {img_path}")
        meta[key] = float(val)
    return meta

# -----------------------------------------------------------------------------
# Ray and rotation utils
def pixel_ray(u, v, K, dist=None):
    pt = np.array([[u, v]], dtype=np.float64)
    if dist is not None:
        uvn = cv2.undistortPoints(pt.reshape(-1,1,2), K, dist).reshape(2)
        x, y = uvn
    else:
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        x = (u - cx) / fx
        y = (v - cy) / fy
    ray = np.array([x, y, 1.0], dtype=np.float64)
    return ray / np.linalg.norm(ray)


def rot_matrix(roll, pitch, yaw):
    r, p, y = np.deg2rad([roll, pitch, yaw])
    Rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    Rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
    return Rz @ Ry @ Rx


def world_ray(meta, ray_cam):
    Rf = rot_matrix(meta['f_roll'], meta['f_pitch'], meta['f_yaw'])
    Rg = rot_matrix(meta['g_roll'], meta['g_pitch'], meta['g_yaw'])
    ray = Rf @ (Rg @ ray_cam)
    return ray / np.linalg.norm(ray)

# -----------------------------------------------------------------------------
# Coordinate transforms
def llh_to_ecef(lat, lon, alt):
    x, y, z = to_ecef.transform(lon, lat, alt)
    return np.array([x, y, z], dtype=np.float64)

def ecef_to_llh(x, y, z):
    lon, lat, alt = to_lla.transform(x, y, z)
    return lat, lon, alt

# -----------------------------------------------------------------------------
# Precise triangulation using non-linear optimization
def triangulate_precise(c1, d1, c2, d2):
    def residuals(p):
        p = np.array(p)
        r1 = np.cross(d1, p - c1)
        r2 = np.cross(d2, p - c2)
        return np.hstack((r1, r2))

    # initial guess: midpoint of shortest segment
    A = np.column_stack((d1, -d2))
    s, t = np.linalg.lstsq(A, (c2 - c1), rcond=None)[0]
    p0 = (c1 + s*d1 + c2 + t*d2) / 2

    sol = least_squares(residuals, p0, method='lm')
    return sol.x

def triangulate_from_images(img1, img2, px1=None, px2=None):
    m1 = extract_metadata(Path(img1))
    m2 = extract_metadata(Path(img2))

    # default to principal point
    cx, cy = K[0,2], K[1,2]
    u1, v1 = px1 if px1 else (cx, cy)
    u2, v2 = px2 if px2 else (cx, cy)

    r1_cam = pixel_ray(u1, v1, K, dist_coeffs)
    r2_cam = pixel_ray(u2, v2, K, dist_coeffs)

    d1 = world_ray(m1, r1_cam)
    d2 = world_ray(m2, r2_cam)
    c1 = llh_to_ecef(m1['lat'], m1['lon'], m1['alt'])
    c2 = llh_to_ecef(m2['lat'], m2['lon'], m2['alt'])

    p_ecef = triangulate_precise(c1, d1, c2, d2)
    lat, lon, alt = ecef_to_llh(*p_ecef)
    return lat, lon, alt

lat, lon, alt = triangulate_from_images(
    './Images_with_barcodes/DJI_20250424192953_0006_V.jpeg',
    './Images_with_barcodes/DJI_20250424193037_0036_V.jpeg',
    px1=(1485, 205),
    px2=(3000, 207)
 )

print(f"Latitude: {lat:.8f}\nLongitude: {lon:.8f}\nAltitude: {alt:.3f} m")
