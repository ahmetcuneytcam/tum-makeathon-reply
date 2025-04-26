#!/usr/bin/env python3
"""
add_waypoints.py

Automates adding new <Placemark> entries to DJI waypoint mission files
(template.kml or waylines.wpml) by copying the last Placemark, incrementing
its index, setting new coordinates, and updating heights.

Usage:
    python add_waypoints.py [--ground-elevation GROUND_ELEV] <file> lon1,lat1,height1 [lon2,lat2,height2 ...]
Example:
    python add_waypoints.py wpmz/waylines.wpml 12.180830,49.099500,5.76 12.180770,49.099550,5.76
    python add_waypoints.py --ground-elevation 464.99 wpmz/template.kml 12.180830,49.099500,5.76
"""

import xml.etree.ElementTree as ET
import copy
import sys
import argparse

# Namespaces for parsing
namespaces = {
    'kml': 'http://www.opengis.net/kml/2.2',
    'wpml': 'http://www.dji.com/wpmz/1.0.6'
}
# Register namespaces so they are preserved on write
ET.register_namespace('', namespaces['kml'])
ET.register_namespace('wpml', namespaces['wpml'])

def add_waypoints(file_path, new_points, ground_elevation=548.35):
    """
    Adds new <Placemark> entries to the given WPML/KML file.

    :param file_path: Path to the .wpml or .kml file under wpmz/
    :param new_points: List of dicts with keys 'lon', 'lat', 'height' (AGL in meters)
    :param ground_elevation: Known ground elevation (meters above ellipsoid)
    """
    # Parse XML
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Locate the Folder that contains Placemark entries
    folder = root.find('.//kml:Folder', namespaces)
    if folder is None:
        print("Error: <Folder> element not found in", file_path)
        sys.exit(1)

    # Find all existing Placemark elements
    placemarks = folder.findall('kml:Placemark', namespaces)
    if not placemarks:
        print("Error: No <Placemark> found in", file_path)
        sys.exit(1)

    # Use the last Placemark as a template
    last_pm = placemarks[-1]
    idx_elem = last_pm.find('wpml:index', namespaces)
    last_index = int(idx_elem.text)

    # Detect file type by presence of executeHeight vs. ellipsoidHeight
    is_waylines = last_pm.find('wpml:executeHeight', namespaces) is not None

    # Append new points
    for i, pt in enumerate(new_points):
        new_index = last_index + i + 1
        new_pm = copy.deepcopy(last_pm)

        # Update index
        new_pm.find('wpml:index', namespaces).text = str(new_index)

        # Update coordinates (lon,lat)
        coord_elem = new_pm.find('.//kml:coordinates', namespaces)
        coord_elem.text = f"{pt['lon']},{pt['lat']}"

        # Compute absolute ellipsoid height
        ellipsoid_height = ground_elevation + pt['height']

        if is_waylines:
            # In waylines.wpml, update <executeHeight>
            exec_elem = new_pm.find('wpml:executeHeight', namespaces)
            exec_elem.text = f"{ellipsoid_height}"
        else:
            # In template.kml, update both ellipsoidHeight and height
            new_pm.find('wpml:ellipsoidHeight', namespaces).text = f"{ellipsoid_height}"
            new_pm.find('wpml:height', namespaces).text = f"{pt['height']}"

        # Append to folder
        folder.append(new_pm)

    # Write back to file
    tree.write(file_path, encoding='utf-8', xml_declaration=True)
    print(f"Updated {file_path} with {len(new_points)} new waypoint(s).")

def parse_args():
    parser = argparse.ArgumentParser(description="Add new waypoints to DJI WPML/KML mission files")
    parser.add_argument("file", help="Path to waylines.wpml or template.kml inside wpmz/")
    parser.add_argument("--ground-elevation", type=float, default=464.99,
                        help="Known ground elevation (meters above ellipsoid)")
    parser.add_argument("points", nargs='+',
                        help="New points as 'lon,lat,height' (height = meters AGL)")
    return parser.parse_args()

def main():
    args = parse_args()
    new_points = []
    for p in args.points:
        try:
            lon, lat, h = p.split(',')
            new_points.append({
                "lon": float(lon),
                "lat": float(lat),
                "height": float(h)
            })
        except ValueError:
            print(f"Invalid point format: {p}. Expected 'lon,lat,height'")
            sys.exit(1)

    add_waypoints(args.file, new_points, args.ground_elevation)

if __name__ == "__main__":
    main()

