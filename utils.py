from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

def extract_metadata_from_single_image(img_path):
    with open(img_path, encoding="latin-1") as f:
        data = f.read()

    xmp_start = data.find('<x:xmpmeta')
    xmp_end = data.find('</x:xmpmeta')
    xmp_str = data[xmp_start:xmp_end+12]

    root = ET.fromstring(xmp_str)

    # Find the Description node
    desc = root.find('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description')

    # All attributes
    attributes = desc.attrib
    
    wanted_keys = [
        '{http://www.uav.com/drone-dji/1.0/}GpsLatitude',
        '{http://www.uav.com/drone-dji/1.0/}GpsLongitude',
        '{http://www.uav.com/drone-dji/1.0/}AbsoluteAltitude',
        '{http://www.uav.com/drone-dji/1.0/}RelativeAltitude',
        '{http://www.uav.com/drone-dji/1.0/}GimbalRollDegree',
        '{http://www.uav.com/drone-dji/1.0/}GimbalYawDegree',
        '{http://www.uav.com/drone-dji/1.0/}GimbalPitchDegree',
        '{http://www.uav.com/drone-dji/1.0/}FlightRollDegree',
        '{http://www.uav.com/drone-dji/1.0/}FlightYawDegree',
        '{http://www.uav.com/drone-dji/1.0/}FlightPitchDegree'
    ]

    extracted = { key: attributes.get(key) for key in wanted_keys}

    return [float(extracted[key]) for key in extracted]


def add_meta_information_to_frames_in_tracks(tracks):
    # First, go through all the images and extract the necessary metadata.

    metadata_of_images = []

    folder = Path("./images_with_barcodes")

    for image_file in sorted(folder.glob("*.jpeg")):
        metadata_of_this_image = [] # This will be saved with the following order: [GPSLatitude, GPSLongitude, AbsoluteAltitude, RelativeAltitude, GimbalRollDegree, GimbalYawDegree, GimbalPitchDegree, FlightRollDegree, FlightYawDegree, FlightPitchDegree]

        metadata_of_this_image = extract_metadata_from_single_image(image_file)

        metadata_of_images.append(metadata_of_this_image)

    # Then, add the metadata to the frames in the tracks.


    for track in tracks:
        current_track_dict = tracks[track]

        for track_id in current_track_dict:
            current_track_dict[track_id] = np.append(current_track_dict[track_id], metadata_of_images[int(track_id)])
            


if __name__ == "__main__":
    tracks = {
        "track_1": {
            "0": np.array([200, 300, 500, 600, 0.5]),
            "1": np.array([400, 500, 600, 700, 0.9])
        },
        "track_2": {
            "5": np.array([100, 200, 300, 400, 0.2]),
            "6": np.array([700, 800, 900, 1000, 0.8])
        }
    }

    add_meta_information_to_frames_in_tracks(tracks)

    print(tracks)