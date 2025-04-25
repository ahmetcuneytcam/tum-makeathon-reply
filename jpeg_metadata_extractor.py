import exifread
from pathlib import Path

image_name_to_meta_data = {}

folder = Path("./dev_data")

tags_to_look_for = ["GPS GPSLatitude", "GPS GPSLongitude", "GPS GPSAltitude", "EXIF FocalLength", "EXIF ExifImageWidth", "EXIF ExifImageLength", ""]

for image_file in folder.glob("*.jpeg"):
    with image_file.open("rb") as f:
        tags = exifread.process_file(f)
        image_metadata = {}
        for tag in tags.keys():
            if tag in tags_to_look_for:
                image_metadata[tag] = f"{tags[tag]}"

        image_name_to_meta_data[image_file.name] = image_metadata


print(image_name_to_meta_data)