import exifread
from pathlib import Path

image_name_to_meta_data = {}

folder = Path("./images_with_barcodes")

tags_to_look_for = ["GPS GPSLatitude",
                    "GPS GPSLongitude",
                    "GPS GPSAltitude",
                    "EXIF FocalLength",
                    "EXIF ExifImageWidth",
                    "EXIF ExifImageLength",
                    "EXIF FocalLength"]

for image_file in folder.glob("*006_V.jpeg"):
    with image_file.open("rb") as f:
        tags = exifread.process_file(f)
        image_metadata = {}
        for tag in tags.keys():
            #if tag in tags_to_look_for:
            image_metadata[tag] = f"{tags[tag]}"

        image_name_to_meta_data[image_file.name] = image_metadata
    break

print(image_name_to_meta_data)