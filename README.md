# tum-makeathon-reply
This repository hosts the collaborative work of our team for the Reply Challenge at the 2025 Makeathon, organized by TUMAI.

# KMZ-File-Creation
Using the 'add_waypoints.py' it is possible to add different waypoints to the 'template.kml' and 'waylines.wpml' files by using the following commands with example coordinates.:

```bash
# for waylines.wpml:
python scripts/add_waypoints.py wpmz/waylines.wpml \
    12.180830,49.099500,5.76 12.180770,49.099550,5.76

# for template.kml w. Ground-Elevation:
python scripts/add_waypoints.py --ground-elevation 548.35 \
    wpmz/template.kml 12.180830,49.099500,5.76 12.180770,49.099550,5.76
```

# Faster-RCNN Training and Inference
The "faster-rcnn-training.py" file runs a training on a pre-trained Faster-RCNN model, using annotated images containing barcodes and saves the weights with the name "fine_tuned_faster-rcnn.pth". One can use "faster-rcnn-inference.py" in order to test the performance.

# Faster-RCNN Object Detection Demo Video Generation
Using "object_detection_video_generator.py", one can create a short demo video, showcasing the trained object detection model.