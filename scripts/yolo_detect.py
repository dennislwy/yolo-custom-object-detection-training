import argparse
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Define and parse user input arguments

parser = argparse.ArgumentParser(
    description="YOLOv11 Object Detection Utility",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
    # Predict on single image using GPU (minimum confidence threshold 70%%)
    python %(prog)s --model my_model.pt --source test.jpg --thresh 0.7 --device cuda

    # Predict on video file using CPU
    python %(prog)s --model my_model.pt --source testvid.mp4 --thresh 0.7 --device cpu

    # Predict on USB camera using specific GPU
    python %(prog)s --model my_model.pt --source usb0 --thresh 0.7 --device cuda:1

    # Auto-detect device (default)
    python %(prog)s --model my_model.pt --source test.jpg --thresh 0.7
    """,
)
parser.add_argument(
    "--model",
    "-m",
    help='Path to YOLO model file (example: "my_model.pt")',
    required=True,
)
parser.add_argument(
    "--source",
    "-s",
    help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), index of USB camera ("usb0"), or index of Picamera ("picamera0")',
    required=True,
)
parser.add_argument(
    "--thresh",
    "-t",
    type=float,
    help="Minimum confidence threshold for displaying detected objects. Default 0.5",
    default=0.5,
)
parser.add_argument(
    "--resolution",
    "-r",
    help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
    default=None,
)
parser.add_argument(
    "--record",
    help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
    action="store_true",
)
parser.add_argument(
    "--device",
    "-d",
    help='Device to run inference on: "cpu", "cuda", "cuda:0", "cuda:1", etc. Default: auto-detect',
    default=None,
)

args = parser.parse_args()


# Parse user inputs
model_path = Path(args.model)
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record
device = args.device

# Auto-detect device if not specified
if device is None:
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"CUDA detected. Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        device = "cpu"
        print("CUDA not available. Using CPU for inference.")
else:
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA not available. Falling back to CPU.")
            device = "cpu"
        elif device != "cuda" and ":" in device:
            try:
                gpu_idx = int(device.split(":")[1])
                if gpu_idx >= torch.cuda.device_count():
                    print(f"GPU {gpu_idx} not available. Using default CUDA device.")
                    device = "cuda"
                else:
                    # Show specific GPU info
                    gpu_name = torch.cuda.get_device_name(gpu_idx)
                    gpu_memory = (
                        torch.cuda.get_device_properties(gpu_idx).total_memory / 1024**3
                    )
                    print(f"Using GPU {gpu_idx}: {gpu_name} ({gpu_memory:.1f} GB)")
            except (ValueError, IndexError):
                print("Invalid CUDA device format. Using default CUDA device.")
                device = "cuda"
        else:
            # Show default GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")

print(f"Using device: {device}")

# Check if model file exists and is valid
if not model_path.exists():
    print(f"ERROR: Model file '{model_path}' not found.")
    sys.exit(1)

# Load the model into memory and get labemap
try:
    model = YOLO(model_path, task="detect")
    model.to(device)  # Move model to specified device
    labels = model.names
    print(f"Model loaded successfully on '{device}'")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(0)

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP"]
vid_ext_list = [".avi", ".mov", ".mp4", ".mkv", ".wmv"]

if os.path.isdir(img_source):
    source_type = "folder"

elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = "image"
    elif ext in vid_ext_list:
        source_type = "video"
    else:
        print(f"File extension {ext} is not supported.")
        sys.exit(0)

elif "usb" in img_source:
    source_type = "usb"
    try:
        usb_idx = int(img_source[3:])
    except ValueError:
        print(f"Invalid USB camera index: {img_source}")
        sys.exit(0)

elif "picamera" in img_source:
    source_type = "picamera"
    try:
        picam_idx = int(img_source[8:])
    except ValueError:
        print(f"Invalid PiCamera index: {img_source}")
        sys.exit(0)
else:
    print(f"Input {img_source} is invalid. Please try again.")
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    try:
        res_w, res_h = map(int, user_res.split("x"))
        resize = True
    except (ValueError, IndexError):
        print(f"Invalid resolution format: {user_res}. Use format like '640x480'")
        sys.exit(0)

# Check if recording is valid and set up recording
if record:
    if source_type not in ["video", "usb"]:
        print("Recording only works for video and camera sources. Please try again.")
        sys.exit(0)
    if not user_res:
        print("Please specify resolution to record video at.")
        sys.exit(0)

    # Set up recording
    record_name = "demo1.avi"
    record_fps = 30
    recorder = cv2.VideoWriter(
        record_name, cv2.VideoWriter_fourcc(*"MJPG"), record_fps, (res_w, res_h)
    )

print("Keyboard shortcuts:")
print("  'p' - Pause/Resume")
print("  's' - Save screenshot")
print("  'q' - Quit")

# Load or initialize image source
if source_type == "image":
    imgs_list = [img_source]

elif source_type == "folder":
    imgs_list = []
    filelist = glob.glob(img_source + "/*")
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)

elif source_type in ["video", "usb"]:
    if source_type == "video":
        cap_arg = img_source
    elif source_type == "usb":
        cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # Set camera or video resolution if specified by user
    if user_res:
        ret = cap.set(3, res_w)
        ret = cap.set(4, res_h)

elif source_type == "picamera":
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("Picamera2 not installed. Install with: pip install picamera2")
        sys.exit(0)

    cap = Picamera2()
    cap.configure(
        cap.create_video_configuration(
            main={"format": "RGB888", "size": (res_w, res_h)}
        )
    )
    cap.start()

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [
    (164, 120, 87),
    (68, 148, 228),
    (93, 97, 209),
    (178, 182, 133),
    (88, 159, 106),
    (96, 202, 231),
    (159, 124, 168),
    (169, 162, 241),
    (98, 118, 150),
    (172, 176, 184),
]


# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Begin inference loop
while True:

    t_start = time.perf_counter()

    # Load frame from image source
    if source_type in [
        "image",
        "folder",
    ]:  # If source is image or image folder, load the image using its filename
        if img_count >= len(imgs_list):
            print("All images have been processed. Exiting program.")
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count += 1

    elif (
        source_type == "video"
    ):  # If source is a video, load next frame from video file
        ret, frame = cap.read()
        if not ret:
            print("Reached end of the video file. Exiting program.")
            break

    elif source_type == "usb":  # If source is a USB camera, grab frame from camera
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print(
                "Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program."
            )
            break

    elif (
        source_type == "picamera"
    ):  # If source is a Picamera, grab frames using picamera interface
        frame = cap.capture_array()
        if frame is None:
            print(
                "Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program."
            )
            break

    # Resize frame to desired display resolution
    if resize is True:
        frame = cv2.resize(frame, (res_w, res_h))

    # Run inference on frame
    results = model(frame, verbose=False, device=device)

    # Extract results with confidence filtering
    detections = results[0].boxes

    # Filter detections by confidence threshold efficiently
    if detections is not None and len(detections) > 0:
        # Get confidence scores and filter indices
        confidences = detections.conf.cpu().numpy()
        valid_indices = confidences > min_thresh

        if np.any(valid_indices):
            # Vectorized extraction of all valid detections
            xyxy = detections.xyxy.cpu().numpy()[valid_indices].astype(int)
            classes = detections.cls.cpu().numpy()[valid_indices].astype(int)
            conf_filtered = confidences[valid_indices]

            # Count objects efficiently
            object_count = len(conf_filtered)

            # Draw all bounding boxes efficiently
            for i, (bbox, classidx, conf) in enumerate(
                zip(xyxy, classes, conf_filtered)
            ):
                xmin, ymin, xmax, ymax = bbox
                classname = labels[classidx]
                color = bbox_colors[classidx % 10]

                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                # Draw label
                label = f"{classname}: {int(conf*100)}%"
                labelSize, baseLine = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(
                    frame,
                    (xmin, label_ymin - labelSize[1] - 10),
                    (xmin + labelSize[0], label_ymin + baseLine - 10),
                    color,
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    label,
                    (xmin, label_ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )
        else:
            object_count = 0
    else:
        object_count = 0

    # Calculate and draw framerate (if using video, USB, or Picamera source)
    if source_type in ["video", "usb", "picamera"]:
        cv2.putText(
            frame,
            f"FPS: {avg_frame_rate:0.2f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )  # Draw framerate

    # Display detection results
    cv2.putText(
        frame,
        f"Number of objects: {object_count}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )  # Draw total number of detected objects
    cv2.imshow("YOLO detection results", frame)  # Display image
    if record:
        recorder.write(frame)

    # If inferencing on individual images, wait for user keypress before moving to next image.
    # Otherwise, wait 5ms before moving to next frame.
    if source_type in ["image", "folder"]:
        key = cv2.waitKey()
    elif source_type in ["video", "usb", "picamera"]:
        key = cv2.waitKey(5)  # wait 5ms

    if key == ord("q") or key == ord("Q"):  # Press 'q' to quit
        break
    if key == ord("p") or key == ord("P"):  # Press 'p' to pause inference
        cv2.waitKey()
    elif key == ord("s") or key == ord(
        "S"
    ):  # Press 's' to save a picture of results on this frame
        cv2.imwrite("capture.png", frame)

    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1 / (t_stop - t_start))

    # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)

    # Calculate average FPS for past frames
    avg_frame_rate = np.mean(frame_rate_buffer)


# Clean up
print(f"Average pipeline FPS: {avg_frame_rate:.2f}")
if source_type in ["video", "usb"]:
    cap.release()
elif source_type == "picamera":
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()
