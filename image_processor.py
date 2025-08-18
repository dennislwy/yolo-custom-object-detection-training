"""
Image Preprocessing Utility
Crops and resizes images to square dimensions for object detection training.
Supports multiprocessing for faster batch processing.
"""

import argparse
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

from PIL import Image


def get_user_confirmation(message):
    """
    Get user confirmation for potentially destructive operations.

    Args:
        message: The confirmation message to display

    Returns:
        bool: True if user confirms, False otherwise
    """
    while True:
        response = input(f"{message} (y/n): ").lower().strip()
        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        else:
            print("Please enter 'y' or 'n'")


def would_overwrite_original(input_path, output_path, prefix, suffix):
    """
    Check if the output would overwrite the original file.

    Args:
        input_path: Original input file path
        output_path: Generated output file path
        prefix: Filename prefix
        suffix: Filename suffix

    Returns:
        bool: True if output would overwrite original, False otherwise
    """
    # If user specified prefix or suffix, it won't overwrite
    if prefix or suffix:
        return False

    # Check if paths are the same
    return Path(input_path).resolve() == Path(output_path).resolve()


def get_crop_coordinates(image, crop_from):
    """
    Calculate crop coordinates to create a square image.

    Args:
        image: PIL Image object
        crop_from: Reference point for cropping ('left'/'top', 'middle'/'center', 'right'/'bottom', or integer 0-100)

    Returns:
        tuple: (left, top, right, bottom) coordinates for cropping
    """
    width, height = image.size

    # Determine if image is landscape or portrait
    is_landscape = width > height

    # Convert string values to percentage equivalents
    if isinstance(crop_from, str):
        if crop_from in ["top", "left"]:
            percentage = 0
        elif crop_from in ["bottom", "right"]:
            percentage = 100
        else:  # middle/center
            percentage = 50
    else:
        # crop_from is already an integer percentage
        percentage = max(0, min(100, crop_from))  # Clamp to 0-100 range

    if is_landscape:
        # For landscape images, we crop horizontally
        crop_size = height  # Square size will be the height
        available_space = width - crop_size
        left = int((available_space * percentage) / 100)

        top = 0
        right = left + crop_size
        bottom = height

    else:
        # For portrait images, we crop vertically
        crop_size = width  # Square size will be the width
        available_space = height - crop_size
        top = int((available_space * percentage) / 100)

        left = 0
        right = width
        bottom = top + crop_size

    return (left, top, right, bottom)


def process_image_worker(args):
    """
    Worker function for multiprocessing.

    Args:
        args: Tuple containing (input_path, output_path, crop_from, imgsz)

    Returns:
        tuple: (success: bool, input_path: str, message: str)
    """
    input_path, output_path, crop_from, imgsz = args

    try:
        # Open and validate image
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")

            original_size = img.size

            # Skip if already square and correct size
            if original_size[0] == original_size[1] == imgsz:
                img.save(output_path)
                return (True, str(input_path), f"Already {imgsz}x{imgsz}, copied")

            # Get crop coordinates
            crop_coords = get_crop_coordinates(img, crop_from)

            # Crop to square
            cropped_img = img.crop(crop_coords)

            # Resize to target size
            if cropped_img.size != (imgsz, imgsz):
                resized_img = cropped_img.resize(
                    (imgsz, imgsz), Image.Resampling.LANCZOS
                )
                message = (
                    f"{original_size[0]}x{original_size[1]} → cropped → {imgsz}x{imgsz}"
                )
            else:
                resized_img = cropped_img
                message = f"{original_size[0]}x{original_size[1]} → cropped to {imgsz}x{imgsz}"

            # Save the processed image
            resized_img.save(output_path)

            return (True, str(input_path), message)

    except Exception as e:
        return (False, str(input_path), f"Error: {str(e)}")


def process_image(input_path, output_path, crop_from, imgsz, prefix, suffix):
    """
    Process a single image: crop to square and resize.

    Args:
        input_path: Path to input image
        output_path: Path for output image
        crop_from: Reference point for cropping
        imgsz: Target size for the square image
        prefix: Filename prefix
        suffix: Filename suffix

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open and validate image
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")

            original_size = img.size
            print(f"Processing: {input_path} ({original_size[0]}x{original_size[1]})")

            # Skip if already square and correct size
            if original_size[0] == original_size[1] == imgsz:
                print(f"  Image already {imgsz}x{imgsz}, copying...")
                img.save(output_path)
                return True

            # Get crop coordinates
            crop_coords = get_crop_coordinates(img, crop_from)
            print(f"  Cropping from {crop_from}: {crop_coords}")

            # Crop to square
            cropped_img = img.crop(crop_coords)

            # Resize to target size
            if cropped_img.size != (imgsz, imgsz):
                resized_img = cropped_img.resize(
                    (imgsz, imgsz), Image.Resampling.LANCZOS
                )
                print(f"  Resized to: {imgsz}x{imgsz}")
            else:
                resized_img = cropped_img
                print(f"  Already correct size: {imgsz}x{imgsz}")

            # Save the processed image
            resized_img.save(output_path)
            print(f"  Saved: {output_path}")

            return True

    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False


def generate_output_path(input_path, output_arg, prefix, suffix):
    """
    Generate the output file path based on input and arguments.

    Args:
        input_path: Original input file path
        output_arg: Output argument (can be file or directory)
        prefix: Filename prefix
        suffix: Filename suffix

    Returns:
        str: Output file path
    """
    input_path = Path(input_path)

    if output_arg:
        output_path = Path(output_arg)

        # If output is a directory, generate filename in that directory
        if output_path.is_dir() or (
            not output_path.suffix and not output_path.exists()
        ):
            output_dir = output_path
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{prefix}{input_path.stem}{suffix if suffix else ''}{input_path.suffix}"
            return str(output_dir / filename)
        else:
            # Output is a specific file path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return str(output_path)
    else:
        # Use same directory as input
        filename = (
            f"{prefix}{input_path.stem}{suffix if suffix else ''}{input_path.suffix}"
        )
        return str(input_path.parent / filename)


def is_image_file(file_path):
    """
    Check if a file path corresponds to an image file.

    Args:
        file_path (str): The path of the file to check.

    Returns:
        bool: True if the file is an image file, False otherwise.
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    return any(file_path.lower().endswith(ext) for ext in image_extensions)


def main():
    parser = argparse.ArgumentParser(
        description="Image Preprocessing Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image with default settings (center crop)
  python %(prog)s --input image.jpg
  
  # Process with custom size and crop from left
  python %(prog)s --input image.jpg --imgsz 320 --crop-from left
  
  # Process with fine-grained cropping (25%% from left/top)
  python %(prog)s --input image.jpg --crop-from 25
  
  # Process directory with multiprocessing (auto-detect CPU cores)
  python %(prog)s --input ./images/ --output ./processed/
  
  # Process with 4 workers and custom prefix
  python %(prog)s --input ./images/ --workers 4 --prefix "train_"
  
  # Sequential processing (single worker)
  python %(prog)s --input ./images/ --workers 1
        """,
    )

    parser.add_argument(
        "--input", "-i", required=True, help="Input image file or directory path"
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output image file or directory path (default: same as input)",
    )

    parser.add_argument(
        "--crop-from",
        "-c",
        default="middle",
        help="Reference point to crop image into square. Options: 'left'/'top', 'middle'/'center', 'right'/'bottom', or integer 0-100 (percentage from left/top) (default: middle)",
    )

    parser.add_argument(
        "--imgsz",
        "-s",
        type=int,
        default=640,
        help="Target image size in pixels (creates NxN square) (default: 640)",
    )

    parser.add_argument(
        "--prefix", default="", help='Filename prefix for output images (default: "")'
    )

    parser.add_argument(
        "--suffix",
        default="",
        help='Filename suffix for output images (default: "")',
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=0,
        help="Number of worker processes for batch processing (0=auto, 1=sequential) (default: 0)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.imgsz <= 0:
        print("Error: --imgsz must be a positive integer")
        sys.exit(1)

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input path '{args.input}' does not exist")
        sys.exit(1)

    # Process crop_from argument
    crop_from = args.crop_from
    if crop_from == "center":
        crop_from = "middle"
    elif crop_from.isdigit():
        crop_from = int(crop_from)
        if crop_from < 0 or crop_from > 100:
            print("Error: --crop-from percentage must be between 0 and 100")
            sys.exit(1)
    elif crop_from not in ["left", "top", "middle", "right", "bottom"]:
        print(
            "Error: --crop-from must be 'left', 'top', 'middle', 'center', 'right', 'bottom', or an integer 0-100"
        )
        sys.exit(1)

    # Check for potential overwrite situations before processing
    if not args.prefix and not args.suffix:
        if input_path.is_file():
            # Single file case
            output_path = generate_output_path(
                input_path, args.output, args.prefix, args.suffix
            )
            if would_overwrite_original(
                input_path, output_path, args.prefix, args.suffix
            ):
                if not get_user_confirmation(
                    "WARNING: No prefix or suffix specified. This will overwrite the original image file. Continue?"
                ):
                    print("Operation cancelled by user.")
                    sys.exit(0)
        elif input_path.is_dir() and not args.output:
            # Directory case with no output specified (same directory)
            if not get_user_confirmation(
                "WARNING: No prefix, suffix, or output directory specified. This will overwrite the original image files. Continue?"
            ):
                print("Operation cancelled by user.")
                sys.exit(0)

    # Determine number of workers
    if args.workers == 0:
        num_workers = cpu_count()
    else:
        num_workers = args.workers

    print("Image Preprocessing Utility")
    print(f"Target size: {args.imgsz}x{args.imgsz}")
    print(f"Crop from: {crop_from}")
    print(f"Prefix: '{args.prefix}'")
    print(f"Suffix: '{args.suffix}'")
    print(
        f"Workers: {num_workers} ({'auto-detected' if args.workers == 0 else 'manual'})"
    )
    print("-" * 50)

    # Process files
    processed_count = 0
    failed_count = 0
    start_time = time.time()

    if input_path.is_file():
        # Process single file
        if not is_image_file(str(input_path)):
            print(f"Warning: '{input_path}' may not be a supported image format")

        output_path = generate_output_path(
            input_path, args.output, args.prefix, args.suffix
        )

        if process_image(
            input_path,
            output_path,
            crop_from,
            args.imgsz,
            args.prefix,
            args.suffix,
        ):
            processed_count += 1
        else:
            failed_count += 1

    elif input_path.is_dir():
        # Process directory
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = [
            f
            for f in input_path.rglob("*")
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not image_files:
            print(f"No supported image files found in '{input_path}'")
            sys.exit(1)

        print(f"Found {len(image_files)} image files to process")

        # Prepare arguments for multiprocessing
        process_args = []
        for img_file in image_files:
            output_path = generate_output_path(
                img_file, args.output, args.prefix, args.suffix
            )
            process_args.append((img_file, output_path, crop_from, args.imgsz))

        if num_workers == 1 or len(image_files) == 1:
            # Sequential processing
            print("Using sequential processing...")
            print()

            for img_file in image_files:
                output_path = generate_output_path(
                    img_file, args.output, args.prefix, args.suffix
                )

                if process_image(
                    img_file,
                    output_path,
                    crop_from,
                    args.imgsz,
                    args.prefix,
                    args.suffix,
                ):
                    processed_count += 1
                else:
                    failed_count += 1
                print()
        else:
            # Multiprocessing
            print(f"Using multiprocessing with {num_workers} workers...")
            print()

            try:
                with Pool(processes=num_workers) as pool:
                    # Process images in batches to provide progress updates
                    batch_size = max(1, len(process_args) // 10)  # 10 progress updates

                    for i in range(0, len(process_args), batch_size):
                        batch = process_args[i : i + batch_size]
                        batch_results = pool.map(process_image_worker, batch)

                        # Process results
                        for success, file_path, message in batch_results:
                            if success:
                                processed_count += 1
                                print(f"✓ {Path(file_path).name}: {message}")
                            else:
                                failed_count += 1
                                print(f"✗ {Path(file_path).name}: {message}")

                        # Progress update
                        total_processed = processed_count + failed_count
                        progress = (total_processed / len(image_files)) * 100
                        print(
                            f"Progress: {total_processed}/{len(image_files)} ({progress:.1f}%)"
                        )
                        print()

            except KeyboardInterrupt:
                print("\nProcessing interrupted by user")
                sys.exit(1)
            except Exception as e:
                print(f"Error in multiprocessing: {str(e)}")
                print("Falling back to sequential processing...")

                # Fallback to sequential processing
                processed_count = 0
                failed_count = 0

                for img_file in image_files:
                    output_path = generate_output_path(
                        img_file, args.output, args.prefix, args.suffix
                    )

                    if process_image(
                        img_file,
                        output_path,
                        crop_from,
                        args.imgsz,
                        args.prefix,
                        args.suffix,
                    ):
                        processed_count += 1
                    else:
                        failed_count += 1

    else:
        print(f"Error: '{args.input}' is neither a file nor a directory")
        sys.exit(1)

    # Calculate processing time
    end_time = time.time()
    total_time = end_time - start_time

    # Summary
    print("-" * 50)
    print("Processing complete!")
    print(f"Successfully processed: {processed_count} images")
    if failed_count > 0:
        print(f"Failed to process: {failed_count} images")

    print(f"Total time: {total_time:.2f} seconds")
    if processed_count > 0:
        print(f"Average time per image: {total_time/processed_count:.2f} seconds")

    if processed_count > 1 and num_workers > 1:
        estimated_sequential = total_time * num_workers
        speedup = estimated_sequential / total_time if total_time > 0 else 1
        print(f"Estimated speedup: {speedup:.1f}x (vs sequential processing)")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
