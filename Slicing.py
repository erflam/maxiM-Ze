import os
import time
import pandas as pd
from PIL import Image
from multiprocessing import Pool, cpu_count
import signal

def init_worker():
    """Initialize worker process to ignore keyboard interrupts."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def slice_single_png(args):
    png_path, pixel_csv_path, slice_dir, group_tag = args
    try:
        png_filename = os.path.basename(png_path)
        base = png_filename.replace("EIC_", "").replace(f"_{group_tag}.png", "")

        # Check if pixel mapping file exists
        if not os.path.exists(pixel_csv_path):
            return f"[SKIP] {base}: Pixel mapping file not found"

        # Read pixel mapping CSV
        df = pd.read_csv(pixel_csv_path)

        if df.empty:
            return f"[SKIP] {base}: Empty pixel mapping file"

        # Load the PNG image
        img = Image.open(png_path)
        img_width, img_height = img.size

        slices_created = 0

        # Iterate through each row and create slices
        for idx, row in df.iterrows():
            mz_str = str(row['m/z'])
            seg_id = int(row['Segment_ID'])
            px_start = int(row['Pixel_start'])
            px_end = int(row['Pixel_end'])

            # Validate pixel bounds
            if px_start < 0 or px_end > img_width or px_start >= px_end:
                continue

            # Crop the image (left, top, right, bottom)
            # Slicing vertically, keep full height
            slice_img = img.crop((px_start, 0, px_end, img_height))

            # Create output filename
            # Format: {base}_mz{mz}_seg{seg_id}_{group_tag}.png
            slice_filename = f"{base}_mz{mz_str}_seg{seg_id}_{group_tag}.png"
            slice_path = os.path.join(slice_dir, slice_filename)

            # Save the slice
            slice_img.save(slice_path)
            slices_created += 1

        return f"[OK] {base}: Created {slices_created} slices"

    except Exception as e:
        return f"[ERROR] {os.path.basename(png_path)}: {str(e)}"

def process_file_checkpoint5(cls_instance, dirs, group_name):
    png_dir = dirs['png']
    pixel_dir = dirs['pixel']
    slice_dir = dirs['slice']
    group_tag = str(group_name).replace(" ", "")  # "Group 1" -> "Group1"

    # Ensure slice directory exists
    os.makedirs(slice_dir, exist_ok=True)

    # Find all PNG files for this group
    pngs = [
        os.path.join(png_dir, f)
        for f in os.listdir(png_dir)
        if f.startswith("EIC_") and f.endswith(f"_{group_tag}.png")
    ]

    if not pngs:
        print(f"[!] No PNGs found for {group_name} in {png_dir}")
        print("    (Tip) Here are the first 10 files in that folder:")
        try:
            for x in sorted(os.listdir(png_dir))[:10]:
                print("    -", x)
        except Exception:
            pass
        return

    # Build arguments for parallel processing
    args = []
    for png_path in pngs:
        # Extract base name
        png_filename = os.path.basename(png_path)
        base = png_filename.replace("EIC_", "").replace(f"_{group_tag}.png", "")

        # Construct pixel mapping file path
        pixel_csv_path = os.path.join(pixel_dir, f"{base}_pixelmapping_{group_tag}.csv")

        args.append((png_path, pixel_csv_path, slice_dir, group_tag))

    print(f"Starting Checkpoint 5: PNG Slicing for {group_name}")
    print(f"Found {len(pngs)} PNG files to process")

    start_time = time.time()

    # Process files in parallel
    with Pool(processes=max(1, cpu_count() - 1), initializer=init_worker) as pool:
        results = pool.map(slice_single_png, args)

    # Print results
    for r in results:
        print(r)

    elapsed = time.time() - start_time

    # Count successful slices
    success_count = sum(1 for r in results if r.startswith("[OK]"))
    error_count = sum(1 for r in results if r.startswith("[ERROR]"))
    skip_count = sum(1 for r in results if r.startswith("[SKIP]"))

    print(f"Checkpoint 5 (PNG Slicing) completed in {elapsed:.2f} seconds!")
    print(f"Success: {success_count} | Errors: {error_count} | Skipped: {skip_count}")
    print(f"Slices saved to: {slice_dir}")