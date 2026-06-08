import os
import time
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_max_workers(default: int = 4) -> int:
    """
    Read the same worker setting used by the edited Pipeline.py.

    In Command Prompt you can control this with:
        set MAXIMIZE_MAX_WORKERS=4
        set MAXIMIZE_MAX_WORKERS=6

    We cap this to a reasonable number because checkpoint 5 writes many PNG files.
    Too many workers can make Windows slower, not faster.
    """
    try:
        value = int(os.environ.get("MAXIMIZE_MAX_WORKERS", default))
    except Exception:
        value = default

    return max(1, min(value, 8))


def slice_single_png(args):
    png_path, pixel_csv_path, slice_dir, group_tag = args

    try:
        png_filename = os.path.basename(png_path)
        base = png_filename.replace("EIC_", "").replace(f"_{group_tag}.png", "")

        if not os.path.exists(pixel_csv_path):
            return f"[SKIP] {base}: Pixel mapping file not found"

        # Read only the columns needed for slicing.
        # This is faster than loading extra columns from the CSV.
        try:
            df = pd.read_csv(
                pixel_csv_path,
                usecols=["m/z", "Segment_ID", "Pixel_start", "Pixel_end"],
            )
        except ValueError:
            # Fallback in case an older CSV has slightly different columns.
            df = pd.read_csv(pixel_csv_path)

        if df.empty:
            return f"[SKIP] {base}: Empty pixel mapping file"

        slices_created = 0
        slices_skipped_existing = 0

        # Open the image once per PNG file.
        # The 'with' block closes the file cleanly on Windows.
        with Image.open(png_path) as img:
            img_width, img_height = img.size

            # itertuples() is much faster than iterrows().
            for row in df.itertuples(index=False):
                try:
                    mz_str = str(getattr(row, "_0", None))
                    if mz_str == "None":
                        mz_str = str(getattr(row, "m/z"))
                except Exception:
                    mz_str = str(row[0])

                try:
                    seg_id = int(getattr(row, "Segment_ID"))
                    px_start = int(getattr(row, "Pixel_start"))
                    px_end = int(getattr(row, "Pixel_end"))
                except Exception:
                    # Fallback for unusual column-name handling by pandas.
                    seg_id = int(row[1])
                    px_start = int(row[2])
                    px_end = int(row[3])

                # Validate pixel bounds.
                if px_start < 0 or px_end > img_width or px_start >= px_end:
                    continue

                slice_filename = f"{base}_mz{mz_str}_seg{seg_id}_{group_tag}.png"
                slice_path = os.path.join(slice_dir, slice_filename)

                # If a previous run already made this slice, do not remake it.
                # This helps reruns, but clean runs will still create all slices.
                if os.path.exists(slice_path):
                    slices_skipped_existing += 1
                    continue

                slice_img = img.crop((px_start, 0, px_end, img_height))
                slice_img.save(slice_path)
                slices_created += 1

                # Close cropped image object promptly on Windows.
                slice_img.close()

        if slices_skipped_existing:
            return (
                f"[OK] {base}: Created {slices_created} slices "
                f"({slices_skipped_existing} already existed)"
            )

        return f"[OK] {base}: Created {slices_created} slices"

    except Exception as e:
        return f"[ERROR] {os.path.basename(png_path)}: {str(e)}"


def process_file_checkpoint5(cls_instance, dirs, group_name):
    png_dir = dirs["png"]
    pixel_dir = dirs["pixel"]
    slice_dir = dirs["slice"]
    group_tag = str(group_name).replace(" ", "")  # "Group 1" -> "Group1"

    os.makedirs(slice_dir, exist_ok=True)

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

    args = []
    for png_path in pngs:
        png_filename = os.path.basename(png_path)
        base = png_filename.replace("EIC_", "").replace(f"_{group_tag}.png", "")
        pixel_csv_path = os.path.join(pixel_dir, f"{base}_pixelmapping_{group_tag}.csv")
        args.append((png_path, pixel_csv_path, slice_dir, group_tag))

    max_workers = min(get_max_workers(), len(args))

    print(f"Starting Checkpoint 5: PNG Slicing for {group_name}")
    print(f"Found {len(pngs)} PNG files to process")
    print(f"Using {max_workers} slicing workers")

    start_time = time.time()

    # Use threads instead of multiprocessing here.
    # Why:
    #   - This step opens PNGs, reads CSVs, crops images, and writes many PNG files.
    #   - On Windows, spawning many Python processes inside a PyInstaller exe is expensive.
    #   - Threads avoid that spawn cost and still overlap file I/O.
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_arg = {executor.submit(slice_single_png, arg): arg for arg in args}
        for future in as_completed(future_to_arg):
            try:
                results.append(future.result())
            except Exception as e:
                png_path = future_to_arg[future][0]
                results.append(f"[ERROR] {os.path.basename(png_path)}: {str(e)}")

    # Keep printed output stable/readable.
    for r in sorted(results):
        print(r)

    elapsed = time.time() - start_time

    success_count = sum(1 for r in results if r.startswith("[OK]"))
    error_count = sum(1 for r in results if r.startswith("[ERROR]"))
    skip_count = sum(1 for r in results if r.startswith("[SKIP]"))

    print(f"Checkpoint 5 (PNG Slicing) completed in {elapsed:.2f} seconds!")
    print(f"Success: {success_count} | Errors: {error_count} | Skipped: {skip_count}")
    print(f"Slices saved to: {slice_dir}")
