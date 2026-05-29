#!/usr/bin/env python3
"""
Generate an ArUco marker image and write it as an ASCII PPM file.

Usage:
  python generate_aruco_ppm.py --id 0 --out path/to/aruco_4x4_50_id0.ppm --size 512

This script requires OpenCV with aruco support (opencv-python or opencv-contrib-python).
If OpenCV is not available it will print an explanatory error and exit.
"""
import sys
import argparse
import os
import shutil

def write_ppm_ascii(path, rgb_image):
    h = len(rgb_image)
    w = len(rgb_image[0]) if h>0 else 0
    with open(path, "w") as f:
        f.write("P3\n")
        f.write(f"{w} {h}\n")
        f.write("255\n")
        for row in rgb_image:
            line = " ".join(str(v) for px in row for v in px)
            f.write(line + "\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--id", type=int, default=0, help="Marker id (DICT_4X4_50)")
    p.add_argument("--size", type=int, default=512, help="Output image size in pixels (square)")
    p.add_argument("--out", required=True, help="Output .ppm path")
    args = p.parse_args()

    try:
        import cv2
    except Exception as e:
        print("OpenCV import failed. Install opencv-python or opencv-contrib-python in your venv.", file=sys.stderr)
        print("Example: pip install opencv-python-headless", file=sys.stderr)
        sys.exit(2)

    # aruco module API varies; try a couple of access patterns
    try:
        aruco = cv2.aruco
    except AttributeError:
        print("cv2.aruco not found in this OpenCV build", file=sys.stderr)
        sys.exit(3)

    # Get dictionary
    if hasattr(aruco, 'getPredefinedDictionary'):
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    elif hasattr(aruco, 'Dictionary_get'):
        dictionary = aruco.Dictionary_get(aruco.DICT_4X4_50)
    else:
        print("Cannot access ArUco dictionary in this OpenCV build", file=sys.stderr)
        sys.exit(4)

    # Try modern API first (OpenCV >= 4.7), then legacy drawMarker.
    marker = None
    if hasattr(aruco, 'generateImageMarker'):
        try:
            import numpy as np
            marker = np.zeros((args.size, args.size), dtype="uint8")
            aruco.generateImageMarker(dictionary, args.id, args.size, marker, 1)
        except Exception as e:
            print("cv2.aruco.generateImageMarker raised:", e, file=sys.stderr)

    if marker is None and hasattr(aruco, 'drawMarker'):
        try:
            marker = aruco.drawMarker(dictionary, args.id, args.size)
        except Exception as e:
            print("cv2.aruco.drawMarker raised:", e, file=sys.stderr)

    if marker is None:
        # Fallback: if the repository contains an existing PPM for this id, copy it
        repo_ppm = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'mujoco', f'aruco_4x4_50_id{args.id}.ppm'))
        if os.path.exists(repo_ppm):
            try:
                shutil.copyfile(repo_ppm, args.out)
                print(f"Copied existing PPM from {repo_ppm} to {args.out}")
                return
            except Exception as e:
                print("Failed to copy existing PPM:", e, file=sys.stderr)

        # Final fallback: try to construct marker from dictionary.bytesList if present
        try:
            import numpy as np
            bytes_list = getattr(dictionary, 'bytesList', None)
            if bytes_list is not None:
                arr = np.array(bytes_list)
                # arr shape: (nmarkers, markerSize, 1) or similar; try to derive marker size
                # For safety, assume a small grid and construct a simple binary tile
                # We'll attempt to reconstruct a square by reading bits from bytes
                b = arr[args.id]
                # Flatten bytes
                flat = np.asarray(b).flatten().tolist()
                # Convert to bits
                bits = []
                for byte in flat:
                    for i in range(8):
                        bits.append((int(byte) >> i) & 1)
                # Guess marker interior size (4 for DICT_4X4_50)
                marker_size = 4
                grid = [[0]*marker_size for _ in range(marker_size)]
                # Fill grid row-major from bits (best-effort)
                for r in range(marker_size):
                    for c in range(marker_size):
                        idx = r*marker_size + c
                        grid[r][c] = 255 if bits[idx] == 1 else 0
                # Scale up to requested size
                scale = args.size // (marker_size + 2)
                import math
                if scale < 1:
                    scale = 1
                canvas_size = scale * (marker_size + 2)
                # create canvas with white border then place grid
                canvas = [[(255,255,255) for _ in range(canvas_size)] for __ in range(canvas_size)]
                for r in range(marker_size):
                    for c in range(marker_size):
                        val = grid[r][c]
                        for rr in range(scale):
                            for cc in range(scale):
                                y = (1*scale) + r*scale + rr
                                x = (1*scale) + c*scale + cc
                                canvas[y][x] = (val, val, val)
                write_ppm_ascii(args.out, canvas)
                print(f"Wrote best-effort PPM to {args.out} using dictionary.bytesList")
                return
        except Exception:
            pass

        print("Failed to draw marker and no repository PPM available. Install a compatible OpenCV build or supply a PPM.", file=sys.stderr)
        sys.exit(5)

    # Convert to RGB array (list of rows of (r,g,b) tuples)
    try:
        import numpy as np
        if marker.ndim == 2:
            rgb = np.stack([marker]*3, axis=-1)
        else:
            rgb = marker
        # Ensure dtype is uint8
        rgb = rgb.astype('uint8')
        # Convert to nested Python lists for PPM writer
        rgb_list = rgb.tolist()
    except Exception:
        # Fallback: marker is already a Python list-like
        rgb_list = [[(int(px), int(px), int(px)) for px in row] for row in marker]

    write_ppm_ascii(args.out, rgb_list)
    print(f"Wrote PPM to {args.out}")

if __name__ == '__main__':
    main()
