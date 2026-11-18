"""manual_microchannel.py

Created by Robert Blum at Georgia Institute of Technology.

    Comprehensive microfluidic channel analysis tool with configurable ROI selection and automatic fluorescence quantification.

    Workflow:
    1. User configures analysis settings:
       - Number of replicates and partitions per channel
       - Device type (standard or thin) for channel detection parameters
       - Advanced options: debug mode, ROI selection mode (fixed size or free drag), custom ROI dimensions
    2. For each replicate:
       - User selects ROI via interactive window with live magnified preview
       - Channels automatically detected using two-stage segmentation (solid blocks + edge-based refinement)
       - Missing channels intelligently inferred from spacing patterns
       - User validates detected channels (with option to retry ROI selection)
    3. Fluorescence analysis:
       - Calculates total and partition-based (horizontal) green/red percentages
       - Uses color-based thresholds with configurable sensitivity
       - Processes each partition relative to total channel pixel count
    4. Results exported to Excel with separate sheets per replicate plus debug log

    Key Features:
    - Dual ROI selection modes: fixed-size draggable box or free drag rectangle definition
    - Advanced channel detection with post-processing, validation, and outlier correction
    - Intelligent missing channel inference at device boundaries and inter-channel gaps
    - Interactive validation workflow with retry capability
    - Live magnified ROI preview during selection
    - Device-specific detection parameters (standard: 17-30px, thin: 8-15px channels)
    - Comprehensive debug logging and optional visualization
    - Formatted Excel output with percentage columns
"""

# Import libraries
import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import os
import glob
from io import StringIO
import sys

root = tk.Tk()
root.withdraw()
debug_messages = []

# ============================================================
# DEVICE CONFIGURATION
# ============================================================
DEVICE_CONFIG = {
    "standard": {
        "min_height": 17,
        "max_height": 30,
        "min_gap_height": 8,
        "edge_min_height": 10,
        "edge_max_height": 40,
        "device_channel_range": (15, 25),
        "roi_width": 20,
        "roi_height": 115,
        "name": "Standard Device"
    },
    "thin": {
        "min_height": 8,
        "max_height": 15,
        "min_gap_height": 12,
        "edge_min_height": 5,
        "edge_max_height": 25,
        "device_channel_range": (8, 15),
        "roi_width": 22,
        "roi_height": 115,
        "name": "Thin Device"
    }
}

# Error Popup in case of critical errors 
#   that will not allow the program to run
def show_error_popup(title, message):
    """Display a CRITICAL error message and halt the program"""
    error_window = tk.Toplevel(root)
    error_window.title(title)
    error_window.geometry("500x300")
    error_window.resizable(False, False)
    error_window.attributes('-topmost', True)
    title_label = tk.Label(error_window, text="❌ Critical Error", font=("Arial", 16, "bold"), fg="red")
    title_label.pack(pady=20)
    
    #Error Message
    message_label = tk.Label(error_window, text=message, font=("Arial", 12), wraplength=450, justify="center")
    message_label.pack(pady=20)
    
    def exit_program():
        root.destroy()
        sys.exit(1)
    
    ok_button = tk.Button(error_window, text="Exit Program", font=("Arial", 12, "bold"),
                          bg="#ff4444", fg="white", command=exit_program, padx=40, pady=15)
    ok_button.pack(pady=20)
    
    error_window.protocol("WM_DELETE_WINDOW", exit_program)
    error_window.mainloop()
    
# Choose a folder containing all three images
folder_path = filedialog.askdirectory(title='Select folder containing OV, Red, and Green images')

if not folder_path:
    show_error_popup("No Folder Selected", "No folder was selected.\n\nPlease select a folder and try again.")
# Change working directory to the selected folder
os.chdir(folder_path)

# Get all .tif files in the folder
tif_files = glob.glob(os.path.join(folder_path, '*.tif'))

if len(tif_files) == 0:
    show_error_popup("No Images Found", "No .tif files found in the selected folder.\n\nPlease ensure your folder contains .tif images.")

# Auto-detect images based on filename patterns (case-insensitive)
OV_imglist = []
red_imglist = []
green_imglist = []

for file_path in sorted(tif_files):
    filename_lower = os.path.basename(file_path).lower()
    
    if 'ov' in filename_lower or 'overview' in filename_lower:
        OV_imglist.append(file_path)
    elif 'red' in filename_lower:
        red_imglist.append(file_path)
    elif 'green' in filename_lower:
        green_imglist.append(file_path)

# Verify all three image types were found
missing_images = []
if not OV_imglist:
    missing_images.append("OV/Overview image")
if not red_imglist:
    missing_images.append("Red image")
if not green_imglist:
    missing_images.append("Green image")

if missing_images:
    missing_text = "\n".join(missing_images)
    show_error_popup("Missing Required Images", 
                     f"Could not find the following required images:\n\n{missing_text}\n\nPlease check your filenames and try again.")

print("Loaded files:")
print(f"  OV: {os.path.basename(OV_imglist[0])}")
print(f"  Red: {os.path.basename(red_imglist[0])}")
print(f"  Green: {os.path.basename(green_imglist[0])}")

# Set up file to save data
df_channels = pd.DataFrame()

#Set ROI selection image popup window height and width. Required for Windows users, Mac OS auto-scales.
popup_width, popup_height = 300, 486

def debug_print(message):
    """Capture debug messages for Excel output"""
    print(message)
    debug_messages.append(message)

def detect_edges_and_pair(line_mask, min_height=10, max_height=40, device_channel_range=(15, 25)):
    """
    Detect channels using horizontal edge pairs instead of solid blocks.
    Works for both bright AND dark channel interiors.
    Prioritizes CLOSEST edge pairs to avoid incorrect pairing.
    Detects alternating channel/gap patterns and uses device constraints to disambiguate.
    
    Parameters
    ----------
    line_mask : np.ndarray
        Binary mask showing contrast lines (from majority vote)
    min_height : int
        Minimum distance between edge pairs (default: 10)
    max_height : int
        Maximum distance between edge pairs (default: 40)
    device_channel_range : tuple
        Expected channel size range in pixels (default: (15, 25))
    
    Returns
    -------
    edge_pairs : list of tuple
        List of (y_top, y_bottom) edge pairs forming channels
    alternating_detected : bool
        Whether alternating pattern was detected
    """
    H, W = line_mask.shape
    
    # Find all rows with significant white pixels (horizontal edges)
    edge_rows = []
    for y in range(H):
        white_count = np.sum(line_mask[y, :] == 255)
        if white_count > W * 0.3:  # At least 30% of row is white
            edge_rows.append(y)
    
    if len(edge_rows) < 2:
        debug_print("DEBUG EDGE: Too few edge rows detected")
        return [], False
    
    # Group nearby edges (within 2 pixels) into single edges
    grouped_edges = []
    current_group = [edge_rows[0]]
    
    for i in range(1, len(edge_rows)):
        if edge_rows[i] - edge_rows[i-1] <= 2:
            current_group.append(edge_rows[i])
        else:
            grouped_edges.append(int(np.median(current_group)))
            current_group = [edge_rows[i]]
    
    if current_group:
        grouped_edges.append(int(np.median(current_group)))
    
    debug_print(f"DEBUG EDGE: Found {len(grouped_edges)} distinct horizontal edges")
    
    # Calculate consecutive spacings
    consecutive_spacings = []
    for i in range(len(grouped_edges) - 1):
        spacing = grouped_edges[i+1] - grouped_edges[i]
        if min_height <= spacing <= max_height:
            consecutive_spacings.append(spacing)
    
    if not consecutive_spacings:
        debug_print("DEBUG EDGE: No valid consecutive spacings found")
        return [], False
    
    median_spacing = np.median(consecutive_spacings)
    spacing_std = np.std(consecutive_spacings)
    
    debug_print(f"DEBUG EDGE: Consecutive spacing stats: median={median_spacing:.1f}, std={spacing_std:.1f}")
    
    # Check for alternating pattern
    alternate_spacings = []
    for i in range(0, len(grouped_edges) - 2, 2):
        spacing = grouped_edges[i+2] - grouped_edges[i]
        if min_height * 2 <= spacing <= max_height * 2:
            alternate_spacings.append(spacing)
    
    alternating_detected = False
    
    if len(alternate_spacings) >= 3:
        median_alternate = np.median(alternate_spacings)
        ratio = median_alternate / median_spacing
        cv_consecutive = spacing_std / median_spacing
        cv_alternate = np.std(alternate_spacings) / median_alternate
        
        debug_print(f"DEBUG EDGE: Alternate spacing median={median_alternate:.1f}, ratio={ratio:.2f}")
        debug_print(f"DEBUG EDGE: CV consecutive={cv_consecutive:.3f}, CV alternate={cv_alternate:.3f}")
        
        if cv_alternate < cv_consecutive * 0.85 and 1.8 <= ratio <= 2.2:
            alternating_detected = True
            debug_print(f"DEBUG EDGE: ALTERNATING pattern detected")
            
            channel_spacings = [s for s in consecutive_spacings 
                               if device_channel_range[0] <= s <= device_channel_range[1]]
            gap_spacings = [s for s in consecutive_spacings 
                           if s < device_channel_range[0]]
            
            debug_print(f"DEBUG EDGE: Channel-sized spacings: {len(channel_spacings)} "
                       f"(median={np.median(channel_spacings) if channel_spacings else 0:.1f})")
            debug_print(f"DEBUG EDGE: Gap-sized spacings: {len(gap_spacings)} "
                       f"(median={np.median(gap_spacings) if gap_spacings else 0:.1f})")
            
            if len(channel_spacings) >= 3:
                # Use channel spacings to set range
                expected_channel_height = np.median(channel_spacings)
                adaptive_min = expected_channel_height * 0.80
                adaptive_max = expected_channel_height * 1.20
                debug_print(f"DEBUG EDGE: Using device constraint - channel height={expected_channel_height:.1f}")
            else:
                # Fallback
                actual_channel_height = median_alternate / 2
                adaptive_min = actual_channel_height * 0.70
                adaptive_max = actual_channel_height * 1.30
        else:
            debug_print(f"DEBUG EDGE: Normal pattern - using consecutive spacing")
            adaptive_min = median_spacing * 0.75
            adaptive_max = median_spacing * 1.25
    else:
        debug_print(f"DEBUG EDGE: Insufficient alternate samples - using consecutive spacing")
        adaptive_min = median_spacing * 0.75
        adaptive_max = median_spacing * 1.25
    
    debug_print(f"DEBUG EDGE: Using adaptive height range: [{adaptive_min:.1f}, {adaptive_max:.1f}]")
    
    # SEQUENTIAL PAIRING with RUNNING AVERAGES
    edge_pairs = []
    used_edges = set()
    
    # Bootstrap
    BOOTSTRAP_CHANNELS = 3
    channel_heights = []
    channel_gaps = []
    
    i = 0
    channels_found = 0
    
    while i < len(grouped_edges) and channels_found < BOOTSTRAP_CHANNELS:
        if i in used_edges:
            i += 1
            continue
        
        top_edge = grouped_edges[i]
        best_match = None
        best_distance = float('inf')
        
        for j in range(i+1, len(grouped_edges)):
            if j in used_edges:
                continue
            
            bottom_edge = grouped_edges[j]
            height = bottom_edge - top_edge
            
            if adaptive_min <= height <= adaptive_max:
                distance = abs(height - (adaptive_max + adaptive_min) / 2)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = j
                
                if distance < (adaptive_max + adaptive_min) / 2 * 0.1:
                    break
            
            if height > adaptive_max * 1.5:
                break
        
        if best_match is not None:
            bottom_edge = grouped_edges[best_match]
            height = bottom_edge - top_edge
            edge_pairs.append((top_edge, bottom_edge))
            used_edges.add(i)
            used_edges.add(best_match)
            channel_heights.append(height)
            
            if best_match + 1 < len(grouped_edges):
                gap = grouped_edges[best_match + 1] - bottom_edge
                channel_gaps.append(gap)
            
            debug_print(f"DEBUG EDGE: Bootstrap ch{channels_found+1}: {top_edge}-{bottom_edge} (height={height})")
            channels_found += 1
            i = best_match + 1
        else:
            i += 1
    
    # Adaptive mode
    if len(channel_heights) >= 2 and len(channel_gaps) >= 1:
        debug_print(f"DEBUG EDGE: Bootstrap complete, switching to adaptive mode")
        
        while i < len(grouped_edges):
            if i in used_edges:
                i += 1
                continue
            
            avg_height = np.mean(channel_heights)
            avg_gap = np.mean(channel_gaps) if channel_gaps else 0
            
            last_channel_end = edge_pairs[-1][1]
            predicted_start = last_channel_end + avg_gap
            
            height_tolerance = avg_height * 0.25
            gap_tolerance = avg_gap * 0.4 if avg_gap > 0 else avg_height
            
            debug_print(f"DEBUG EDGE: Predicted next channel near {predicted_start:.0f}, avg_height={avg_height:.1f}, avg_gap={avg_gap:.1f}")
            
            top_edge = grouped_edges[i]
            actual_gap = top_edge - last_channel_end
            
            if abs(actual_gap - avg_gap) > gap_tolerance:
                debug_print(f"DEBUG EDGE: Gap violation at edge {top_edge}: expected ~{avg_gap:.1f}, got {actual_gap:.1f}, skipping")
                i += 1
                continue
            
            best_match = None
            best_distance = float('inf')
            
            for j in range(i+1, len(grouped_edges)):
                if j in used_edges:
                    continue
                
                bottom_edge = grouped_edges[j]
                height = bottom_edge - top_edge
                
                if abs(height - avg_height) <= height_tolerance:
                    distance = abs(height - avg_height)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = j
                    
                    if distance < avg_height * 0.1:
                        break
                
                if height > avg_height + height_tolerance:
                    break
            
            if best_match is not None:
                bottom_edge = grouped_edges[best_match]
                height = bottom_edge - top_edge
                edge_pairs.append((top_edge, bottom_edge))
                used_edges.add(i)
                used_edges.add(best_match)
                
                channel_heights.append(height)
                if best_match + 1 < len(grouped_edges):
                    gap = grouped_edges[best_match + 1] - bottom_edge
                    channel_gaps.append(gap)
                
                debug_print(f"DEBUG EDGE: Paired edges {top_edge}-{bottom_edge} (height={height}, gap_before={actual_gap:.1f})")
                i = best_match + 1
            else:
                debug_print(f"DEBUG EDGE: No valid match found near prediction, skipping edge {top_edge}")
                i += 1
    
    debug_print(f"DEBUG EDGE: Created {len(edge_pairs)} channel pairs from edges")
    
    return edge_pairs, alternating_detected


def post_process_channels(channels, line_mask, max_expansion=3, min_white_threshold=0.15):
    """
    Combined post-processing: expand boundaries and validate content.
    
    Parameters
    ----------
    channels : list of tuple
        Detected channels (y_start, y_end)
    line_mask : np.ndarray
        Binary mask showing contrast lines
    max_expansion : int
        Maximum pixels to expand in each direction (default: 3)
    min_white_threshold : float
        Minimum fraction of white pixels required (default: 0.15 = 15%)
    
    Returns
    -------
    processed_channels : list of tuple
        Expanded and validated channels
    """
    H, W = line_mask.shape
    processed = []
    
    for idx, (start, end) in enumerate(channels):
        # Step 1: Expand boundaries
        new_start = start
        new_end = end
        
        for offset in range(1, max_expansion + 1):
            check_y = start - offset
            if check_y < 0:
                break
            if np.sum(line_mask[check_y, :] == 255) > W * 0.3:
                new_start = check_y
            else:
                break
        
        for offset in range(1, max_expansion + 1):
            check_y = end + offset
            if check_y >= H:
                break
            if np.sum(line_mask[check_y, :] == 255) > W * 0.3:
                new_end = check_y
            else:
                break
        
        # Step 2: Validate content
        channel_region = line_mask[new_start:new_end, :]
        total_pixels = channel_region.size
        white_pixels = np.sum(channel_region == 255)
        white_fraction = white_pixels / total_pixels if total_pixels > 0 else 0
        
        if white_fraction >= min_white_threshold:
            processed.append((new_start, new_end))
            debug_print(f"DEBUG POST-PROCESS: Ch{idx+1}: {start}-{end} → {new_start}-{new_end} "
                       f"(gained {(start-new_start)+(new_end-end)} px, white={white_fraction:.2%}) ✓")
        else:
            debug_print(f"DEBUG POST-PROCESS: Ch{idx+1}: REJECTED (white={white_fraction:.2%})")
    
    debug_print(f"DEBUG POST-PROCESS: Kept {len(processed)}/{len(channels)} channels")
    return processed

def validate_and_fix_channels(channels, line_mask, all_edges):
    """
    Validate detected channels and fix outliers using statistical analysis.
    Prioritizes largest white blocks in problem regions.
    
    Parameters
    ----------
    channels : list of tuple
        Initial detected channels (y_start, y_end)
    line_mask : np.ndarray
        Binary mask showing contrast lines
    all_edges : list of int
        All detected horizontal edge positions
    
    Returns
    -------
    fixed_channels : list of tuple
        Corrected channel list
    """
    if len(channels) < 3:
        return channels
    
    # Calculate statistics
    heights = [end - start for start, end in channels]
    gaps = [channels[i+1][0] - channels[i][1] for i in range(len(channels)-1)]
    
    median_height = np.median(heights)
    median_gap = np.median(gaps)
    
    # Define outlier thresholds (±30% from median)
    min_acceptable_height = median_height * 0.7
    max_acceptable_height = median_height * 1.3
    
    debug_print(f"DEBUG VALIDATE: median_height={median_height:.1f}, median_gap={median_gap:.1f}")
    debug_print(f"DEBUG VALIDATE: acceptable height range = [{min_acceptable_height:.1f}, {max_acceptable_height:.1f}]")
    
    # Find outliers
    outliers = []
    for idx, (start, end) in enumerate(channels):
        height = end - start
        if height < min_acceptable_height or height > max_acceptable_height:
            outliers.append(idx)
            debug_print(f"DEBUG VALIDATE: Channel {idx+1} is OUTLIER (height={height})")
    
    if not outliers:
        debug_print("DEBUG VALIDATE: No outliers found")
        return channels
    
    # Fix outliers by finding largest white block in problem regions
    fixed_channels = list(channels)
    H, W = line_mask.shape
    
    for outlier_idx in outliers:
        start, end = channels[outlier_idx]
        height = end - start
        
        # Define search region (expand by 1.5x median_gap on each side)
        region_start = max(0, start - int(median_gap * 1.5))
        region_end = min(H, end + int(median_gap * 1.5))
        
        debug_print(f"DEBUG VALIDATE: Analyzing outlier ch{outlier_idx+1} in region [{region_start}, {region_end}]")
        
        # Find all contiguous white blocks in this region
        white_blocks = []
        in_block = False
        block_start = None
        
        for y in range(region_start, region_end):
            row_has_white = np.sum(line_mask[y, :] == 255) > W * 0.3
            
            if row_has_white:
                if not in_block:
                    in_block = True
                    block_start = y
            else:
                if in_block:
                    block_end = y
                    block_height = block_end - block_start
                    white_blocks.append((block_start, block_end, block_height))
                    in_block = False
        
        # Handle block extending to end of region
        if in_block:
            block_end = region_end
            block_height = block_end - block_start
            white_blocks.append((block_start, block_end, block_height))
        
        debug_print(f"DEBUG VALIDATE: Found {len(white_blocks)} white blocks: {[(s, e, h) for s, e, h in white_blocks]}")
        
        if not white_blocks:
            debug_print(f"DEBUG VALIDATE: No white blocks found for ch{outlier_idx+1}")
            continue
        
        # Find the LARGEST white block (by height)
        largest_block = max(white_blocks, key=lambda x: x[2])
        largest_start, largest_end, largest_height = largest_block
        
        debug_print(f"DEBUG VALIDATE: Largest block: {largest_start}-{largest_end} (height={largest_height})")
        
        # Validate this block is reasonable
        if min_acceptable_height <= largest_height <= max_acceptable_height:
            debug_print(f"DEBUG VALIDATE: ✓ Replacing ch{outlier_idx+1} with largest block: {largest_start}-{largest_end}")
            fixed_channels[outlier_idx] = (largest_start, largest_end)
        else:
            # Largest block is still unreasonable - try to find edges within it
            edges_in_largest = [e for e in all_edges if largest_start <= e <= largest_end]
            
            if len(edges_in_largest) >= 2:
                # Use first and last edge in largest block
                corrected_start = edges_in_largest[0]
                corrected_end = edges_in_largest[-1]
                corrected_height = corrected_end - corrected_start
                
                debug_print(f"DEBUG VALIDATE: Largest block unreasonable, using edges: {corrected_start}-{corrected_end} (height={corrected_height})")
                fixed_channels[outlier_idx] = (corrected_start, corrected_end)
            else:
                debug_print(f"DEBUG VALIDATE: ✗ Could not fix ch{outlier_idx+1}")
    
    # Re-sort channels
    fixed_channels = sorted(fixed_channels, key=lambda x: x[0])
    
    # Remove overlapping channels (keep the one closer to median height)
    final_channels = []
    skip_next = False
    
    for i in range(len(fixed_channels)):
        if skip_next:
            skip_next = False
            continue
        
        if i < len(fixed_channels) - 1:
            curr_start, curr_end = fixed_channels[i]
            next_start, next_end = fixed_channels[i+1]
            
            # Check for overlap
            if curr_end >= next_start:
                debug_print(f"DEBUG VALIDATE: Overlap detected between ch{i+1} and ch{i+2}")
                # Keep the one with height closer to median
                curr_height = curr_end - curr_start
                next_height = next_end - next_start
                curr_diff = abs(curr_height - median_height)
                next_diff = abs(next_height - median_height)
                
                if curr_diff < next_diff:
                    final_channels.append((curr_start, curr_end))
                    skip_next = True
                else:
                    skip_next = True
                    continue
            else:
                final_channels.append((curr_start, curr_end))
        else:
            final_channels.append(fixed_channels[i])
    
    debug_print(f"DEBUG VALIDATE: Validation complete. {len(channels)} → {len(final_channels)} channels")
    
    return final_channels

def detect_channels(OV_img, min_height=17, max_height=30, min_gap_height=8, device_channel_range=(15, 25)):
    """
    Simplified channel detection using line-by-line majority vote with intelligent
    missing channel detection based on spacing patterns.
    """
    # 1. Convert to grayscale
    if OV_img.ndim == 3:
        gray = cv2.cvtColor(OV_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = OV_img.copy()

    # 2. Apply Otsu threshold
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    manual_thresh = otsu_thresh * 1.01
    _, binary = cv2.threshold(gray, manual_thresh, 255, cv2.THRESH_BINARY)

    pre_mask = binary.copy()
    H, W = binary.shape

    # 3. Line-by-line majority vote mask
    line_mask = np.zeros((H, W), dtype=np.uint8)
    num_slices = 30
    slice_width = W // num_slices

    for y in range(H):
        row = binary[y, :]
        slice_votes = []

        for i in range(num_slices):
            start = i * slice_width
            end = (i + 1) * slice_width if i < num_slices - 1 else W
            slice_region = row[start:end]
            white_pixels = np.sum(slice_region > 0)
            total_pixels = end - start
            is_white = white_pixels > (0.4 * total_pixels)
            slice_votes.append(is_white)
            row[start:end] = 255 if is_white else 0

        white_vote_count = sum(slice_votes)
        line_mask[y, :] = 255 if white_vote_count >= (num_slices * 0.2) else 0

    # 4. TWO-STAGE DETECTION
    debug_print("DEBUG: STAGE 1 - Attempting solid-block detection")
    channels = []
    in_channel = False
    start = None
    gap_count = 0
    used_edge_detection = False

    for y in range(H):
        is_white_row = np.any(line_mask[y, :] == 255)

        if is_white_row:
            if not in_channel:
                in_channel = True
                start = y
            gap_count = 0
        else:
            if in_channel:
                gap_count += 1
                if gap_count >= min_gap_height:
                    end = y - gap_count
                    height = end - start
                    if height >= min_height and (max_height is None or height <= max_height):
                        channels.append((start, end))
                    in_channel = False
                    start = None
                    gap_count = 0

    if in_channel:
        end = H
        height = end - start
        if height >= min_height and (max_height is None or height <= max_height):
            channels.append((start, end))
    
    debug_print(f"DEBUG: STAGE 1 found {len(channels)} channels using solid-block method")
    
    # STAGE 2: Edge-based detection
    if len(channels) < 10:
        debug_print("DEBUG: STAGE 2 - Too few channels, switching to edge-based detection")
        used_edge_detection = True
        
        # Get all edges for validation
        all_edge_rows = []
        for y in range(H):
            white_count = np.sum(line_mask[y, :] == 255)
            if white_count > W * 0.3:
                all_edge_rows.append(y)
        
        grouped_all_edges = []
        if all_edge_rows:
            current_group = [all_edge_rows[0]]
            for i in range(1, len(all_edge_rows)):
                if all_edge_rows[i] - all_edge_rows[i-1] <= 2:
                    current_group.append(all_edge_rows[i])
                else:
                    grouped_all_edges.append(int(np.median(current_group)))
                    current_group = [all_edge_rows[i]]
            if current_group:
                grouped_all_edges.append(int(np.median(current_group)))
        
        edge_channels, alternating_detected = detect_edges_and_pair(
            line_mask, min_height=min_height//2, max_height=max_height*1.3, 
            device_channel_range=device_channel_range
        )
        
        if len(edge_channels) >= 2:
            debug_print(f"DEBUG: STAGE 2 - Edge-based detection found {len(edge_channels)} channels")
            channels = validate_and_fix_channels(edge_channels, line_mask, grouped_all_edges)
            debug_print(f"DEBUG: STAGE 2 - After validation: {len(channels)} channels")
            
            # ✨ NEW: Combined post-processing
            channels = post_process_channels(channels, line_mask, max_expansion=3, min_white_threshold=0.15)
            debug_print(f"DEBUG: STAGE 2 - After post-processing: {len(channels)} channels")
        else:
            debug_print(f"DEBUG: STAGE 2 - Edge-based detection failed, keeping {len(channels)} channels from STAGE 1")
            if len(channels) > 0:
                channels = post_process_channels(channels, line_mask, max_expansion=3, min_white_threshold=0.15)
                debug_print(f"DEBUG: STAGE 2 - After post-processing STAGE 1 channels: {len(channels)} channels")
    else:
        debug_print(f"DEBUG: STAGE 1 sufficient - keeping {len(channels)} channels")
        channels = post_process_channels(channels, line_mask, max_expansion=3, min_white_threshold=0.15)
        debug_print(f"DEBUG: STAGE 1 - After post-processing: {len(channels)} channels")

    #Conditional merging - skip if STAGE 2 edge detection was used
    if used_edge_detection:
        debug_print("DEBUG: Skipping merge step - STAGE 2 edge detection already paired channels correctly")
    else:
        # 5. Merge channels separated by small gaps (STAGE 1 only)
        merged_channels = []
        if channels:
            current_start, current_end = channels[0]
            for i in range(1, len(channels)):
                next_start, next_end = channels[i]
                if next_start - current_end < min_gap_height:
                    current_end = next_end
                else:
                    merged_channels.append((current_start, current_end))
                    current_start, current_end = next_start, next_end
            merged_channels.append((current_start, current_end))
            channels = merged_channels
            debug_print(f"DEBUG: After merging: {len(channels)} channels")

    # 5a. Detect missing channels based on spacing pattern analysis
    if len(channels) >= 2:
        # Get all edges for missing channel detection
        all_edge_rows = []
        for y in range(H):
            white_count = np.sum(line_mask[y, :] == 255)
            if white_count > W * 0.3:
                all_edge_rows.append(y)

        grouped_all_edges = []
        if all_edge_rows:
            current_group = [all_edge_rows[0]]
            for i in range(1, len(all_edge_rows)):
                if all_edge_rows[i] - all_edge_rows[i-1] <= 2:
                    current_group.append(all_edge_rows[i])
                else:
                    grouped_all_edges.append(int(np.median(current_group)))
                    current_group = [all_edge_rows[i]]
            if current_group:
                grouped_all_edges.append(int(np.median(current_group)))

        debug_print(f"DEBUG: Found {len(grouped_all_edges)} distinct edges for missing channel detection")

        # Calculate average channel height (all detected channels should be similar)
        heights = [end - start for start, end in channels]
        avg_height = np.mean(heights)

        # Calculate average inter-channel gap (gap between consecutive channels)
        gaps = [channels[i+1][0] - channels[i][1] for i in range(len(channels)-1)]
        avg_gap = np.median(gaps) if gaps else 0

        # Expected spacing for one channel + one gap
        expected_spacing = avg_height + avg_gap

        # Debug output
        debug_print(f"DEBUG: avg_height={avg_height:.1f}, avg_gap={avg_gap:.1f}, expected_spacing={expected_spacing:.1f}")
        debug_print(f"DEBUG: Threshold for missing channel = {expected_spacing * 1.15:.1f}")

        # Helper to extract white blocks from a region
        def find_white_blocks(mask_region, start_y):
            """Find contiguous white blocks in a region of line_mask"""
            white_blocks = []
            in_white = False
            block_start = None

            for y_index in range(len(mask_region)):
                if np.any(mask_region[y_index] == 255):
                    if not in_white:
                        in_white = True
                        block_start = start_y + y_index
                else:
                    if in_white:
                        block_end = start_y + y_index - 1
                        white_blocks.append((block_start, block_end))
                        in_white = False
                        block_start = None

            # Handle white block that continues to end of region
            if in_white:
                block_end = start_y + len(mask_region) - 1
                white_blocks.append((block_start, block_end))

            return white_blocks

        missing_channels = []

        # CHECK FOR MISSING CHANNELS AT THE BEGINNING
        first_channel_start = channels[0][0]
        gap_at_start = first_channel_start

        debug_print(f"DEBUG: Gap at start: {gap_at_start}, threshold={expected_spacing * 1.15:.1f}")

        if gap_at_start > expected_spacing * 1.15:
            n_missing = int(round((gap_at_start - avg_gap) / expected_spacing))
            n_missing = max(1, n_missing)
            debug_print(f"DEBUG: FOUND MISSING at START! n_missing={n_missing}")

            # METHOD 1: Try white blocks first
            white_blocks = find_white_blocks(line_mask[0:first_channel_start], 0)
            debug_print(f"DEBUG: White clusters at start = {len(white_blocks)}")

            last_confirmed_end = 0

            for m in range(n_missing):
                predicted_center = last_confirmed_end + avg_gap + (avg_height / 2)
                predicted_start = predicted_center - avg_height / 2
                predicted_end = predicted_center + avg_height / 2

                # Find candidates from white blocks
                candidate_clusters = []
                for ws, we in white_blocks:
                    if ws >= last_confirmed_end and ws < first_channel_start:
                        overlap = min(predicted_end, we) - max(predicted_start, ws)
                        if overlap > 0:
                            candidate_clusters.append((ws, we))

                if candidate_clusters:
                    cluster_start = min(ws for ws, _ in candidate_clusters)
                    cluster_end = max(we for _, we in candidate_clusters)
                    total_height = cluster_end - cluster_start

                    min_acceptable_span = avg_height * 0.5
                    max_acceptable_span = avg_height * 1.8

                    if min_acceptable_span <= total_height <= max_acceptable_span:
                        missing_channels.append((cluster_start, cluster_end))
                        debug_print(f"DEBUG: Added missing channel at START (from {len(candidate_clusters)} clusters): {cluster_start}-{cluster_end}, height={total_height:.1f}")
                        last_confirmed_end = cluster_end
                    else:
                        debug_print(f"DEBUG: SKIPPED at START - clusters too small (height={total_height:.1f}), trying edge pairs")
                        # METHOD 2: Try edge pairs in predicted region
                        region_start = max(0, int(predicted_start - avg_height * 0.5))
                        region_end = min(first_channel_start, int(predicted_end + avg_height * 0.5))

                        edges_in_region = [e for e in grouped_all_edges if region_start <= e <= region_end]
                        debug_print(f"DEBUG: Found {len(edges_in_region)} edges in start region: {edges_in_region}")

                        if len(edges_in_region) >= 2:
                            # Use first and last edge as channel boundaries
                            edge_start = edges_in_region[0]
                            edge_end = edges_in_region[-1]
                            edge_height = edge_end - edge_start

                            if avg_height * 0.6 <= edge_height <= avg_height * 1.5:
                                missing_channels.append((edge_start, edge_end))
                                debug_print(f"DEBUG: Added missing channel at START (from edges): {edge_start}-{edge_end}, height={edge_height}")
                                last_confirmed_end = edge_end
                            else:
                                debug_print(f"DEBUG: Edge height {edge_height} out of range, skipping")
                                break
                        else:
                            debug_print(f"DEBUG: Not enough edges, likely device gap")
                            break
                else:
                    debug_print(f"DEBUG: No white clusters at START, trying edge pairs")
                    # METHOD 2: Try edge pairs in predicted region
                    region_start = max(0, int(predicted_start - avg_height * 0.5))
                    region_end = min(first_channel_start, int(predicted_end + avg_height * 0.5))

                    edges_in_region = [e for e in grouped_all_edges if region_start <= e <= region_end]
                    debug_print(f"DEBUG: Found {len(edges_in_region)} edges in start region: {edges_in_region}")

                    if len(edges_in_region) >= 2:
                        # Use first and last edge as channel boundaries
                        edge_start = edges_in_region[0]
                        edge_end = edges_in_region[-1]
                        edge_height = edge_end - edge_start

                        if avg_height * 0.6 <= edge_height <= avg_height * 1.5:
                            missing_channels.append((edge_start, edge_end))
                            debug_print(f"DEBUG: Added missing channel at START (from edges): {edge_start}-{edge_end}, height={edge_height}")
                            last_confirmed_end = edge_end
                        else:
                            debug_print(f"DEBUG: Edge height {edge_height} out of range, skipping")
                            break
                    else:
                        debug_print(f"DEBUG: Not enough edges, likely device gap")
                        break
                    
        # CHECK FOR MISSING CHANNELS AT THE END
        last_channel_end = channels[-1][1]
        gap_at_end = H - last_channel_end

        debug_print(f"DEBUG: Gap at end: {gap_at_end}, threshold={expected_spacing * 1.15:.1f}")

        if gap_at_end > expected_spacing * 1.15:
            n_missing = int(round((gap_at_end - avg_gap) / expected_spacing))
            n_missing = max(1, n_missing)
            debug_print(f"DEBUG: FOUND MISSING at END! n_missing={n_missing}")

            # METHOD 1: Try white blocks first
            white_blocks = find_white_blocks(line_mask[last_channel_end:H], last_channel_end)
            debug_print(f"DEBUG: White clusters at end = {len(white_blocks)}")

            last_confirmed_end = last_channel_end

            for m in range(n_missing):
                predicted_center = last_confirmed_end + avg_gap + (avg_height / 2)
                predicted_start = predicted_center - avg_height / 2
                predicted_end = predicted_center + avg_height / 2

                # Don't go beyond image bounds
                predicted_end = min(predicted_end, H)

                # Find candidates from white blocks
                candidate_clusters = []
                for ws, we in white_blocks:
                    if ws >= last_confirmed_end:
                        overlap = min(predicted_end, we) - max(predicted_start, ws)
                        if overlap > 0:
                            candidate_clusters.append((ws, we))

                if candidate_clusters:
                    cluster_start = min(ws for ws, _ in candidate_clusters)
                    cluster_end = max(we for _, we in candidate_clusters)
                    total_height = cluster_end - cluster_start

                    min_acceptable_span = avg_height * 0.5
                    max_acceptable_span = avg_height * 1.8

                    if min_acceptable_span <= total_height <= max_acceptable_span:
                        missing_channels.append((cluster_start, cluster_end))
                        debug_print(f"DEBUG: Added missing channel at END (from {len(candidate_clusters)} clusters): {cluster_start}-{cluster_end}, height={total_height:.1f}")
                        last_confirmed_end = cluster_end
                    else:
                        debug_print(f"DEBUG: SKIPPED at END - clusters too small (height={total_height:.1f}), trying edge pairs")
                        # METHOD 2: Try edge pairs in predicted region
                        region_start = max(last_channel_end, int(predicted_start - avg_height * 0.5))
                        region_end = min(H, int(predicted_end + avg_height * 0.5))

                        edges_in_region = [e for e in grouped_all_edges if region_start <= e <= region_end]
                        debug_print(f"DEBUG: Found {len(edges_in_region)} edges in end region: {edges_in_region}")

                        if len(edges_in_region) >= 2:
                            # Use first and last edge as channel boundaries
                            edge_start = edges_in_region[0]
                            edge_end = edges_in_region[-1]
                            edge_height = edge_end - edge_start

                            if avg_height * 0.6 <= edge_height <= avg_height * 1.5:
                                missing_channels.append((edge_start, edge_end))
                                debug_print(f"DEBUG: Added missing channel at END (from edges): {edge_start}-{edge_end}, height={edge_height}")
                                last_confirmed_end = edge_end
                            else:
                                debug_print(f"DEBUG: Edge height {edge_height} out of range, skipping")
                                break
                        else:
                            debug_print(f"DEBUG: Not enough edges, likely device gap")
                            break
                else:
                    debug_print(f"DEBUG: No white clusters at END, trying edge pairs")
                    # METHOD 2: Try edge pairs in predicted region
                    region_start = max(last_channel_end, int(predicted_start - avg_height * 0.5))
                    region_end = min(H, int(predicted_end + avg_height * 0.5))

                    edges_in_region = [e for e in grouped_all_edges if region_start <= e <= region_end]
                    debug_print(f"DEBUG: Found {len(edges_in_region)} edges in end region: {edges_in_region}")

                    if len(edges_in_region) >= 2:
                        # Use first and last edge as channel boundaries
                        edge_start = edges_in_region[0]
                        edge_end = edges_in_region[-1]
                        edge_height = edge_end - edge_start

                        if avg_height * 0.6 <= edge_height <= avg_height * 1.5:
                            missing_channels.append((edge_start, edge_end))
                            debug_print(f"DEBUG: Added missing channel at END (from edges): {edge_start}-{edge_end}, height={edge_height}")
                            last_confirmed_end = edge_end
                        else:
                            debug_print(f"DEBUG: Edge height {edge_height} out of range, skipping")
                            break
                    else:
                        debug_print(f"DEBUG: Not enough edges, likely device gap")
                        break

        # CHECK FOR MISSING CHANNELS BETWEEN EXISTING CHANNELS
        for i in range(len(channels) - 1):
            cur_start, cur_end = channels[i]
            next_start, next_end = channels[i + 1]

            # Actual gap size
            actual_gap = next_start - cur_end

            # Debug: print each gap
            debug_print(f"DEBUG: Gap between ch{i+1} and ch{i+2}: actual_gap={actual_gap}, threshold={expected_spacing * 1.15:.1f}")

            # If the gap is large enough to possibly contain missing channels
            if actual_gap > expected_spacing * 1.15:
                n_missing = int(round((actual_gap - avg_gap) / expected_spacing))
                n_missing = max(1, n_missing)
                debug_print(f"DEBUG: FOUND MISSING! n_missing={n_missing}")

                # Find white blocks in the gap region
                white_blocks = find_white_blocks(line_mask[cur_end:next_start], cur_end)
                debug_print(f"DEBUG: White clusters found = {len(white_blocks)} in this gap")

                last_confirmed_end = cur_end

                for m in range(n_missing):
                    # Calculate prediction from the LAST confirmed channel
                    predicted_center = last_confirmed_end + avg_gap + (avg_height / 2)
                    predicted_start = predicted_center - avg_height / 2
                    predicted_end = predicted_center + avg_height / 2

                    debug_print(f"DEBUG: Looking for missing channel {m+1}/{n_missing} starting from y={last_confirmed_end}")

                    # Find ALL white clusters that overlap with predicted channel region
                    candidate_clusters = []
                    for ws, we in white_blocks:
                        # Only consider clusters that are AFTER the last confirmed channel
                        if ws >= last_confirmed_end and ws < next_start:
                            overlap = min(predicted_end, we) - max(predicted_start, ws)
                            if overlap > 0:
                                candidate_clusters.append((ws, we))

                    if candidate_clusters:
                        # Use the full span from first to last cluster
                        cluster_start = min(ws for ws, _ in candidate_clusters)
                        cluster_end = max(we for _, we in candidate_clusters)

                        # Validate the TOTAL span is reasonable for a channel
                        total_height = cluster_end - cluster_start
                        min_acceptable_span = avg_height * 0.5
                        max_acceptable_span = avg_height * 1.8

                        if min_acceptable_span <= total_height <= max_acceptable_span:
                            missing_channels.append((cluster_start, cluster_end))
                            debug_print(f"DEBUG: Added missing channel (from {len(candidate_clusters)} clusters): {cluster_start}-{cluster_end}, height={total_height:.1f}")

                            # KEY: Update reference point to this newly found channel
                            last_confirmed_end = cluster_end
                        else:
                            # Total span is unreasonable, fall back to predicted
                            missing_channels.append((int(predicted_start), int(predicted_end)))
                            debug_print(f"DEBUG: Added missing channel (predicted, span {total_height:.1f} out of range): {int(predicted_start)}-{int(predicted_end)}")
                            last_confirmed_end = int(predicted_end)
                    else:
                        # No clusters found in predicted region, use predicted range
                        missing_channels.append((int(predicted_start), int(predicted_end)))
                        debug_print(f"DEBUG: Added missing channel (no clusters found, using predicted): {int(predicted_start)}-{int(predicted_end)}")
                        last_confirmed_end = int(predicted_end)

        # Add missing channels to the main list
        if missing_channels:
            channels.extend(missing_channels)
            channels = sorted(channels, key=lambda x: x[0])

    # 6. Create masks
    masks = []
    full_mask_img = np.zeros((H, W), dtype=np.uint8)
    for y_start, y_end in channels:
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y_start:y_end, :] = 1
        masks.append(mask)
        full_mask_img[y_start:y_end, :] = 255

    return channels, masks, full_mask_img, pre_mask, line_mask

#Debugging images displays each mask stage in a separate image
def visualize_channels(OV_img, channels, mask_img, pre_mask=None, line_mask=None,
                       ROI_x=0, ROI_y=0, ROI_w=None, ROI_h=None, 
                       window_name="4",
                       green_img=None, red_img=None,
                       save_dir=None, prefix="replicate"):
    """
    Visualize images with detected channel boundaries:
      - OV_img, mask_img, pre_mask, line_mask (existing)
      - Optional green_img and red_img
    """
    if ROI_w is None or ROI_h is None:
        ROI_h, ROI_w = mask_img.shape

    # -------------------------
    # 1. Crop ROI
    # -------------------------
    raw_ROI = OV_img[ROI_y:ROI_y+ROI_h, ROI_x:ROI_x+ROI_w]
    display_mask = mask_img[ROI_y:ROI_y+ROI_h, ROI_x:ROI_x+ROI_w]

    if pre_mask is not None:
        pre_mask_ROI = pre_mask[ROI_y:ROI_y+ROI_h, ROI_x:ROI_x+ROI_w]
    if line_mask is not None:
        line_mask_ROI = line_mask[ROI_y:ROI_y+ROI_h, ROI_x:ROI_x+ROI_w]

    # Convert to BGR for visualization if grayscale
    raw_display = cv2.cvtColor(raw_ROI, cv2.COLOR_GRAY2BGR) if raw_ROI.ndim == 2 else raw_ROI.copy()
    mask_display = cv2.cvtColor(display_mask, cv2.COLOR_GRAY2BGR)
    if pre_mask is not None:
        pre_mask_display = cv2.cvtColor(pre_mask_ROI, cv2.COLOR_GRAY2BGR)
    if line_mask is not None:
        line_mask_display = cv2.cvtColor(line_mask_ROI, cv2.COLOR_GRAY2BGR)

    # -------------------------
    # 2. Draw channel boundaries on mask AND full original image
    # -------------------------
    original_with_lines = cv2.cvtColor(OV_img.copy(), cv2.COLOR_GRAY2BGR) if OV_img.ndim == 2 else OV_img.copy()
    for idx, (y_start, y_end) in enumerate(channels):
        # On mask
        cv2.line(mask_display, (0, y_start), (ROI_w-1, y_start), (255, 0, 0), 2)
        cv2.line(mask_display, (0, y_end), (ROI_w-1, y_end), (0, 0, 255), 2)
        cv2.putText(mask_display, f'Ch{idx+1}', (10, y_start+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # On full OV_img
        cv2.line(original_with_lines, (ROI_x, ROI_y + y_start), (ROI_x + ROI_w - 1, ROI_y + y_start), (255, 0, 0), 2)
        cv2.line(original_with_lines, (ROI_x, ROI_y + y_end), (ROI_x + ROI_w - 1, ROI_y + y_end), (0, 0, 255), 2)
        cv2.putText(original_with_lines, f'Ch{idx+1}', (ROI_x + 10, ROI_y + y_start + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # -------------------------
    # 3. Resize images to same height
    # -------------------------
    target_height = 700
    padding_width = 10  # Add 100 pixels on each side (200 total extra width)
    
    def resize_img(img):
        scale = target_height / img.shape[0]
        target_width = int(img.shape[1] * scale)
        resized = cv2.resize(img, (target_width, target_height))
        # Add black padding on left and right
        padded = cv2.copyMakeBorder(resized, 0, 0, padding_width, padding_width, 
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded

    raw_display = resize_img(raw_display)
    mask_display = resize_img(mask_display)
    original_with_lines_resized = resize_img(original_with_lines)
    if pre_mask is not None:
        pre_mask_display = resize_img(pre_mask_display)
    if line_mask is not None:
        line_mask_display = resize_img(line_mask_display)

    # -------------------------
    # 4. Show popups for OV images
    # -------------------------
    cv2.imshow("1", raw_display)
    if pre_mask is not None:
        cv2.imshow("2", pre_mask_display)
    if line_mask is not None:
        cv2.imshow("3", line_mask_display)
    cv2.imshow(window_name, mask_display) #4
    cv2.imshow("5", original_with_lines_resized)

    # -------------------------
    # 5. Optional: Show Green and Red Images
    # -------------------------
    def show_color_image(color_img, color_name):
        if color_img is None:
            return None
        ROI = color_img[ROI_y:ROI_y+ROI_h, ROI_x:ROI_x+ROI_w]
        display = cv2.cvtColor(ROI, cv2.COLOR_GRAY2BGR) if ROI.ndim == 2 else ROI.copy()
        for idx, (y_start, y_end) in enumerate(channels):
            cv2.line(display, (0, y_start), (ROI_w-1, y_start), (255, 0, 0), 2)
            cv2.line(display, (0, y_end), (ROI_w-1, y_end), (0, 0, 255), 2)
            cv2.putText(display, f'Ch{idx+1}', (5, y_start+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        display = resize_img(display)
        cv2.imshow(f"{color_name}", display)
        cv2.imshow(f"{color_name}", display) 
        return display   

    green_disp = show_color_image(green_img, "Green")
    red_disp = show_color_image(red_img, "Red")

    # -------------------------
    # 6. Print channel info
    # -------------------------
    print(f"\nDetected {len(channels)} channels:")
    for idx, (y_start, y_end) in enumerate(channels):
        print(f"  Channel {idx+1}: rows {y_start}-{y_end} (height: {y_end-y_start})")

    # -------------------------
    # Save debug images if requested
    # -------------------------
    if save_dir is not None:

        os.makedirs(save_dir, exist_ok=True)

        def _save(name, img):
            if img is None:
                return
            out_path = os.path.join(save_dir, f"{prefix}_{name}.png")
            cv2.imwrite(out_path, img)

        _save("1_raw", raw_display)
        if pre_mask is not None:
            _save("2_pre_mask", pre_mask_display)
        if line_mask is not None:
            _save("3_line_mask", line_mask_display)

        _save("4_mask", mask_display)
        _save("5_channels", original_with_lines_resized)

        _save("6_green", green_disp)
        _save("7_red", red_disp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Create quiry popup for num of replicates
def get_num_replicates():
    result = {
        "num": None, 
        "partitions": None, 
        "debug": False, 
        "device": "standard",
        "free_roi": False,
        "roi_width": 25,
        "roi_height":115
    }
#4stack: 25, 115
#2stack: 40, 175
#single: 50, 215

    def open_settings():
        """Open advanced settings popup"""
        settings_window = tk.Toplevel(window)
        settings_window.title("Advanced Settings")
        settings_window.geometry("480x420")
        settings_window.resizable(False, False)
        settings_window.transient(window)
        settings_window.grab_set()

        # Debug checkbox
        debug_var = tk.BooleanVar(value=result["debug"])
        debug_checkbox = tk.Checkbutton(settings_window, text="Enable Debug Mode (show visualizations)", 
                                         variable=debug_var, font=("Arial", 11))
        debug_checkbox.pack(pady=15, padx=20, anchor=tk.W)

        # Free ROI selection checkbox
        free_roi_var = tk.BooleanVar(value=result["free_roi"])
        
        def toggle_roi_fields():
            state = 'disabled' if free_roi_var.get() else 'normal'
            roi_width_entry.config(state=state)
            roi_height_entry.config(state=state)

        free_roi_checkbox = tk.Checkbutton(settings_window, 
                                           text="Enable drag/free ROI selection (instead of fixed size)", 
                                           variable=free_roi_var, 
                                           font=("Arial", 11),
                                           command=toggle_roi_fields)
        free_roi_checkbox.pack(pady=10, padx=20, anchor=tk.W)

        # ROI dimensions frame
        roi_frame = tk.LabelFrame(settings_window, text="Fixed ROI Dimensions (disabled when drag is enabled)", 
                                   font=("Arial", 10, "bold"), padx=20, pady=15)
        roi_frame.pack(pady=15, padx=20, fill=tk.X)

        # ROI Width
        width_label = tk.Label(roi_frame, text="ROI Width (pixels):", font=("Arial", 10))
        width_label.pack(anchor=tk.W, pady=5)
        roi_width_var = tk.StringVar(value=str(result["roi_width"]))
        roi_width_entry = tk.Entry(roi_frame, textvariable=roi_width_var, font=("Arial", 11), width=20)
        roi_width_entry.pack(anchor=tk.W, pady=5)

        # ROI Height
        height_label = tk.Label(roi_frame, text="ROI Height (pixels):", font=("Arial", 10))
        height_label.pack(anchor=tk.W, pady=5)
        roi_height_var = tk.StringVar(value=str(result["roi_height"]))
        roi_height_entry = tk.Entry(roi_frame, textvariable=roi_height_var, font=("Arial", 11), width=20)
        roi_height_entry.pack(anchor=tk.W, pady=5)

        # Initialize ROI field states
        toggle_roi_fields()

        # Message label for errors
        message_label = tk.Label(settings_window, text="", fg="red", font=("Arial", 10))
        message_label.pack(pady=5)

        # Save button
        def save_settings():
            try:
                roi_w = int(roi_width_var.get())
                roi_h = int(roi_height_var.get())
                if roi_w < 1:
                    message_label.config(text="ROI Width must be positive.")
                    return
                if roi_h < 1:
                    message_label.config(text="ROI Height must be positive.")
                    return
                
                result["debug"] = debug_var.get()
                result["free_roi"] = free_roi_var.get()
                result["roi_width"] = roi_w
                result["roi_height"] = roi_h
                settings_window.destroy()
            except ValueError:
                message_label.config(text="ROI dimensions must be integers.")

        save_button = tk.Button(settings_window, text="Save Settings", font=("Arial", 11, "bold"),
                               bg="#4CAF50", fg="white", command=save_settings, padx=20, pady=10)
        save_button.pack(pady=15)

        settings_window.wait_window()

    def submit(event=None):
        rep_val = rep_entry.get()
        part_val = part_entry.get()
        try:
            rep_int = int(rep_val)
            part_int = int(part_val)
            if rep_int < 1:
                message_label.config(text="Replicates must be a positive number.")
            elif part_int < 1:
                message_label.config(text="Partitions must be a positive number.")
            else:
                result["num"] = rep_int
                result["partitions"] = part_int
                result["device"] = device_var.get()
                window.destroy()
        except ValueError:
            message_label.config(text="Invalid input. Enter integers only.")

    def on_close():
        result["num"] = 0
        result["partitions"] = 5
        window.destroy()

    window = tk.Toplevel(root)
    window.title("Analysis Settings")
    window.geometry("400x380")
    window.resizable(False, False)

    # Replicates input
    rep_instruction = tk.Label(window, text="Enter the number of replicates:", font=("Arial", 12))
    rep_instruction.pack(pady=(20, 5))

    rep_entry = tk.Entry(window, font=("Arial", 14))
    rep_entry.pack(pady=5)
    rep_entry.focus_set()

    # Partitions input
    part_instruction = tk.Label(window, text="Enter the number of partitions per channel:", font=("Arial", 12))
    part_instruction.pack(pady=(15, 5))

    part_entry = tk.Entry(window, font=("Arial", 14))
    part_entry.insert(0, "5")
    part_entry.pack(pady=5)

    # Device selection
    device_instruction = tk.Label(window, text="Select device type:", font=("Arial", 12))
    device_instruction.pack(pady=(15, 5))

    device_var = tk.StringVar(value="standard")
    device_frame = tk.Frame(window)
    device_frame.pack(pady=5)

    standard_radio = tk.Radiobutton(device_frame, text="Standard Device (17-30px channels)", 
                                    variable=device_var, value="standard", font=("Arial", 10))
    standard_radio.pack(anchor=tk.W, padx=20)

    thin_radio = tk.Radiobutton(device_frame, text="Thin Device (smaller channels)", 
                                variable=device_var, value="thin", font=("Arial", 10))
    thin_radio.pack(anchor=tk.W, padx=20)

    # Settings button
    settings_button = tk.Button(window, text="⚙ Advanced Settings", font=("Arial", 11, "bold"),
                               bg="#2196F3", fg="white", command=open_settings, padx=15, pady=8)
    settings_button.pack(pady=15)

    # Submit instruction
    submit_instruction = tk.Label(window, text="Press Enter to continue", font=("Arial", 10), fg="gray")
    submit_instruction.pack(pady=10)

    message_label = tk.Label(window, text="", fg="red", font=("Arial", 12))
    message_label.pack(pady=5)

    # Bind Enter key to submit
    rep_entry.bind("<Return>", submit)
    part_entry.bind("<Return>", submit)

    # Handle window close
    window.protocol("WM_DELETE_WINDOW", on_close)

    window.wait_window()
    return result["num"], result["partitions"], result["debug"], result["device"], result["free_roi"], result["roi_width"], result["roi_height"]

num_replicates, num_partitions, debug_mode, selected_device, free_roi_mode, custom_roi_w, custom_roi_h = get_num_replicates()
device_params = DEVICE_CONFIG[selected_device]
# Use custom ROI dimensions if provided, otherwise use device defaults
ROI_W = custom_roi_w if not free_roi_mode else device_params['roi_width']
ROI_H = custom_roi_h if not free_roi_mode else device_params['roi_height']
debug_print(f"Using {device_params['name']}...")
debug_print(f"ROI Mode: {'Free/Drag' if free_roi_mode else f'Fixed ({ROI_W}x{ROI_H})'}")
for i in range(num_replicates):  #For each replicate

    OV_img = cv2.imread(OV_imglist[0])

    if OV_img is None:
        raise FileNotFoundError(f"Could not read image: {OV_imglist[0]}")
    orig_h, orig_w = OV_img.shape[:2] #Returns height and width of image

    #Scales original image to display sized image
    scale_w = popup_width / orig_w
    scale_h = popup_height / orig_h
    scale = min(scale_w, scale_h)
    display_img = cv2.resize(OV_img, (int(orig_w * scale), int(orig_h * scale)))
    display_h, display_w = display_img.shape[:2]

    #Use globals for mouse callback
    global roi_x, roi_y, roi_x_end, roi_y_end, dragging, roi_width_current, roi_height_current
    
    if free_roi_mode:
        # Free drag mode: click and drag to define ROI
        roi_x = 0
        roi_y = 0
        roi_x_end = ROI_W
        roi_y_end = ROI_H
        dragging = False
        roi_width_current = ROI_W
        roi_height_current = ROI_H
    else:
        # Fixed size mode: drag box around
        roi_x = (display_w - ROI_W) // 2    #where the selection box starts
        roi_y = display_h // 8              #where the selection box starts
        roi_width_current = ROI_W
        roi_height_current = ROI_H
        dragging = False

    #Use mouse to drag ROI selection box
    if free_roi_mode:
        # Free drag mode: click and drag to define ROI
        def mouse_callback(event, x, y, flags, param):
            global roi_x, roi_y, roi_x_end, roi_y_end, dragging, roi_width_current, roi_height_current
            
            if event == cv2.EVENT_LBUTTONDOWN:
                roi_x = x
                roi_y = y
                roi_x_end = x
                roi_y_end = y
                dragging = True
            elif event == cv2.EVENT_MOUSEMOVE and dragging:
                roi_x_end = max(0, min(x, display_w))
                roi_y_end = max(0, min(y, display_h))
            elif event == cv2.EVENT_LBUTTONUP:
                dragging = False
                # Ensure coordinates are in correct order
                roi_x, roi_x_end = min(roi_x, roi_x_end), max(roi_x, roi_x_end)
                roi_y, roi_y_end = min(roi_y, roi_y_end), max(roi_y, roi_y_end)
                # Update width and height
                roi_width_current = roi_x_end - roi_x
                roi_height_current = roi_y_end - roi_y
    else:
        # Fixed size mode: drag box around
        def mouse_callback(event, x, y, flags, param):
            global roi_x, roi_y, dragging
            if event == cv2.EVENT_LBUTTONDOWN:
                if roi_x <= x <= roi_x + ROI_W and roi_y <= y <= roi_y + ROI_H:
                    dragging = True
            elif event == cv2.EVENT_LBUTTONUP:
                dragging = False
            elif event == cv2.EVENT_MOUSEMOVE and dragging:
                roi_x = max(0, min(x - ROI_W // 2, display_w - ROI_W))
                roi_y = max(0, min(y - ROI_H // 2, display_h - ROI_H))

    cv2.namedWindow(f"Set ROI Replicate {i + 1}", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(f"Set ROI Replicate {i + 1}", mouse_callback)
    cv2.resizeWindow(f"Set ROI Replicate {i + 1}", popup_width, popup_height)
    
    # Create magnified preview window
    magnify_window = f"ROI Preview (Magnified) - Replicate {i + 1}"
    cv2.namedWindow(magnify_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(magnify_window, 150, 700) #Magnification Box size

#Selection process of ROI repeated until 'Enter' or 'Esc'
    while True:
        display = display_img.copy()
        
        if free_roi_mode and dragging:
            # Draw rectangle being dragged
            cv2.rectangle(display, (roi_x, roi_y), (roi_x_end, roi_y_end), (0, 255, 0), 2)
            # Fill with semi-transparent green to show selection
            overlay = display.copy()
            cv2.rectangle(overlay, (roi_x, roi_y), (roi_x_end, roi_y_end), (0, 255, 0), -1)
            display = cv2.addWeighted(overlay, 0.2, display, 0.8, 0)
            cv2.rectangle(display, (roi_x, roi_y), (roi_x_end, roi_y_end), (0, 255, 0), 2)
        else:
            # Fixed size box or idle state in free mode
            cv2.rectangle(display, (roi_x, roi_y), (roi_x + roi_width_current, roi_y + roi_height_current), (0, 255, 0), 2)
        
        cv2.imshow(f"Set ROI Replicate {i + 1}", display)
        
        # Show magnified ROI from original full-resolution image
        if free_roi_mode:
            ROI_x_temp = int(roi_x / scale)
            ROI_y_temp = int(roi_y / scale)
            ROI_w_temp = int(roi_width_current / scale)
            ROI_h_temp = int(roi_height_current / scale)
        else:
            ROI_x_temp = int(roi_x / scale)
            ROI_y_temp = int(roi_y / scale)
            ROI_w_temp = int(ROI_W / scale)
            ROI_h_temp = int(ROI_H / scale)
        
        # Clamp to image bounds
        ROI_x_temp = max(0, min(ROI_x_temp, orig_w - 1))
        ROI_y_temp = max(0, min(ROI_y_temp, orig_h - 1))
        ROI_w_temp = min(ROI_w_temp, orig_w - ROI_x_temp)
        ROI_h_temp = min(ROI_h_temp, orig_h - ROI_y_temp)
        
        if ROI_w_temp > 0 and ROI_h_temp > 0:
            roi_preview = OV_img[ROI_y_temp:ROI_y_temp+ROI_h_temp, ROI_x_temp:ROI_x_temp+ROI_w_temp]
            if roi_preview.size > 0:
                magnified = cv2.resize(roi_preview, (400, 600), interpolation=cv2.INTER_LINEAR)
                cv2.imshow(magnify_window, magnified)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # 'Enter' locks the ROI
            break
        elif key == 27:  # 'Esc' cancels
            roi_x, roi_y, roi_width_current, roi_height_current = 0, 0, 0, 0
            break

    cv2.destroyAllWindows()

# Scale ROI coordinates back to original image
    if free_roi_mode:
        ROI_x = int(roi_x / scale)
        ROI_y = int(roi_y / scale)
        ROI_w = int(roi_width_current / scale)
        ROI_h = int(roi_height_current / scale)
    else:
        ROI_x = int(roi_x / scale)
        ROI_y = int(roi_y / scale)
        ROI_w = int(ROI_W / scale)
        ROI_h = int(ROI_H / scale)

    # ✨ NEW: Add retry loop for human validation
    user_approved = False
    
    while not user_approved:
        # Crop ROI from OV image
        ROI_img = OV_img[ROI_y:ROI_y+ROI_h, ROI_x:ROI_x+ROI_w]
        channels, masks, final_mask_img, pre_mask, line_mask = detect_channels(
            ROI_img, 
            min_height=device_params['min_height'],
            max_height=device_params['max_height'],
            min_gap_height=device_params['min_gap_height'],
            device_channel_range=device_params['device_channel_range']
        )

        # ✨ NEW: If debug mode is OFF, show validation window
        if not debug_mode:
            # Create the preview image using CROPPED ROI only
            roi_with_lines = cv2.cvtColor(ROI_img.copy(), cv2.COLOR_GRAY2BGR) if ROI_img.ndim == 2 else ROI_img.copy()
            
            # Draw channel boundaries with translucent overlay
            overlay = roi_with_lines.copy()
            
            for idx, (y_start, y_end) in enumerate(channels):
                # Alternate between two colors for adjacent channels
                if idx % 2 == 0:
                    color = (0, 255, 0)  # Green
                else:
                    color = (0, 255, 255)  # Cyan
                
                # Fill channel with translucent color
                cv2.rectangle(overlay, (0, y_start), (ROI_w - 1, y_end), color, -1)
                
                # Add thin boundary lines for clarity
                cv2.line(overlay, (0, y_start), (ROI_w - 1, y_start), (255, 255, 255), 1)
                cv2.line(overlay, (0, y_end), (ROI_w - 1, y_end), (255, 255, 255), 1)
                
                # Add channel label centered horizontally and vertically
                label = f'Ch{idx+1}'
                
                # Get text dimensions
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Calculate center position (horizontal and vertical)
                label_x = (ROI_w - text_width) // 2  # Center horizontally
                label_y = (y_start + y_end) // 2 + text_height // 2  # Center vertically
                
                # Black outline for text visibility
                cv2.putText(overlay, label, (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                # White text on top
                cv2.putText(overlay, label, (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Blend overlay with original (30% overlay, 70% original)
            roi_with_lines = cv2.addWeighted(overlay, 0.3, roi_with_lines, 0.7, 0)
            
            # Resize for display
            target_height = 650
            scale_preview = target_height / roi_with_lines.shape[0]
            target_width = int(roi_with_lines.shape[1] * scale_preview)
            preview_resized = cv2.resize(roi_with_lines, (target_width, target_height))
            
            # Convert from BGR to RGB for Tkinter
            preview_rgb = cv2.cvtColor(preview_resized, cv2.COLOR_BGR2RGB)
            from PIL import Image, ImageTk
            pil_image = Image.fromarray(preview_rgb)
            
            # Create validation window
            validation_result = {"approved": None}
            
            def on_accept():
                validation_result["approved"] = True
                validation_window.destroy()
            
            def on_retry():
                validation_result["approved"] = False
                validation_window.destroy()
            
            validation_window = tk.Toplevel(root)
            validation_window.title(f"Validate Channels - Replicate {i + 1}")
            validation_window.geometry(f"{target_width + 350}x{target_height}")
            validation_window.resizable(False, False)
            validation_window.attributes('-topmost', True)

            # Main container frame
            main_frame = tk.Frame(validation_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # LEFT SIDE: Title and Buttons
            left_frame = tk.Frame(main_frame)
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=10)

            # Title
            title_label = tk.Label(left_frame, 
                                   text=f"Detected {len(channels)} channels.\n\nIs the mask correct?\n\nIf not, try adjusting\nselection", 
                                   font=("Arial", 11, "bold"), wraplength=180, justify=tk.CENTER)
            title_label.pack(pady=20)

            # Buttons
            accept_button = tk.Button(left_frame, text="✓ Accept\n(Enter)", 
                                     font=("Arial", 11, "bold"), bg="#4CAF50", fg="white",
                                     command=on_accept, padx=15, pady=15)
            accept_button.pack(pady=10, fill=tk.X)

            retry_button = tk.Button(left_frame, text="↻ Try Again", 
                                    font=("Arial", 11), bg="#FF9800", fg="white",
                                    command=on_retry, padx=15, pady=15)
            retry_button.pack(pady=10, fill=tk.X)

            # RIGHT SIDE: Image
            right_frame = tk.Frame(main_frame)
            right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

            # Image
            tk_image = ImageTk.PhotoImage(pil_image)
            image_label = tk.Label(right_frame, image=tk_image)
            image_label.image = tk_image  # Keep a reference
            image_label.pack(padx=10)

            # Bind Enter key to accept
            validation_window.bind("<Return>", lambda e: on_accept())

            validation_window.wait_window()
            
            if validation_result["approved"]:
                user_approved = True
            else:
                # User wants to retry - reopen ROI selection
                cv2.namedWindow(f"Set ROI Replicate {i + 1}", cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(f"Set ROI Replicate {i + 1}", mouse_callback)
                cv2.resizeWindow(f"Set ROI Replicate {i + 1}", popup_width, popup_height)
                
                magnify_window = f"ROI Preview (Magnified) - Replicate {i + 1}"
                cv2.namedWindow(magnify_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(magnify_window, 150, 700)
                
                while True:
                    display = display_img.copy()
                    
                    if free_roi_mode and dragging:
                        # Draw rectangle being dragged
                        cv2.rectangle(display, (roi_x, roi_y), (roi_x_end, roi_y_end), (0, 255, 0), 2)
                        # Fill with semi-transparent green to show selection
                        overlay = display.copy()
                        cv2.rectangle(overlay, (roi_x, roi_y), (roi_x_end, roi_y_end), (0, 255, 0), -1)
                        display = cv2.addWeighted(overlay, 0.2, display, 0.8, 0)
                        cv2.rectangle(display, (roi_x, roi_y), (roi_x_end, roi_y_end), (0, 255, 0), 2)
                    else:
                        # Fixed size box or idle state in free mode
                        cv2.rectangle(display, (roi_x, roi_y), (roi_x + roi_width_current, roi_y + roi_height_current), (0, 255, 0), 2)
                    
                    cv2.imshow(f"Set ROI Replicate {i + 1}", display)
                    
                    # Show magnified ROI from original full-resolution image
                    if free_roi_mode:
                        ROI_x_temp = int(roi_x / scale)
                        ROI_y_temp = int(roi_y / scale)
                        ROI_w_temp = int(roi_width_current / scale)
                        ROI_h_temp = int(roi_height_current / scale)
                    else:
                        ROI_x_temp = int(roi_x / scale)
                        ROI_y_temp = int(roi_y / scale)
                        ROI_w_temp = int(ROI_W / scale)
                        ROI_h_temp = int(ROI_H / scale)
                    
                    # Clamp to image bounds
                    ROI_x_temp = max(0, min(ROI_x_temp, orig_w - 1))
                    ROI_y_temp = max(0, min(ROI_y_temp, orig_h - 1))
                    ROI_w_temp = min(ROI_w_temp, orig_w - ROI_x_temp)
                    ROI_h_temp = min(ROI_h_temp, orig_h - ROI_y_temp)
                    
                    if ROI_w_temp > 0 and ROI_h_temp > 0:
                        roi_preview = OV_img[ROI_y_temp:ROI_y_temp+ROI_h_temp, ROI_x_temp:ROI_x_temp+ROI_w_temp]
                        if roi_preview.size > 0:
                            magnified = cv2.resize(roi_preview, (400, 600), interpolation=cv2.INTER_LINEAR)
                            cv2.imshow(magnify_window, magnified)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == 13:
                        break
                    elif key == 27:
                        roi_x, roi_y, roi_width_current, roi_height_current = 0, 0, 0, 0
                        if free_roi_mode:
                            roi_x_end, roi_y_end = 0, 0
                        break
                
                cv2.destroyAllWindows()
                
                # Update ROI coordinates
                if free_roi_mode:
                    ROI_x = int(roi_x / scale)
                    ROI_y = int(roi_y / scale)
                    ROI_w = int(roi_width_current / scale)
                    ROI_h = int(roi_height_current / scale)
                else:
                    ROI_x = int(roi_x / scale)
                    ROI_y = int(roi_y / scale)
                    ROI_w = int(ROI_W / scale)
                    ROI_h = int(ROI_H / scale)
        else:
            # Debug mode is ON - skip validation
            user_approved = True

    # Load full color images
    green_full = cv2.imread(green_imglist[0])[:, :, 1]
    red_full = cv2.imread(red_imglist[0])[:, :, 2]
    green_crop = green_full[ROI_y:ROI_y+ROI_h, ROI_x:ROI_x+ROI_w]
    red_crop = red_full[ROI_y:ROI_y+ROI_h, ROI_x:ROI_x+ROI_w]

    # ============================================================
    # COLOR-BASED THRESHOLDS (instead of Otsu)
    # ============================================================
    # Define thresholds based on hex colors
    GREEN_THRESHOLDS = {
        'reject': 10,      # #011500 - Too dark (autofluorescence)
        'minimum': 40,     # #004800 - Minimum low value for green
        'medium': 145,     # #039100 - Medium/ideal value for green
        'high': 250        # #00FA00 - High green
    }

    RED_THRESHOLDS = {
        'reject': 10,      # #220000 - Too dark (autofluorescence)
        'minimum': 110,     # #510000 - Minimum low value for red
        'medium': 145,     # #730000 - Medium/ideal value for red
        'high': 250        # #FA0100 - High red
    }

    # Use 'minimum' as the threshold to count positive pixels
    green_threshold = GREEN_THRESHOLDS['minimum']
    red_threshold = RED_THRESHOLDS['minimum']

    debug_print(f"Using color-based thresholds: Green >= {green_threshold}, Red >= {red_threshold}")

    debug_print(f"\nDEBUG: Detected channels for replicate {i+1}:")
    for idx, (start, end) in enumerate(channels, 1):
        channel_mask = (final_mask_img[start:end, :] > 0)
        channel_height = end - start

        total_channel_pixels = np.sum(channel_mask)

        # Green region - TOTAL (use green_threshold)
        green_region = green_crop[start:end, :]
        masked_green = green_region[channel_mask]

        if masked_green.size > 0:
            green_percent_total = np.sum(masked_green >= green_threshold) / masked_green.size * 100
        else:
            green_percent_total = 0.0

        # Red region - TOTAL (use red_threshold)
        red_region = red_crop[start:end, :]
        masked_red = red_region[channel_mask]

        if masked_red.size > 0:
            red_percent = np.sum(masked_red >= red_threshold) / masked_red.size * 100
        else:
            red_percent = 0.0

        # Calculate GREEN percentage for each partition (along WIDTH, not height)
        channel_width = green_crop.shape[1]  # Get the width
        partition_width = channel_width / num_partitions
        green_partitions = []

        for p in range(num_partitions):
            p_x_start = int(p * partition_width)
            p_x_end = int((p + 1) * partition_width) if p < num_partitions - 1 else channel_width

            # Get mask and green region for this HORIZONTAL partition
            p_mask = (final_mask_img[start:end, p_x_start:p_x_end] > 0)
            p_green_region = green_crop[start:end, p_x_start:p_x_end]
            p_masked_green = p_green_region[p_mask]

            if total_channel_pixels > 0:
                p_green_percent = np.sum(p_masked_green >= green_threshold) / total_channel_pixels * 100
            else:
                p_green_percent = 0.0

            green_partitions.append(p_green_percent)

        # Calculate RED percentage for each partition (same logic as green)
        red_partitions = []

        for p in range(num_partitions):
            p_x_start = int(p * partition_width)
            p_x_end = int((p + 1) * partition_width) if p < num_partitions - 1 else channel_width

            # Get mask and red region for this HORIZONTAL partition
            p_mask = (final_mask_img[start:end, p_x_start:p_x_end] > 0)
            p_red_region = red_crop[start:end, p_x_start:p_x_end]
            p_masked_red = p_red_region[p_mask]

            if total_channel_pixels > 0:
                p_red_percent = np.sum(p_masked_red >= red_threshold) / total_channel_pixels * 100
            else:
                p_red_percent = 0.0

            red_partitions.append(p_red_percent)

        debug_print(f"  Channel {idx}: start={start}, end={end}, height={channel_height}, "
              f"green_total%={green_percent_total:.2f}%, red_total%={red_percent:.2f}%")
        debug_print(f"    Green partitions: {[f'{p:.2f}%' for p in green_partitions]}")
        debug_print(f"    Red partitions:   {[f'{p:.2f}%' for p in red_partitions]}")

        # Save channel info into Excel with specific column order
        channel_info = {
            'Replicate': i+1,
            'Channel': idx,
            'Start': start,
            'End': end,
            'Height': channel_height,
            'Green % (Total)': green_percent_total
        }
    
        # Add green partition columns
        for p_idx, p_val in enumerate(green_partitions, 1):
            channel_info[f'Green % Part{p_idx}'] = p_val
    
        # Add red total
        channel_info['Red % (Total)'] = red_percent
    
        # Add red partition columns
        for p_idx, p_val in enumerate(red_partitions, 1):
            channel_info[f'Red % Part{p_idx}'] = p_val

        df_channels = pd.concat([df_channels, pd.DataFrame([channel_info])], ignore_index=True)


    if debug_mode:

        debug_folder = os.path.join(folder_path, "debug_images", f"replicate_{i+1}")

        visualize_channels(
            ROI_img,
            channels,
            final_mask_img,
            pre_mask=pre_mask,
            line_mask=line_mask,
            ROI_x=0,
            ROI_y=0,
            ROI_w=ROI_w,
            ROI_h=ROI_h,
            green_img=green_crop,
            red_img=red_crop,
            save_dir=debug_folder,
            prefix=f"rep{i+1}"
    )

# ============================================================
# WRITE EXCEL FILE WITH CHANNEL DATA AND DEBUG INFO
# ============================================================
print("\n" + "="*60)
print("Writing Excel file...")

with pd.ExcelWriter('Analysis.xlsx', engine='openpyxl') as writer:
    # Write each replicate to its own sheet
    for rep_num in range(1, num_replicates + 1):
        # Filter data for this replicate
        rep_data = df_channels[df_channels['Replicate'] == rep_num]
        
        # Write to sheet named "Replicate 1", "Replicate 2", etc.
        sheet_name = f'Replicate {rep_num}'
        rep_data.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
        
        # Format percentage columns
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        
        headers = list(rep_data.columns)
        percentage_cols = [i+1 for i, col in enumerate(headers) if '%' in col]
        
        for row in range(2, len(rep_data) + 2):
            for col_idx in percentage_cols:
                worksheet.cell(row=row, column=col_idx).number_format = '0.000"%"'
    
    # Add debug log to a separate sheet
    debug_sheet = workbook.create_sheet('Debug Log')
    debug_sheet.cell(row=1, column=1, value="DEBUG LOG:")
    
    for idx, msg in enumerate(debug_messages, start=1):
        debug_sheet.cell(row=idx + 1, column=1, value=msg)

print("Analysis complete! Data saved to 'Analysis.xlsx'")
print(f"- Created {num_replicates} replicate sheet(s)")
print(f"- Total channels detected: {len(df_channels)}")
print(f"- Debug log: {len(debug_messages)} messages captured")
print("="*60 + "\n")