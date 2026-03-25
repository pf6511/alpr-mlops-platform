"""
Video processing utilities for ALPR demo.
Handles video annotation and GIF processing.
"""

import cv2
import numpy as np
from PIL import Image
import tempfile
import os


def create_annotated_video(pipeline, video_path, output_path, conf_threshold=0.5, max_fps=10):
    """
    Create annotated video with license plate detections.
    
    Args:
        pipeline: ALPRPipeline instance
        video_path: Input video path
        output_path: Output video path
        conf_threshold: Detection confidence threshold
        max_fps: Maximum FPS to process (for performance)
        
    Returns:
        dict with statistics
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Limit FPS for performance
    process_fps = min(fps, max_fps)
    frame_skip = max(1, int(fps / process_fps))
    
    # Video writer - Smart codec selection
    # H.264 (avc1) is preferred for MP4 but often requires standard ffmpeg installation
    # VP9 (vp09) is preferred for WebM and works well in browsers
    # mp4v is a fallback for MP4
    
    fourcc_options = []
    
    if output_path.endswith('.webm'):
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'vp80'),  # VP8 (WebM) - Better compatibility than VP9
            cv2.VideoWriter_fourcc(*'vp09'),  # VP9 (WebM)
        ]
    else:
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'avc1'),  # H.264 (MP4) - Best if available
            cv2.VideoWriter_fourcc(*'H264'), 
            cv2.VideoWriter_fourcc(*'X264'),
            cv2.VideoWriter_fourcc(*'mp4v'),  # Fallback
        ]
    
    out = None
    used_codec = ""
    
    for fourcc in fourcc_options:
        try:
            temp_out = cv2.VideoWriter(output_path, fourcc, process_fps, (width, height))
            if temp_out.isOpened():
                out = temp_out
                # Get string representation of fourcc for logging
                # This is a bit hacky but works for standard tags
                print(f"  ✓ Using codec: {fourcc}")
                break
        except Exception as e:
            print(f"  ⚠️ Codec failed: {e}")
            continue
    
    if not out or not out.isOpened():
        # Last resort fallback if everything failed - try mp4v even if .webm (might fail but worth a try)
        # or just fail gracefully
        print("  ⚠️ All preferred codecs failed. Trying generic 'mp4v'...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, process_fps, (width, height))
        
    if not out.isOpened():
        raise RuntimeError(f"Failed to initialize video writer for {output_path}")
    
    frame_idx = 0
    processed_frames = 0
    processed_frames = 0
    total_detections = 0
    detected_plates = set()
    
    print(f"🎬 Processing video: {total_frames} frames @ {fps:.1f} FPS")
    print(f"📊 Output: {process_fps:.1f} FPS (skip every {frame_skip} frames)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames to match target FPS
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue
        
        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Save temp image
        temp_path = f"/tmp/frame_{frame_idx}.jpg"
        cv2.imwrite(temp_path, frame)
        
        # Process through pipeline
        results = pipeline.process_image(temp_path, conf_threshold)
        
        # Get annotated frame
        annotated_rgb = results['step5_final']
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        
        # Count detections
        num_detections = len(results['metadata']['detections'])
        num_detections = len(results['metadata']['detections'])
        total_detections += num_detections
        
        # Collect detected plates
        if results.get('step4_ocr'):
            for ocr in results['step4_ocr']:
                if ocr.get('text'):
                    detected_plates.add(ocr['text'])
        
        # Write frame
        out.write(annotated_bgr)
        
        # Cleanup
        os.remove(temp_path)
        
        processed_frames += 1
        if processed_frames % 30 == 0:
            print(f"  ✓ Processed {processed_frames} frames...")
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print(f"✅ Video processing complete!")
    print(f"   Processed: {processed_frames} frames")
    print(f"   Total detections: {total_detections}")
    
    return {
        'total_frames': total_frames,
        'processed_frames': processed_frames,
        'total_detections': total_detections,
        'detected_plates': list(detected_plates),
        'output_fps': process_fps
    }


def process_gif(pipeline, gif_path, output_path, conf_threshold=0.5, max_frames=50):
    """
    Process animated GIF and create annotated version.
    
    Args:
        pipeline: ALPRPipeline instance
        gif_path: Input GIF path
        output_path: Output GIF path
        conf_threshold: Detection confidence threshold
        max_frames: Maximum frames to process
        
    Returns:
        dict with statistics
    """
    # Open GIF
    gif = Image.open(gif_path)
    
    # Get frame count
    try:
        total_frames = gif.n_frames
    except AttributeError:
        total_frames = 1
    
    frames_to_process = min(total_frames, max_frames)
    
    print(f"🎞️ Processing GIF: {total_frames} frames")
    print(f"📊 Processing: {frames_to_process} frames")
    
    annotated_frames = []
    annotated_frames = []
    total_detections = 0
    detected_plates = set()
    
    for frame_idx in range(frames_to_process):
        gif.seek(frame_idx)
        
        # Convert to RGB
        frame_rgb = gif.convert('RGB')
        frame_array = np.array(frame_rgb)
        
        # Save temp image
        temp_path = f"/tmp/gif_frame_{frame_idx}.jpg"
        frame_rgb.save(temp_path)
        
        # Process through pipeline
        results = pipeline.process_image(temp_path, conf_threshold)
        
        # Get annotated frame
        annotated_rgb = results['step5_final']
        annotated_pil = Image.fromarray(annotated_rgb)
        annotated_frames.append(annotated_pil)
        
        # Count detections
        num_detections = len(results['metadata']['detections'])
        total_detections += num_detections
        
        # Collect detected plates
        if results.get('step4_ocr'):
            for ocr in results['step4_ocr']:
                if ocr.get('text'):
                    detected_plates.add(ocr['text'])
        
        # Cleanup
        os.remove(temp_path)
        
        if (frame_idx + 1) % 10 == 0:
            print(f"  ✓ Processed {frame_idx + 1} frames...")
    
    # Save annotated GIF
    if annotated_frames:
        annotated_frames[0].save(
            output_path,
            save_all=True,
            append_images=annotated_frames[1:],
            duration=gif.info.get('duration', 100),
            loop=gif.info.get('loop', 0)
        )
    
    print(f"✅ GIF processing complete!")
    print(f"   Processed: {frames_to_process} frames")
    print(f"   Total detections: {total_detections}")
    
    return {
        'total_frames': total_frames,
        'processed_frames': frames_to_process,
        'total_detections': total_detections,
        'detected_plates': list(detected_plates)
    }


def sample_video_frames(pipeline, video_path, num_samples=10, conf_threshold=0.5):
    """
    Sample and process frames from video for quick preview.
    
    Args:
        pipeline: ALPRPipeline instance
        video_path: Input video path
        num_samples: Number of frames to sample
        conf_threshold: Detection confidence threshold
        
    Returns:
        List of processed frame results
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly
    frame_indices = np.linspace(0, total_frames - 1, min(num_samples, total_frames), dtype=int)
    
    results = []
    
    print(f"🎬 Sampling {len(frame_indices)} frames from video...")
    
    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save temp image
        temp_path = f"/tmp/sample_frame_{idx}.jpg"
        cv2.imwrite(temp_path, frame)
        
        # Process frame
        frame_result = pipeline.process_image(temp_path, conf_threshold)
        frame_result['frame_number'] = int(idx)
        frame_result['timestamp'] = idx / cap.get(cv2.CAP_PROP_FPS)
        results.append(frame_result)
        
        # Cleanup
        os.remove(temp_path)
        
        print(f"  ✓ Frame {i+1}/{len(frame_indices)} (#{idx})")
    
    cap.release()
    
    print(f"✅ Sampled {len(results)} frames")
    
    return results


def create_static_video(image, output_path, duration=3, fps=10):
    """
    Create a static video from a single image (for preview).
    
    Args:
        image: Numpy array (RGB)
        output_path: Path to save MP4
        duration: Duration in seconds
        fps: Frames per second
    """
    height, width, _ = image.shape
    
    # Smart codec selection
    fourcc_options = []
    
    if output_path.endswith('.webm'):
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'vp09'),
            cv2.VideoWriter_fourcc(*'vp80'),
        ]
    else:
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'avc1'),
            cv2.VideoWriter_fourcc(*'mp4v'),
        ]
    
    out = None
    for fourcc in fourcc_options:
        try:
            temp_out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if temp_out.isOpened():
                out = temp_out
                break
        except:
            continue
            
    if not out or not out.isOpened():
        # Fallback
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write same frame for duration * fps
    num_frames = int(duration * fps)
    
    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for _ in range(num_frames):
        out.write(frame_bgr)
        
    out.release()
    return output_path
