"""
Visualization utilities for progressive pipeline display.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_confidence_badge(confidence, threshold_high=0.85, threshold_low=0.6):
    """
    Create confidence badge with emoji and color.
    
    Args:
        confidence: Confidence score (0-1)
        threshold_high: Threshold for high confidence
        threshold_low: Threshold for low confidence
        
    Returns:
        tuple: (emoji, color, label)
    """
    if confidence >= threshold_high:
        return "✔️", "#00ff00", "High Confidence"
    elif confidence >= threshold_low:
        return "⚠️", "#ffaa00", "Medium Confidence"
    else:
        return "❌", "#ff0000", "Low Confidence"


def annotate_detection(image, bbox, confidence, label="Plate"):
    """
    Annotate image with detection bounding box.
    
    Args:
        image: RGB image array
        bbox: Bounding box (x1, y1, x2, y2)
        confidence: Detection confidence
        label: Text label
        
    Returns:
        Annotated image
    """
    img = image.copy()
    x1, y1, x2, y2 = bbox
    
    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # Add confidence badge
    emoji, color, _ = create_confidence_badge(confidence)
    conf_text = f"{emoji} {confidence:.2%}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    
    # Get text size
    (text_w, text_h), _ = cv2.getTextSize(conf_text, font, font_scale, thickness)
    
    # Draw background
    cv2.rectangle(img, (x1, y1 - text_h - 15), (x1 + text_w + 10, y1), (0, 200, 0), -1)
    
    # Draw text
    cv2.putText(img, conf_text, (x1 + 5, y1 - 8), font, font_scale, (255, 255, 255), thickness)
    
    return img


def create_pipeline_visualization(step_name, image, description=""):
    """
    Create visualization for a pipeline step.
    
    Args:
        step_name: Name of the pipeline step
        image: Image array
        description: Optional description
        
    Returns:
        Formatted visualization
    """
    # This will be used by Gradio to display each step
    return {
        'step': step_name,
        'image': image,
        'description': description
    }


def create_analysis_report(results):
    """
    Create detailed analysis report from pipeline results.
    
    Args:
        results: Pipeline results dict
        
    Returns:
        Formatted markdown report
    """
    metadata = results['metadata']
    conditions = metadata['conditions']
    detections = results['step4_ocr']
    
    report = f"""
## 🔍 Analysis Report

### Image Conditions
- **Lighting**: {conditions['lighting_emoji']} {conditions['lighting']}
- **Brightness**: {conditions['brightness']:.1f} / 255
- **Sharpness**: {conditions['blur']} (score: {conditions['blur_score']:.1f})

### Detections
"""
    
    if detections:
        for i, det in enumerate(detections, 1):
            emoji, _, conf_label = create_confidence_badge(det['confidence'])
            
            report += f"""
#### Plate {i}
- **Text**: `{det['text'] if det['text'] else 'Not detected'}`
- **OCR Confidence**: {emoji} {det['confidence']:.2%} ({conf_label})
- **Detection Confidence**: {det['detection_confidence']:.2%}
"""
            
            # Add warnings for low confidence
            if det['confidence'] < 0.6:
                report += f"\n> ⚠️ **Low OCR confidence** - Possible reasons: {conditions['lighting']}, {conditions['blur']}, extreme angle, or dirty plate\n"
    else:
        report += "\n❌ **No plates detected**\n"
    
    return report


def create_step_images(results):
    """
    Create list of images for each pipeline step.
    
    Args:
        results: Pipeline results dict
        
    Returns:
        List of (step_name, image, description) tuples
    """
    steps = []
    
    # Step 1: Raw image
    steps.append((
        "1️⃣ Raw Input",
        results['step1_raw'],
        "Original image as received"
    ))
    
    # Step 2: Detection
    if results['step2_detection'] is not None:
        num_detections = len(results['metadata']['detections'])
        steps.append((
            "2️⃣ YOLOv8 Detection",
            results['step2_detection'],
            f"Detected {num_detections} plate(s)"
        ))
    
    # Step 3: ROI extraction
    if results['step3_roi']:
        for i, roi in enumerate(results['step3_roi'], 1):
            steps.append((
                f"3️⃣ ROI Extraction #{i}",
                roi,
                "Cropped license plate region"
            ))
    
    # Step 5: Final result
    if results['step5_final'] is not None:
        ocr_results = results['step4_ocr']
        if ocr_results:
            texts = [r['text'] for r in ocr_results if r['text']]
            desc = f"Final result: {', '.join(texts) if texts else 'No text detected'}"
        else:
            desc = "No plates detected"
        
        steps.append((
            "5️⃣ Final Result",
            results['step5_final'],
            desc
        ))
    
    return steps


def format_ocr_result(text, confidence):
    """
    Format OCR result with character-by-character display.
    
    Args:
        text: OCR text
        confidence: OCR confidence
        
    Returns:
        Formatted string
    """
    emoji, color, label = create_confidence_badge(confidence)
    
    if not text:
        return f"{emoji} No text detected (confidence: {confidence:.2%})"
    
    # Format with spacing for visual effect
    formatted_text = " ".join(list(text))
    
    return f"""
### OCR Result
**Text**: `{formatted_text}`  
**Confidence**: {emoji} {confidence:.2%} ({label})
"""
