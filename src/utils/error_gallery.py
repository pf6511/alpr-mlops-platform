"""
Error gallery management for showcasing model limitations.
"""

import json
import os
from pathlib import Path


class ErrorGallery:
    """Manage and display error examples."""
    
    def __init__(self, gallery_path="assets/error_examples"):
        """
        Initialize error gallery.
        
        Args:
            gallery_path: Path to error examples directory
        """
        self.gallery_path = Path(gallery_path)
        self.annotations_path = self.gallery_path / "annotations.json"
        self.annotations = self._load_annotations()
    
    def _load_annotations(self):
        """Load error annotations from JSON file."""
        if self.annotations_path.exists():
            with open(self.annotations_path, 'r') as f:
                return json.load(f)
        return {}
    
    def get_examples(self):
        """
        Get all error examples with annotations.
        
        Returns:
            List of (image_path, annotation) tuples
        """
        examples = []
        
        if not self.gallery_path.exists():
            return examples
        
        for img_file in self.gallery_path.glob("*.jpg"):
            filename = img_file.name
            annotation = self.annotations.get(filename, {
                "issue": "Unknown issue",
                "detection_confidence": 0.0,
                "ocr_confidence": 0.0,
                "expected": "N/A",
                "predicted": "N/A"
            })
            
            examples.append((str(img_file), annotation))
        
        return examples
    
    def format_example(self, image_path, annotation):
        """
        Format error example for display.
        
        Args:
            image_path: Path to error image
            annotation: Annotation dict
            
        Returns:
            Formatted markdown string
        """
        import base64
        
        # Use Base64 encoding to embed image directly in Markdown
        # This avoids all issues with paths, permissions, and servers (local vs HF)
        valid_image = False
        img_tag = "*(Image not found)*"
        
        try:
            if os.path.exists(image_path):
                with open(image_path, "rb") as img_file:
                    b64_string = base64.b64encode(img_file.read()).decode('utf-8')
                    img_tag = f'<img src="data:image/jpeg;base64,{b64_string}" alt="Error Image" style="max-width: 100%; max-height: 400px; border-radius: 8px;">'
                    valid_image = True
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            
        return f"""
### ⚠️ Error Case: {annotation.get('issue', 'Unknown')}

{img_tag}

**Expected**: `{annotation.get('expected', 'N/A')}`  
**Predicted**: `{annotation.get('predicted', 'N/A')}`

**Detection Confidence**: {annotation.get('detection_confidence', 0.0):.2%}  
**OCR Confidence**: {annotation.get('ocr_confidence', 0.0):.2%}

**Analysis**: {annotation.get('analysis', 'This case demonstrates a limitation of the model.')}
"""

    def clear_gallery(self):
        """Delete all error examples and annotations."""
        try:
            # Delete all jpg images in directory
            for f in self.gallery_path.glob("*.jpg"):
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Error deleting {f}: {e}")
            
            # Clear annotations
            self.annotations = {}
            if self.annotations_path.exists():
                os.remove(self.annotations_path)
                
            return True, "Gallery cleared successfully"
        except Exception as e:
            return False, f"Error clearing gallery: {e}"
    
    def create_gallery_markdown(self):
        """
        Create complete gallery markdown.
        
        Returns:
            Markdown string with all examples
        """
        examples = self.get_examples()
        
        if not examples:
            return """
## 🎯 Error Gallery

No error examples available yet. This section will showcase real failure cases to demonstrate understanding of model limitations.

**Why show errors?**
- Transparency about model capabilities
- Understanding edge cases
- Continuous improvement opportunities
"""
        
        markdown = """
## 🎯 Error Gallery

These examples showcase real failure cases where the model struggled. Understanding these limitations is crucial for production deployment.

---
"""
        
        for img_path, annotation in examples:
            markdown += self.format_example(img_path, annotation)
            markdown += "\n---\n"
        
        return markdown
    
    def add_example(self, image_path, issue, expected, predicted, 
                   detection_conf, ocr_conf, analysis=""):
        """
        Add new error example to gallery with FIFO limit of 10.
        
        Args:
            image_path: Path to source image (will be copied)
            issue: Description of the issue
            expected: Expected plate text
            predicted: Predicted plate text
            detection_conf: Detection confidence
            ocr_conf: OCR confidence
            analysis: Optional detailed analysis
        """
        # Ensure directory exists
        self.gallery_path.mkdir(parents=True, exist_ok=True)
        
        # Check limit
        current_examples = self.get_examples()
        if len(current_examples) >= 10:
            # Remove oldest (first in list from get_examples which uses glob, likely alphabetical)
            # Better to rely on valid filenames or time. 
            # For simplicity, let's just use the list order or file modification time.
            # glob order is not guaranteed. Let's sort by modification time.
            files = list(self.gallery_path.glob("*.jpg"))
            files.sort(key=os.path.getmtime)
            
            if files:
                oldest_file = files[0]
                try:
                    os.remove(oldest_file)
                    # Also remove from annotations if present
                    if oldest_file.name in self.annotations:
                        del self.annotations[oldest_file.name]
                except Exception as e:
                    print(f"Error removing oldest file: {e}")

        # Generate unique filename
        import shutil
        import time
        timestamp = int(time.time())
        new_filename = f"error_{timestamp}.jpg"
        target_path = self.gallery_path / new_filename
        
        try:
            # Copy image
            shutil.copy2(image_path, target_path)
            
            self.annotations[new_filename] = {
                "issue": issue,
                "expected": expected,
                "predicted": predicted,
                "detection_confidence": detection_conf,
                "ocr_confidence": ocr_conf,
                "analysis": analysis
            }
            
            # Save annotations
            with open(self.annotations_path, 'w') as f:
                json.dump(self.annotations, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error adding example: {e}")
            return False


# Example annotations for initial gallery
EXAMPLE_ANNOTATIONS = {
    "extreme_angle.jpg": {
        "issue": "Extreme angle + motion blur",
        "expected": "ABC123",
        "predicted": "A8C12",
        "detection_confidence": 0.92,
        "ocr_confidence": 0.34,
        "analysis": "The plate was detected successfully, but the extreme viewing angle combined with motion blur made character recognition difficult. The OCR confused 'B' with '8' and failed to recognize '3'."
    },
    "dirty_plate.jpg": {
        "issue": "Heavily soiled plate",
        "expected": "XYZ789",
        "predicted": "XY789",
        "detection_confidence": 0.88,
        "ocr_confidence": 0.41,
        "analysis": "Dirt and grime obscured the 'Z' character, causing the OCR to skip it entirely. This demonstrates the importance of image preprocessing for real-world conditions."
    },
    "low_light.jpg": {
        "issue": "Very low light + reflections",
        "expected": "DEF456",
        "predicted": "",
        "detection_confidence": 0.67,
        "ocr_confidence": 0.12,
        "analysis": "Nighttime conditions with headlight reflections created high contrast that confused the OCR model. Detection was marginal, and OCR failed completely."
    },
    "partial_occlusion.jpg": {
        "issue": "Partial occlusion by frame",
        "expected": "GHI321",
        "predicted": "HI321",
        "detection_confidence": 0.79,
        "ocr_confidence": 0.58,
        "analysis": "The license plate frame partially covered the first character. While the model detected the plate, it couldn't recognize the occluded 'G'."
    },
    "multiple_plates.jpg": {
        "issue": "Multiple plates in frame",
        "expected": "JKL654 (front)",
        "predicted": "JKL654, MNO987",
        "detection_confidence": 0.94,
        "ocr_confidence": 0.89,
        "analysis": "The model successfully detected both front and rear plates, but in some applications, this could cause confusion about which plate to prioritize."
    }
}
