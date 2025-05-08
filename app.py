import cv2
import numpy as np
import gradio as gr
from detection import detect_and_annotate

def process_image(input_image):
    # Convert to BGR
    image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    annotated_image, valid_detections = detect_and_annotate(image)
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # Decide title based on detection result
    title = "Detected Faces with Age" if valid_detections > 0 else "No face found in the image."
    return annotated_image_rgb, title

# Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=[gr.Image(type="numpy", label="Result"), gr.Textbox(label="Status")],
    title="Face Age Detection App",
    description="Upload a photo — if faces are found, it predicts ages, otherwise tells you no faces detected."
)

if __name__ == "__main__":
    iface.launch()
