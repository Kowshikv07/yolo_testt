import cv2
import torch
import numpy as np
from ultralytics import YOLO
import threading
import time
import pyttsx3
import math

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", camera_index=0, confidence=0.5, iou=0.4):
        # Initialize YOLO model
        self.model = YOLO(model_path)
        self.model.conf = confidence
        self.model.iou = iou
        self.class_names = self.model.names
        
        # Initialize camera
        self.camera_index = camera_index
        self.cap = None
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Dictionary to track objects and when they were last announced
        self.last_announced = {}
        self.announcement_cooldown = 3  # seconds between announcements of same object
        
        # Priority objects that are more important to announce
        self.priority_objects = ["person", "car", "truck", "bus", "bicycle", "motorcycle", 
                                "traffic light", "stop sign", "chair", "pothole", "stairs", "door"]
        
        # Objects that represent obstacles
        self.obstacle_objects = ["person", "car", "truck", "bus", "bicycle", "motorcycle", 
                                "bench", "chair", "table", "pothole", "rock", "tree", "pole"]
        
        # Thread for audio feedback
        self.audio_thread = None
        self.audio_queue = []
        self.audio_lock = threading.Lock()
        self.running = True
        
    def start_camera(self):
        """Initialize and start the camera"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera at index {self.camera_index}")
        return self.cap.isOpened()
    
    def start_audio_thread(self):
        """Start the audio feedback thread"""
        self.audio_thread = threading.Thread(target=self._audio_worker)
        self.audio_thread.daemon = True
        self.audio_thread.start()
    
    def _audio_worker(self):
        """Worker thread for audio feedback"""
        while self.running:
            if self.audio_queue:
                with self.audio_lock:
                    message = self.audio_queue.pop(0)
                self.engine.say(message)
                self.engine.runAndWait()
            time.sleep(0.1)
    
    def queue_audio(self, message):
        """Add message to audio queue"""
        with self.audio_lock:
            if message not in self.audio_queue:  # Avoid duplicate messages
                self.audio_queue.append(message)
    
    def calculate_object_position(self, x1, y1, x2, y2, frame_width, frame_height):
        """Calculate relative position of object in frame"""
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Determine horizontal position
        if center_x < frame_width * 0.33:
            h_pos = "left"
        elif center_x < frame_width * 0.66:
            h_pos = "center"
        else:
            h_pos = "right"
        
        # Determine vertical position
        if center_y < frame_height * 0.33:
            v_pos = "top"
        elif center_y < frame_height * 0.66:
            v_pos = "middle"
        else:
            v_pos = "bottom"
        
        # Estimate distance based on object size relative to frame
        object_area = (x2 - x1) * (y2 - y1)
        frame_area = frame_width * frame_height
        area_ratio = object_area / frame_area
        
        if area_ratio > 0.25:
            distance = "very close"
        elif area_ratio > 0.1:
            distance = "close"
        elif area_ratio > 0.02:
            distance = "medium distance"
        else:
            distance = "far"
        
        return h_pos, v_pos, distance
    
    def process_frame(self, frame):
        """Process a single frame with YOLO detection"""
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (w // 2, h // 2))
        
        # Run YOLO object detection
        results = self.model(small_frame, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()
        
        current_objects = []
        current_time = time.time()
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            # Scale back to original size
            x1, y1, x2, y2 = int(x1 * 2), int(y1 * 2), int(x2 * 2), int(y2 * 2)
            
            # Get class name
            class_name = self.class_names[int(cls)]
            current_objects.append(class_name)
            
            # Draw bounding box and label
            color = (0, 255, 0) if class_name in self.priority_objects else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Determine position and announce important objects
            h_pos, v_pos, distance = self.calculate_object_position(x1, y1, x2, y2, w, h)
            
            # Only announce priority objects or obstacles
            if class_name in self.priority_objects or class_name in self.obstacle_objects:
                # Check if this object can be announced (cooldown)
                if class_name not in self.last_announced or (current_time - self.last_announced[class_name]) > self.announcement_cooldown:
                    self.last_announced[class_name] = current_time
                    
                    # Create announcement message
                    message = f"{class_name} {distance} on your {h_pos}"
                    self.queue_audio(message)
        
        # Add a direction indicator for visual debugging
        cv2.line(frame, (w//2, h), (w//2, h-100), (255, 0, 0), 2)
        
        return frame, current_objects
    
    def run_detection_loop(self):
        """Main loop for object detection"""
        if not self.start_camera():
            print("Error: Could not start camera")
            return
        
        self.start_audio_thread()
        self.queue_audio("Starting object detection")
        
        try:
            while self.cap.isOpened() and self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                processed_frame, objects = self.process_frame(frame)
                
                # Display result
                cv2.imshow("Smart Navigation - Object Detection", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            print(f"Error in detection loop: {e}")
        finally:
            self.running = False
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

# Run the detector when executed as a script
if __name__ == "__main__":
    detector = ObjectDetector()
    try:
        detector.run_detection_loop()
    except KeyboardInterrupt:
        print("Stopping detection...")
    finally:
        detector.cleanup()