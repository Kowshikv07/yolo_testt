import cv2
import numpy as np
import time
import os
import threading
import queue
import subprocess
# import RPi.GPIO as GPIO

class ObjectDetector:
    def __init__(self):
        self.camera = None
        self.detection_queue = queue.Queue(maxsize=1)
        self.command_queue = queue.Queue(maxsize=5)
        self.running = False
        self.detection_thread = None
        self.audio_thread = None
        self.feedback_thread = None
        
    
        """Download the MobileNet SSD model files"""
        import requests
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        urls = {
            "prototxt": "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
            "caffemodel": "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"
        }
        
        files = {"prototxt": self.prototxt, "caffemodel": self.model}

        for key, url in urls.items():
            print(f"Downloading {key}...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(files[key], "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                print(f"{key} downloaded successfully.")
            else:
                print(f"Failed to download {key}. Status code: {response.status_code}")

        # Initialize GPIO for haptic feedback
        # GPIO.setmode(GPIO.BCM)
        # self.haptic_pin = 18  # GPIO pin for vibration motor
        # GPIO.setup(self.haptic_pin, GPIO.OUT)
        
        # Load MobileNet SSD model (OpenCV DNN)
        # This works well on Raspberry Pi without requiring PyTorch
        self.model_path = os.path.join(os.path.dirname(__file__), 'models')
        self.prototxt = os.path.join(self.model_path, 'MobileNetSSD_deploy.prototxt')
        self.model = os.path.join(self.model_path, 'MobileNetSSD_deploy.caffemodel')
        
        # Check if model files exist
        if not os.path.exists(self.prototxt) or not os.path.exists(self.model):
            print("Model files not found. Downloading...")
            self._download_model()
        
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)
        
        # Classes the model can detect
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        
        # Important objects for navigation
        self.priority_objects = ["person", "car", "bicycle", "bus", "motorbike", "chair"]
        
        # Obstacle distances (simulated - would be from ultrasonic sensors)
        self.distance_thresholds = {
            "immediate": 1.0,  # meters
            "close": 2.0,      # meters
            "medium": 3.5,     # meters
            "far": 5.0         # meters
        }
    
    def _download_model(self):
        """Download the MobileNet SSD model files"""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
        # Download prototxt
        subprocess.run([
            "wget", 
            "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt", 
            "-O", self.prototxt
        ])
        
        # Download caffemodel
        subprocess.run([
            "wget", 
            "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc", 
            "-O", self.model
        ])
    
    def start(self):
        """Start the object detection system"""
        if self.running:
            return
        
        self.running = True
        self.camera = cv2.VideoCapture(0)  # Use Pi camera
        
        # Set camera resolution to 640x480 for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        # Start audio feedback thread
        self.audio_thread = threading.Thread(target=self._audio_feedback_loop)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        # Start haptic feedback thread
        self.feedback_thread = threading.Thread(target=self._haptic_feedback_loop)
        self.feedback_thread.daemon = True
        self.feedback_thread.start()
        
        print("Object detection system started")
    
    def stop(self):
        """Stop the object detection system"""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
        if self.feedback_thread:
            self.feedback_thread.join(timeout=1.0)
        if self.camera:
            self.camera.release()
        # GPIO.cleanup()
        print("Object detection system stopped")
    
    def _detection_loop(self):
        """Main detection loop running in a separate thread"""
        frame_skip = 2  # Process every 3rd frame for better performance
        count = 0
        
        while self.running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                count += 1
                if count % frame_skip != 0:
                    continue
                
                # Detect objects in the frame
                objects = self._detect_objects(frame)
                
                # Update the detection queue (discard old detection if not consumed)
                try:
                    self.detection_queue.put(objects, block=False)
                except queue.Full:
                    try:
                        self.detection_queue.get_nowait()
                        self.detection_queue.put(objects, block=False)
                    except queue.Empty:
                        pass
                
            except Exception as e:
                print(f"Error in detection loop: {e}")
                time.sleep(0.1)
    
    def _detect_objects(self, frame):
        """Detect objects in the frame using MobileNet SSD"""
        # Prepare image for neural network
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
                                     0.007843, (300, 300), 127.5)
        
        # Pass the blob through the network and get detections
        self.net.setInput(blob)
        detections = self.net.forward()
        
        detected_objects = []
        
        # Loop over the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter out weak detections
            if confidence > 0.5:
                # Get the class ID
                class_id = int(detections[0, 0, i, 1])
                label = self.classes[class_id]
                
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Calculate position in frame
                center_x = (startX + endX) / 2
                relative_x = (center_x - (w/2)) / (w/2)  # -1 to 1, left to right
                
                # Calculate size (proxy for distance)
                size = (endX - startX) * (endY - startY) / (w * h)
                
                # Simulate distance (would be from ultrasonic sensor)
                # Larger objects are closer, smaller are farther
                if size > 0.25:
                    distance = "immediate"
                elif size > 0.1:
                    distance = "close"
                elif size > 0.05:
                    distance = "medium"
                else:
                    distance = "far"
                
                # Add to detected objects
                detected_objects.append({
                    'label': label,
                    'confidence': float(confidence),
                    'position': relative_x,  # -1 (left) to 1 (right)
                    'distance': distance,
                    'priority': label in self.priority_objects
                })
        
        # Sort by priority and distance
        detected_objects.sort(key=lambda x: (not x['priority'], 
                                             list(self.distance_thresholds.keys()).index(x['distance'])))
        
        return detected_objects
    
    def _audio_feedback_loop(self):
        """Generate audio feedback based on detections"""
        last_announcement = {}
        cooldown = 2.0  # seconds between repeated announcements
        
        while self.running:
            try:
                # Get latest detection
                objects = self.detection_queue.get(timeout=0.5)
                
                current_time = time.time()
                announcements = []
                
                for obj in objects[:3]:  # Limit to 3 most important objects
                    label = obj['label']
                    distance = obj['distance']
                    position = obj['position']
                    
                    # Determine direction
                    if position < -0.3:
                        direction = "left"
                    elif position > 0.3:
                        direction = "right"
                    else:
                        direction = "ahead"
                    
                    # Create announcement
                    message = f"{label} {distance} {direction}"
                    
                    # Check if we've announced this recently
                    if (label not in last_announcement or 
                            current_time - last_announcement[label] > cooldown):
                        announcements.append(message)
                        last_announcement[label] = current_time
                        
                        # Add to command queue for audio output
                        try:
                            self.command_queue.put(message, block=False)
                        except queue.Full:
                            pass
                
                if announcements:
                    print("Detected: " + ", ".join(announcements))
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in audio feedback loop: {e}")
                time.sleep(0.1)
    
    def _haptic_feedback_loop(self):
        """Generate haptic feedback based on detections"""
        while self.running:
            try:
                # Get latest detection
                objects = self.detection_queue.get(timeout=0.5)
                
                # Check for immediate obstacles
                immediate_obstacles = [obj for obj in objects 
                                      if obj['distance'] == "immediate"]
                
                if immediate_obstacles:
                    pass
                    # # Activate haptic feedback (vibration motor)
                    # GPIO.output(self.haptic_pin, GPIO.HIGH)
                    # time.sleep(0.5)
                    # GPIO.output(self.haptic_pin, GPIO.LOW)
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in haptic feedback loop: {e}")
                time.sleep(0.1)
    
    def speak(self, text):
        """Text-to-speech using espeak"""
        # Using subprocess to call espeak (must be installed on Raspberry Pi)
        subprocess.run(["espeak", "-v", "en-us", "-s", "150", text])

# Simple test code
if __name__ == "__main__":
    detector = ObjectDetector()
    
    try:
        detector.start()
        print("Press Ctrl+C to exit")
        
        # Run for 30 seconds as a test
        time.sleep(30)
        
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        detector.stop()