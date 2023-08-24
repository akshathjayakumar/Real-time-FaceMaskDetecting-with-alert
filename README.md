# Real-time-FaceMaskDetecting-with-alert
Implements real-time face mask detection in a video stream, and if someone is detected without a mask, it sends an email alert with an attached image of the person for potential enforcement or follow-up actions.

The "Real-time FaceMaskDetecting with Alert" project uses computer vision to detect mask-wearing in a video stream. 
By leveraging OpenCV, the code captures live video frames, employing a pre-trained model (MobileNetV2) to recognize faces and determine mask presence. 

If someone lacks a mask, the system captures their image and sends an email alert to authorized recipients. The "trainner" script handles training the mask detection model, while the "mask detect" script performs real-time mask detection and alerting.

