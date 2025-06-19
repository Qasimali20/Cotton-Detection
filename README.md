# Cotton Detection with YOLOv12 
A YOLOv12-based object detection system for identifying and localizing cotton plants components in farm images. This project helps farmers and agricultural researchers automatically detect and count cotton buds, flowers, and cotton bolls to monitor crop development and estimate yield.

Project Overview
This project implements YOLOv12 (You Only Look Once) for real-time detection of cotton plant components in agricultural images. The system can identify three main classes of cotton plant features, enabling automated crop monitoring and yield estimation.
Detected Classes

Buds: Young cotton flower buds before blooming
Flowers: Fully opened cotton flowers
Cotton: Mature cotton bolls ready for harvest

Key Features

Real-time Detection: Fast inference using YOLOv12 architecture
Multi-class Detection: Simultaneously detects buds, flowers, and cotton bolls
Web Interface: Flask-based web application for easy image upload and analysis
Bounding Box Visualization: Clear visual representation of detected objects
Batch Processing: Support for processing multiple images
Confidence Scoring: Adjustable confidence thresholds for detection
