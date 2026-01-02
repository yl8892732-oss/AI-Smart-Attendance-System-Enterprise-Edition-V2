# AI-Smart-Attendance-System-Enterprise-Edition-V2
An industrial-grade automated employee time-tracking solution built on deep learning (ArcFace + YOLOv8). Designed for data integrity and enterprise-level reporting.

 Key Features
Precision Biometrics: Integrated YOLOv8 for face detection and ArcFace (ResNet50) for feature extraction.

Dual-Image Verification: Cross-references enrollment profiles with real-time snapshots to prevent fraud.

Data Integrity (Atomic Operations): Implements a temporary-file-swap mechanism for JSON updates to prevent data corruption.

Automated Lifecycle: Cascading file system cleanup; deleting an employee profile automatically purges associated biometric assets.

Enterprise Analytics: Dynamic KPI dashboards and precision PDF report generation.

Technical Stack
Backend: Python (Flask), OpenCV, Pandas.

AI Engine: TensorFlow 2.16+ (Keras 3), YOLOv8, ArcFace.

Frontend: Bootstrap 5, JavaScript (ES6+), html2pdf.js.

Storage: Atomic-Persistence JSON Database.
