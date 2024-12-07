# README

## Object Detection using YOLOv3 and Flask

This project demonstrates a web application for performing object detection using the YOLOv3 model. The application allows users to upload images, processes them to detect objects, and displays the results with bounding boxes drawn around detected objects.

---

## Features

- Upload an image for object detection.  
- Detect objects using YOLOv3 pre-trained on the COCO dataset.  
- Visualize the output image with bounding boxes and labels.  
- Serve a simple and intuitive web interface using Flask.  

---

## Requirements

Ensure the following dependencies are installed before running the project:

- Python 3.8+  
- Flask==2.1.1  
- opencv-python==4.5.3.56  
- numpy==1.21.0  
- matplotlib==3.4.3  
- werkzeug==2.1.1  

Install all dependencies using the command:

```bash
pip install -r requirements.txt
```

---

## File Structure

```
project/
│
├── static/
│   └── uploads/              # Directory to store uploaded and processed images
│
├── templates/
│   ├── index.html            # Upload form
│   └── result.html           # Result display page
│
├── yolov3.cfg                # YOLOv3 configuration file
├── yolov3.weights            # YOLOv3 weights file
├── coco.names                # Class labels file
├── app.py                    # Main Flask application
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

---

## Setup Instructions

1. Clone the repository:  
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Place the following YOLO files in the root directory:  
   - `yolov3.cfg`  
   - `yolov3.weights`  
   - `coco.names`  

3. Run the Flask application:  
   ```bash
   python app.py
   ```

4. Open a browser and navigate to `http://127.0.0.1:5000`.  

---

## How It Works

1. **Upload Image**: Choose an image file (supported formats: PNG, JPG, JPEG, GIF) and upload it.  
2. **Object Detection**: The image is processed using YOLOv3, and objects are detected based on the COCO dataset.  
3. **View Results**: The output image is displayed with bounding boxes and labels.  

---

## Notes

- Adjust the confidence threshold in the `detect()` method for more or fewer detections.  
- Ensure the paths to YOLO files are correct in the `app.py` script.  

---

## License

This project is open-source and available for modification and redistribution.
