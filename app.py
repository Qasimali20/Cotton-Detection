from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import torch
from collections import Counter

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

model = YOLO('best.pt') 
CLASS_NAMES = ['bud', 'cotton', 'flower']

def detect_and_count(image_path):
    results = model(image_path)[0]

    names = results.names
    classes = [names[int(cls)] for cls in results.boxes.cls]
    counts = dict(Counter(classes))

    img = cv2.imread(image_path)
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        label = names[int(cls)]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    result_filename = os.path.join(app.config['RESULT_FOLDER'], os.path.basename(image_path))
    cv2.imwrite(result_filename, img)

    counts = {cls: counts.get(cls, 0) for cls in CLASS_NAMES}

    estimated_yield = round(counts['cotton'] * 2.5, 2)

    return result_filename, counts, estimated_yield

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            filename = secure_filename(image.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(upload_path)

            result_path, counts, estimated_yield = detect_and_count(upload_path)
            return render_template('index.html', uploaded=True,
                                   original_img=url_for('static', filename='uploads/' + filename),
                                   result_img=url_for('static', filename='results/' + filename),
                                   counts=counts,
                                   estimated_yield=estimated_yield)

    return render_template('index.html', uploaded=False)

if __name__ == '__main__':
    app.run(debug=True)
