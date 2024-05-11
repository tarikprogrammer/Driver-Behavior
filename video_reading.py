import torch
from torchvision import transforms, models
import torch.nn as nn
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import numpy as np
from PIL import Image
from flask import Response
import time
import json

app = Flask(__name__)
CORS(app)

# Load the pre-trained ResNet50 model
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)
model_path = 'Driver-Behavior-main/model-driver10.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define transformations to apply to input frames
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Define class labels
class_labels = {
    0: "safe driving",
    1: "texting - right",
    2: "talking on the phone - right",
    3: "texting - left",
    4: "talking on the phone - left",
    5: "operating the radio",
    6: "drinking",
    7: "reaching behind",
    8: "hair and makeup",
    9: "talking to passenger"
}

def read_video(input_path, data):
    cap = cv2.VideoCapture(input_path) 
    frame_rate =1 # PER 0.5s
    frame_count =0
    
    if not cap.isOpened():  
        print("Erreur: Impossible d'ouvrir la vidéo.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read() 
        
        if not ret:  
            break
        frame_count= frame_count+1 
        if frame_count % int(cap.get(5) / frame_rate) == 0:
            data.append(frame)
        
    cap.release()



@app.route('/classify', methods=['POST'])
def classify_video():
    file = request.files['video']
    video_path = "C:/Users/pc/Downloads/" + str(file.filename)
    print(video_path)
    data = []
    results = []
    read_video(video_path, results)
    print(results)
    for frame in results:
    # Convertir le tableau NumPy en objet de type fichier avec io.BytesIO
        image_file = io.BytesIO()
        Image.fromarray(frame).save(image_file, format='JPEG')
        image_file.seek(0)
        
        # Ouvrir l'image à l'aide de Image.open()
        image = Image.open(image_file)
        
        # Appliquer les transformations et effectuer l'inférence
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class_index = torch.argmax(probabilities, dim=1).item()
            prediction = {
                "predicted_class": class_labels[predicted_class_index],
                "probability": round(probabilities[0][predicted_class_index].item(), 2),
            }
            print(prediction)
            data.append(prediction)
    return jsonify(data)     

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)


