from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import os

#####################################################
###### this is to ignore the warnings , delete if ,needed
import warnings
warnings.filterwarnings("ignore")
app = Flask(__name__)
CORS(app)
#######################################################
class_dict = {0: "safe driving",
        1: "texting - right",
        2: "talking on the phone - right",
        3: "texting - left",
        4: "talking on the phone - left",
        5: "operating the radio",
        6: "drinking",
        7: "reaching behind",
        8: "hair and makeup",
        9: "talking to passenger"}
# Charger le modèle
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # Remplacer 10 par le nombre de classes de votre problème

# Charger les poids du modèle
model_path = 'Driver-Behavior-main/model-driver10.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Définir la transformation d'image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Définir la route pour recevoir l'image et effectuer l'inférence
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'}), 400
    
    image = request.files['image']
    image = Image.open(image)
    input_tensor = transform(image).unsqueeze(0)  # Ajouter la dimension du batch

    # Effectuer l'inférence
    with torch.no_grad():
        output = model(input_tensor)

    # Convertir les scores de sortie en probabilités en utilisant softmax
    probabilities = torch.softmax(output, dim=1)
    probabilities = [round(float(elem),4) for elem in probabilities[0]]


    # Obtenir la classe prédite avec la probabilité la plus élevée
    predicted_class_index = torch.argmax(output, dim=1).item()


    # Retourner la prédiction
    prediction = {
        "predicted_class": class_dict[predicted_class_index],
        "probability": max(probabilities)
    }

    print("probability : " + str(prediction.get("predicted_class")) + "\n")
    print("probability : "+str(prediction.get("probability"))+"\n")

    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=False)