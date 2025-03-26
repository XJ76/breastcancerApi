# Create your views here.

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from django.core.files.storage import default_storage
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from .serializers import ImageUploadSerializer
from rest_framework import status
import os
import joblib
import pandas as pd
from PIL import Image
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

# Load AI model and scaler
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model/cnn_model.h5")
SCALER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model/scaler.pkl")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model/data.csv")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    CLASS_NAMES = ["Benign", "Malignant"]
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

def preprocess_image(image_file):
    # Read the image
    img = Image.open(image_file)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize to 224x224
    img = img.resize((224, 224))
    
    # Convert to numpy array and normalize
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    
    # Reshape for model input
    img_array = img_array.reshape(1, -1)
    
    # Scale the features
    img_scaled = scaler.transform(img_array)
    
    # Reshape for CNN
    img_reshaped = img_scaled.reshape((1, img_scaled.shape[1], 1))
    
    return img_reshaped

class PredictBreastCancerView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if model is None or scaler is None:
            return Response(
                {"error": "Model or scaler not loaded. Please check server logs."}, 
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
            
        file_serializer = ImageUploadSerializer(data=request.data)
        if file_serializer.is_valid():
            try:
                uploaded_file = file_serializer.save()
                image_file = uploaded_file.image

                # Preprocess Image
                processed_image = preprocess_image(image_file)

                # Model Prediction
                predictions = model.predict(processed_image)
                confidence = float(np.max(predictions))
                label = CLASS_NAMES[np.argmax(predictions)]

                # Clean up the uploaded file
                if os.path.exists(image_file.path):
                    os.remove(image_file.path)

                return Response({
                    "prediction": label,
                    "confidence": confidence,
                    "message": f"The image is predicted to be {label} with {confidence:.2%} confidence"
                }, status=status.HTTP_200_OK)

            except Exception as e:
                return Response({
                    "error": f"Error processing image: {str(e)}"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(
            file_serializer.errors, 
            status=status.HTTP_400_BAD_REQUEST
        )

def generate_metrics_plots(y_true, y_pred):
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Create plots
    plt.figure(figsize=(15, 5))

    # Confusion Matrix
    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Metrics Bar Plot
    plt.subplot(1, 3, 2)
    metrics = [accuracy, precision, recall, f1]
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    plt.bar(labels, metrics)
    plt.title('Model Metrics')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

    # ROC Curve
    plt.subplot(1, 3, 3)
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    # Save plot to base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'plot': plot_data
    }

class ModelMetricsView(APIView):
    def get(self, request):
        try:
            # Load and preprocess data
            data = pd.read_csv(DATA_PATH)
            X = data.drop('diagnosis', axis=1)
            y = data['diagnosis']
            y = (y == 'M').astype(int)

            # Scale features
            X_scaled = scaler.transform(X)
            X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

            # Get predictions
            y_pred = (model.predict(X_reshaped) > 0.5).astype(int)
            
            # Generate metrics and plots
            metrics_data = generate_metrics_plots(y, y_pred)
            
            return Response(metrics_data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                "error": f"Error generating metrics: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
