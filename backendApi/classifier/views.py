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

# Load AI model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model/cnn_model.h5")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    CLASS_NAMES = ["No Cancer", "Cancer"]  # Update based on your dataset
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class PredictBreastCancerView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if model is None:
            return Response(
                {"error": "Model not loaded. Please check server logs."}, 
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
            
        file_serializer = ImageUploadSerializer(data=request.data)
        if file_serializer.is_valid():
            try:
                uploaded_file = file_serializer.save()
                image_path = uploaded_file.image.path

                # Preprocess Image
                img = image.load_img(image_path, target_size=(224, 224), color_mode='grayscale')
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                # Model Prediction
                predictions = model.predict(img_array)
                confidence = float(np.max(predictions))
                label = CLASS_NAMES[np.argmax(predictions)]

                return Response({
                    "prediction": label, 
                    "confidence": confidence
                }, status=status.HTTP_200_OK)

            except Exception as e:
                return Response({
                    "error": f"Error processing image: {str(e)}"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(
            file_serializer.errors, 
            status=status.HTTP_400_BAD_REQUEST
        )
