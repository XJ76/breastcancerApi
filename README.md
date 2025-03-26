# Breast Cancer Detection API

This is a Django-based API that uses a CNN model to detect breast cancer from medical images.

## Setup Instructions

1. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
cd backendApi/model
python train_model.py
```

4. Run the Django server:
```bash
cd backendApi
python manage.py runserver
```

## Frontend Integration

The API is configured to accept requests from Next.js frontend applications running on `localhost:3000`. To integrate with your Next.js frontend:

1. Create a new Next.js project if you haven't already:
```bash
npx create-next-app@latest frontend
cd frontend
```

2. Copy the example component from `frontend-example.js` to your Next.js project.

3. Start your Next.js development server:
```bash
npm run dev
```

4. Make sure your Django backend is running on `http://localhost:8000`

## API Usage

### Endpoint: POST /api/predict/

Send a POST request with a breast cancer image to get the prediction.

#### Request Format:
- Method: POST
- Content-Type: multipart/form-data
- Body: 
  - image: (file) The breast cancer image to analyze

#### Response Format:
```json
{
    "prediction": "Benign/Malignant",
    "confidence": 0.95,
    "message": "The image is predicted to be Benign with 95% confidence"
}
```

## Model Details

The system uses a CNN model trained on the Wisconsin Breast Cancer dataset. The model:
- Takes grayscale images as input
- Resizes images to 224x224
- Uses a 1D CNN architecture
- Provides binary classification (Benign/Malignant)

## Requirements

- Python 3.8+
- TensorFlow 2.8.0
- Django 3.2.5
- Other dependencies listed in requirements.txt

## CORS Configuration

The API is configured to accept requests from:
- http://localhost:3000
- http://127.0.0.1:3000

To modify allowed origins, update the `CORS_ALLOWED_ORIGINS` setting in `backendApi/settings.py`.