from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
# from  import generate_remedy  
from streewiseaiWithoutPKL import generate_remedy
from streewiseaiWithoutPKL import generate_response_for_query
# Load the trained model
try:
    model = tf.keras.models.load_model('trained_model.h5')
except Exception as e:
    raise RuntimeError(f"Error loading the trained model: {str(e)}")

# Create FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with your frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic model for 
class UserInput(BaseModel):
    snoring_rate: float
    respiration_rate: float
    body_temperature: float
    limb_movement: float
    blood_oxygen: float
    eye_movement: float
    sleeping_hours: float
    heart_rate: float

# Define endpoint to receive user data and return predictions
@app.post("/predict/")
def predict_stress_level(input_data: UserInput):
    try:
        # Define acceptable ranges for input data
        ACCEPTABLE_RANGES = {
            "snoring_rate": (0, 100),
            "respiration_rate": (10, 40),
            "body_temperature": (95.0, 104.0),
            "limb_movement": (0, 50),
            "blood_oxygen": (80, 100),
            "eye_movement": (0, 30),
            "sleeping_hours": (0, 24),
            "heart_rate": (40, 180)
        }

        # Validate input ranges
        for field, value in input_data.dict().items():
            min_val, max_val = ACCEPTABLE_RANGES.get(field, (None, None))
            if min_val is not None and max_val is not None and not (min_val <= value <= max_val):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value for {field}: {value}. Must be between {min_val} and {max_val}."
                )

        # Convert input data to a list
        input_list = [[
            input_data.snoring_rate,
            input_data.respiration_rate,
            input_data.body_temperature,
            input_data.limb_movement,
            input_data.blood_oxygen,
            input_data.eye_movement,
            input_data.sleeping_hours,
            input_data.heart_rate
        ]]

        # Convert the input to a NumPy array
        input_array = np.array(input_list, dtype=np.float32)

        # Reshape input to match the model's expected shape (batch_size, time_steps, features)
        input_reshaped = input_array.reshape((input_array.shape[0], 1, input_array.shape[1]))

        # Make prediction using the trained model
        predicted_stress_level = model.predict(input_reshaped)[0][0]

        # Call generate_remedy from stresswise.py
        remedy = generate_remedy(predicted_stress_level)

        # Return both predicted stress level and the generated remedy
        return {
            "predicted_stress_level": int(predicted_stress_level),
            "remedy": remedy
        }

    except HTTPException as http_exc:
        raise http_exc  # Re-raise HTTP exceptions for invalid input
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
    

