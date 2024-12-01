import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from datetime import datetime
import pytz
import google.generativeai as genai
from IPython.display import display, Markdown

# Google API Key setup for generative AI
os.environ["API_KEY"] = "AIzaSyD9zU2jqbpQ-BXdg2Q180v6jjeGN0iboKw"
genai.configure(api_key=os.environ["API_KEY"])
DocterApparao = genai.GenerativeModel("gemini-1.5-flash")

# Load your dataset from a CSV file
dataset = pd.read_csv("Final_Dataset.csv")

# Separate features and target variable
X = dataset.drop(columns=['sl'])  # Features
y = dataset['sl']  # Target variable

# Ideal ranges for each feature
ideal_ranges = {
    'Snoring Rate': (0, 100),
    'Respiration Rate': (0, 50),
    'Body Temperature': (96, 100),
    'Limb Movement': (0, 20),
    'Blood Oxygen': (0, 100),
    'Eye Movement': (0, 100),
    'Sleeping Hours': (0, 24),
    'Heart Rate': (0, 200)
}

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler to a file
joblib.dump(scaler, 'scaler.pkl')

# Reshape the features for LSTM input
X = X.reshape(X.shape[0], 1, X.shape[1])  # Reshape for LSTM

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if the trained model exists; if not, train a new one
if not os.path.isfile("trained_model.h5"):
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.5))  # Dropout layer for regularization
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output layer for regression

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_test, y_test))

    # Save the trained model
    model.save("trained_model.h5")
else:
    # Load the trained model
    model = load_model("trained_model.h5")

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test)
display(Markdown(f'Test MAE: {test_mae:.4f}'))

def predict_stress_level(data, scaler, model):
    # Normalize the new data using the same scaler used for training data
    new_data_scaled = scaler.transform(data)
    new_data_reshaped = new_data_scaled.reshape(new_data_scaled.shape[0], 1, new_data_scaled.shape[1])

    # Make predictions using the LSTM model
    predicted_stress_level = model.predict(new_data_reshaped)[0][0]
    return predicted_stress_level

def generate_remedy(stress_level):
    prompt = f"I'm experiencing a stress level of {stress_level}. Can you recommend evidence-based techniques or remedies to manage stress effectively? Give response in 5 lines"
    response = DocterApparao.generate_content(prompt)
    return response.text

def save_data_to_file(user_name, user_age, stress_level, remedy):
    file_name = f"{user_name}_stress_data.txt"
    indian_timezone = pytz.timezone('Asia/Kolkata')
    timestamp = datetime.now(indian_timezone).strftime("%Y-%m-%d %H:%M:%S")
    with open(file_name, 'a') as file:
        file.write(f"Name: {user_name}, Age: {user_age}, Stress Level: {stress_level}, Remedy: {remedy}, Timestamp: {timestamp}\n")

def check_user_file(user_name):
    file_name = f"{user_name}_stress_data.txt"
    return os.path.isfile(file_name)

def get_new_data_from_user():
    display(Markdown("Please provide the following sleep-related behavior values:"))
    data = []
    user_name = input("What's your name? ")
    user_age = int(input("What's your age? "))
    for feature, (min_val, max_val) in ideal_ranges.items():
        value = float(input(f"{feature} ({min_val} - {max_val}): "))
        data.append(value)
    return user_name, user_age, data

def main():
    while True:
        try:
            user_name, user_age, data = get_new_data_from_user()
            predicted_stress_level = predict_stress_level([data], scaler, model)
            predicted_stress_label = int(round(predicted_stress_level))
            display(Markdown(f"Predicted Stress Level: {predicted_stress_label}"))

            if not check_user_file(user_name):
                display(Markdown("Creating a new file for the user."))
            remedy = generate_remedy(predicted_stress_label)
            display(Markdown(f"Remedy: {remedy}"))

            save_data_to_file(user_name, user_age, predicted_stress_label, remedy)

            conversation = input("Is there anything else I can assist you with? (yes/no): ")
            if conversation.lower() == 'no':
                break
            elif conversation.lower() == 'yes':
                while True:
                    query = input("Ask me any health-related question or type 'exit' to go back to the main menu: ")
                    if query.lower() == 'exit':
                        return
                    else:
                        response = DocterApparao.generate_content(query).text
                        display(Markdown("Health Instructor: " + response['choices'][0]['message']['content']))
        except Exception as e:
            display(Markdown("An error occurred: " + str(e)))

main()
