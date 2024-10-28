import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def load_encoders(encoders_path='models/label_encoders.pkl'):
    if not os.path.exists(encoders_path):
        raise FileNotFoundError(f"Encoders file '{encoders_path}' not found.")
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    return encoders

def load_trained_model(model_path='models/model.h5'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    model = load_model(model_path)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def preprocess_input(age, gender, distance, area, encoders):
    try:
        gender_encoded = encoders['gender'].transform([gender])[0]
        distance_encoded = encoders['distance'].transform([distance])[0]
        area_encoded = encoders['area'].transform([area])[0]
    except ValueError as e:
        raise ValueError(f"Erro ao transformar rótulo: {str(e)}")

    input_data = np.array([[age, gender_encoded, distance_encoded, area_encoded]])
    return input_data

import tensorflow as tf
import logging
tf.get_logger().setLevel('ERROR')