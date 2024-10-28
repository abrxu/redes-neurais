import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import os
import pickle

np.random.seed(42)
tf.random.set_seed(42)

def generate_data(area, total, percentages, age_ranges, gender_distribution, distance_distribution):
    num_accepted = int(total * percentages['accepted'])
    num_rejected = total - num_accepted

    ages_accepted = np.random.choice(range(18, 71), num_accepted, p=create_age_distribution(age_ranges['accepted']))
    genders_accepted = np.random.choice(gender_distribution['genders'], num_accepted, p=gender_distribution['prob_accepted'])
    distances_accepted = np.random.choice(distance_distribution['distances'], num_accepted, p=distance_distribution['prob_accepted'])
    area_accepted = np.array([area] * num_accepted)

    ages_rejected = np.random.choice(range(18, 71), num_rejected)
    genders_rejected = np.random.choice(gender_distribution['genders'], num_rejected)
    distances_rejected = np.random.choice(distance_distribution['distances'], num_rejected)
    area_rejected = np.array([area] * num_rejected)

    accepted = np.column_stack((ages_accepted, genders_accepted, distances_accepted, area_accepted, np.ones(num_accepted)))
    rejected = np.column_stack((ages_rejected, genders_rejected, distances_rejected, area_rejected, np.zeros(num_rejected)))

    return np.vstack((accepted, rejected))

def create_age_distribution(percentage_by_age):
    age_distribution = np.zeros(70 - 18 + 1)
    ranges = [range(18, 23), range(23, 29), range(29, 71)]
    probabilities = percentage_by_age['prob']
    for i, age_range in enumerate(ranges):
        for age in age_range:
            age_distribution[age - 18] = probabilities[i] / len(age_range)
    return age_distribution

def main():
    health_data = generate_data(
        'Saúde', 
        15000,
        percentages={'accepted': 0.78},
        age_ranges={'accepted': {'prob': [0.25, 0.65, 0.10]}},
        gender_distribution={'genders': ['M', 'F'], 'prob_accepted': [0.22, 0.78]},
        distance_distribution={'distances': ['<15km', '>15km'], 'prob_accepted': [0.65, 0.35]}
    )

    tech_data = generate_data(
        'Tecnologia',
        15000,
        percentages={'accepted': 0.63},
        age_ranges={'accepted': {'prob': [0.65, 0.25, 0.10]}},
        gender_distribution={'genders': ['M', 'F'], 'prob_accepted': [0.83, 0.17]},
        distance_distribution={'distances': ['<15km', '>15km'], 'prob_accepted': [0.85, 0.15]}
    )

    business_data = generate_data(
        'Gestão/Negócios',
        15000,
        percentages={'accepted': 0.71},
        age_ranges={'accepted': {'prob': [0.20, 0.50, 0.30]}},
        gender_distribution={'genders': ['M', 'F'], 'prob_accepted': [0.44, 0.56]},
        distance_distribution={'distances': ['<15km', '>15km'], 'prob_accepted': [0.75, 0.25]}
    )

    all_data = np.vstack((health_data, tech_data, business_data))

    le_gender = LabelEncoder()
    le_distance = LabelEncoder()
    le_area = LabelEncoder()

    all_genders = ['M', 'F']
    all_distances = ['<15km', '>15km']
    all_areas = ['Saúde', 'Tecnologia', 'Gestão/Negócios']

    le_gender.fit(all_genders)
    le_distance.fit(all_distances)
    le_area.fit(all_areas)

    all_data[:, 1] = le_gender.transform(all_data[:, 1])
    all_data[:, 2] = le_distance.transform(all_data[:, 2])
    all_data[:, 3] = le_area.transform(all_data[:, 3])

    os.makedirs('models', exist_ok=True)
    encoders = {
        'gender': le_gender,
        'distance': le_distance,
        'area': le_area
    }
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    print("Label encoders salvos em 'models/label_encoders.pkl'.")

    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame(all_data, columns=['Idade', 'Gênero', 'Distância', 'Área', 'Matriculado'])
    df.to_csv('data/data.csv', index=False)
    print("Dados salvos em 'data/data.csv'.")

    X = all_data[:, :-1].astype(float)
    y = all_data[:, -1].astype(float)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_train_accuracies = []
    fold_test_accuracies = []

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))

        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            epochs=200,
            batch_size=20,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )

        train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

        fold_train_accuracies.append(train_accuracy)
        fold_test_accuracies.append(test_accuracy)

        print(f'Fold {fold}: Acurácia de Treinamento = {train_accuracy:.4f}, Acurácia de Teste = {test_accuracy:.4f}')

    print(f'\nMédia de Acurácia de Treinamento: {np.mean(fold_train_accuracies):.4f}')
    print(f'Média de Acurácia de Teste: {np.mean(fold_test_accuracies):.4f}')

    final_model = Sequential()
    final_model.add(Input(shape=(X.shape[1],)))
    final_model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    final_model.add(Dropout(0.3))
    final_model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
    final_model.add(Dropout(0.3))
    final_model.add(Dense(1, activation='sigmoid'))

    final_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping_final = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    final_model.fit(
        X, y,
        epochs=200,
        batch_size=20,
        callbacks=[early_stopping_final],
        verbose=0
    )

    final_model.save('models/model.h5')
    print("Modelo final salvo em 'models/model.h5'.")

if __name__ == "__main__":
    main()