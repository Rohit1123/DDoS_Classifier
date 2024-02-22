import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle
from sklearn.metrics import ConfusionMatrixDisplay

df_sample = pd.read_csv('dataset_sdn.csv')

X = df_sample.drop('label', axis=1)
y = df_sample['label']

numeric_cols = [col for col in X.columns if col not in ['src', 'dst', 'Protocol']]
numeric_imputer = SimpleImputer(strategy='mean')
X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])

categorical_cols = ['src', 'dst', 'Protocol']

numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(drop='first', sparse=False)
X_categorical = categorical_transformer.fit_transform(X[categorical_cols])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

X_transformed = preprocessor.fit_transform(X)

fnn_model = Sequential([
    Dense(units=64, activation='relu', input_dim=X_transformed.shape[1]),
    Dense(units=32, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

fnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train_fnn, X_test_fnn, y_train_fnn, y_test_fnn = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

fnn_model.fit(X_train_fnn, y_train_fnn, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

if y_test_fnn.isnull().any():
    print("NaN values found in y_test_fnn. Handle missing labels before evaluating the model.")
else:
    y_pred_fnn = fnn_model.predict(X_test_fnn)
    y_pred_fnn_binary = (y_pred_fnn > 0.5).astype(int)

    accuracy_fnn = accuracy_score(y_test_fnn, y_pred_fnn_binary)
    f1_fnn = f1_score(y_test_fnn, y_pred_fnn_binary)
    conf_matrix_fnn = confusion_matrix(y_test_fnn, y_pred_fnn_binary)

    print(f'FNN Accuracy: {accuracy_fnn}')
    print(f'FNN F1 Score: {f1_fnn}')
    print('FNN Confusion Matrix:')
    ConfusionMatrixDisplay(conf_matrix_fnn).plot(cmap='viridis', values_format='d')
