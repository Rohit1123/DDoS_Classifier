# DDoS_Classifier
1. Data Preparation: The dataset is loaded from a CSV file (`dataset_sdn.csv`). The input features (`X`) are selected excluding the 'label' column, and the target 
                     variable (`y`) is set to the 'label' column.

2. Data Preprocessing: 
   - Missing values in numeric columns (excluding 'src', 'dst', 'Protocol') are imputed with the mean using `SimpleImputer`.
   - Categorical columns ('src', 'dst', 'Protocol') are one-hot encoded using `OneHotEncoder`.
   - Numeric columns are standardized using `StandardScaler`.

3. Column Transformation: A `ColumnTransformer` is used to apply different transformations to different columns. Numeric columns are scaled, and categorical columns 
                          are one-hot encoded.

4. Model Creation and Training:
   - A Random Forest Classifier is trained on the preprocessed data.
   - A Feedforward Neural Network (FNN) is defined using `Sequential` from Keras. It consists of an input layer with the same number of features as the preprocessed 
     data, followed by two hidden layers with 64 and 32 units respectively, using 'relu' activation. The output layer has one unit with 'sigmoid' activation for 
     binary classification.
   - The FNN is compiled with the 'adam' optimizer and 'binary_crossentropy' loss function, and trained on the preprocessed data.

5. Model Evaluation:
   - The trained FNN is evaluated on the test data.
   - If there are any missing values in the test labels (`y_test_fnn`), a message is printed, and the evaluation is skipped.
   - If there are no missing values, the FNN predicts the labels for the test data and calculates accuracy, F1 score, and confusion matrix using `accuracy_score`, 
     `f1_score`, and `confusion_matrix` functions respectively.
