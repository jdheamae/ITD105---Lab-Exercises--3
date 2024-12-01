
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, AdaBoostRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, Perceptron, ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR, SVC
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import joblib
import io
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pandas import read_csv
# Streamlit app title
def main():
    st.title("Machine Learning Models: Classification and Regression")
    # Load sample dataset (replace with your dataset)

    # Sidebar for selecting the problem type and model
    st.sidebar.title("Model Selection")
    problem_type = st.sidebar.radio("Choose Problem Type:", ["Classification", "Regression"])
    problem_type2 = st.sidebar.selectbox("Choose an option:", ["Sampling Techniques", "Hyperparameter Tuning"])

    data = load_iris()
    X = data.data
    Y = data.target

    def split_data(test_size, random_seed):
        data = load_iris()
        X = data.data
        Y = data.target
        return train_test_split(X, Y, test_size=test_size, random_state=random_seed)


    @st.cache_data
    def heart_disease(uploaded_file):
        names = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
                'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium', 'Heart Disease']
        dataframe = pd.read_csv(uploaded_file, names=names, header=0)
        return dataframe

    @st.cache_data
    def forest_fire(uploaded_file):
        names = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 
                'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']
        dataframe = pd.read_csv(uploaded_file, names=names, header=0)
        
        # Convert month and day to numeric values for model processing
        month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        day_mapping = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}
        
        dataframe['month'] = dataframe['month'].map(month_mapping)
        dataframe['day'] = dataframe['day'].map(day_mapping)
        
        return dataframe
    # ---- Initialize session state variables ----
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "dataframe" not in st.session_state:
        st.session_state.dataframe = None
    if "X_train" not in st.session_state:
        st.session_state.X_train = None
    if "X_test" not in st.session_state:
        st.session_state.X_test = None
    if "Y_train" not in st.session_state:
        st.session_state.Y_train = None
    if "Y_test" not in st.session_state:
        st.session_state.Y_test = None
    if "sampling_complete" not in st.session_state:
        st.session_state.sampling_complete = False
    if "random_seed" not in st.session_state:
        st.session_state.random_seed = 42  # Default seed value
    if "test_size" not in st.session_state:
        st.session_state.test_size = 0.2 
    @st.cache_data
    def heart_disease(uploaded_file):
        names = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
                'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium', 'Heart Disease']
        dataframe = pd.read_csv(uploaded_file, names=names, header=0)
        
    @st.cache_data
    def heart_disease(uploaded_file):
        names = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
                'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium', 'Heart Disease']
        dataframe = pd.read_csv(uploaded_file, names=names, header=0)
        return dataframe

    # Function for user input in Heart Disease model
    if problem_type == "Classification":
        if problem_type2  == "Sampling Techniques":
                filename = 'C:/Users/USER/Documents/lab105/lab3/Heart_Disease_Prediction.csv'
                dataframe = read_csv(filename)
                array = dataframe.values
                X = array[:, 0:13]
                Y = array[:, 13]
                # ---- SAMPLING TECHNIQUE SECTION ----
                st.header("Sampling Technique")
                st.write("Upload your dataset and split it into training and testing sets.")

                # File uploader for CSV
                uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
                if uploaded_file:
                    st.session_state.uploaded_file = uploaded_file
                    dataframe = heart_disease(uploaded_file)
                    st.session_state.dataframe = dataframe

                    st.write("Preview of the processed dataset:")
                    st.dataframe(dataframe.head())

                    X = dataframe.drop(columns=['Heart Disease']).values  # Feature matrix
                    Y = dataframe['Heart Disease'].values  # Target column

                    # Set the test size using a slider
                    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test_size")
                    random_seed = st.slider("Random Seed", 1, 100, 7, key="random_seed")

                    # Split the dataset into test and train
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

                    # Save train-test splits to session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.Y_train = Y_train
                    st.session_state.Y_test = Y_test
                    st.session_state.sampling_complete = True  # Mark sampling as complete

                    st.write("Data has been split into training and test sets.")

                else:
                    st.info("Please upload a CSV file to proceed.")

                # ---- ML accuracy----
                if st.session_state.sampling_complete:
                    st.subheader("")

                    # List of models and names
                    models = [
                        ("Decision Tree", DecisionTreeClassifier()),
                        ("Gaussian Naive Bayes", GaussianNB()),
                        ("AdaBoost", AdaBoostClassifier(n_estimators=50, random_state=st.session_state.random_seed)),
                        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
                        ("Logistic Regression", LogisticRegression(max_iter=200)),
                        ("MLP Classifier", MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu')),
                        ("Perceptron", Perceptron()),
                        ("Random Forest", RandomForestClassifier(n_estimators=100)),
                        ("Support Vector v", SVC(kernel="linear"))
                    ]

                    # Store results in a dictionary
                    accuracy_results = {}

                    # Loop through models, train, and evaluate accuracy
                    for model_name, model in models:
                        model.fit(st.session_state.X_train, st.session_state.Y_train)
                        train_accuracy = accuracy_score(st.session_state.Y_train, model.predict(st.session_state.X_train))
                        test_accuracy = accuracy_score(st.session_state.Y_test, model.predict(st.session_state.X_test))
                        accuracy_results[model_name] = test_accuracy  # Store only test accuracy

                    # Sort models by accuracy
                    sorted_accuracy = sorted(accuracy_results.items(), key=lambda x: x[1], reverse=True)

                    accuracy_df = pd.DataFrame(sorted_accuracy, columns=["Model", "Test Accuracy"])

                    # Display the model with the highest accuracy using st.info
                    highest_model = accuracy_df.loc[accuracy_df["Test Accuracy"].idxmax()]

                    # Display results in a table with color highlights
                    st.write("Model Accuracy Comparison:")
                    accuracy_df = pd.DataFrame(sorted_accuracy, columns=["Model", "Test Accuracy"])

                    # Add color highlights to the table
                    def highlight_min_max(s):
                        if s.name == "Test Accuracy":
                            is_max = s == s.max()
                            is_min = s == s.min()
                            return ["background-color: lightgreen" if v else "background-color: lightcoral" if m else "" for v, m in zip(is_max, is_min)]
                        return [""] * len(s)

                    styled_table = accuracy_df.style.apply(highlight_min_max, axis=0)
                    st.dataframe(styled_table)

                    # Create a bar chart for model accuracies
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Get colors for the bars
                    colors = ["lightcoral" if acc == accuracy_df["Test Accuracy"].min()
                            else "lightgreen" if acc == accuracy_df["Test Accuracy"].max()
                            else "steelblue" for acc in accuracy_df["Test Accuracy"]]

                    sns.barplot(x='Model', y='Test Accuracy', data=accuracy_df, ax=ax, palette=colors)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                    ax.set_title("Model Accuracy Comparison")
                    st.pyplot(fig)
                    st.info(f"The highest performing model is **{highest_model['Model']}** with an accuracy of **{highest_model['Test Accuracy']:.2f}**.")

        elif problem_type2 == "Hyperparameter Tuning":
                        # ---- Hyper Tunign----
                    st.subheader("Hyperparameter Tuning")



                    # Algorithm selection dropdown
                    selected_model_name = st.selectbox("Select a Machine Learning Algorithm", [
                        "Gaussian Naive Bayes", "AdaBoost", "Logistic Regression", "MLP Classifier", "Random Forest"
                    ])

                    # Hyperparameter sliders and tuning results
                    if selected_model_name == "Gaussian Naive Bayes":

                        filename = 'C:/Users/USER/Documents/lab105/lab3/Heart_Disease_Prediction.csv'
                        dataframe = read_csv(filename)
                        array = dataframe.values
                        X = array[:, 0:13]
                        Y = array[:, 13]
                        @st.cache_data
                        def heart_disease(uploaded_file):
                            names = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
                                    'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium', 'Heart Disease']
                            dataframe = pd.read_csv(uploaded_file, names=names, header=0)
                            return dataframe
                        
                        if "random_seed" not in st.session_state:
                            st.session_state.random_seed = 42  # Default seed value
                        if "test_size" not in st.session_state:
                            st.session_state.test_size = 0.2 
                        
                        test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test_size")
                        random_seed = st.slider("Random Seed", 1, 100, 7, key="random_seed")
                        var_smoothing = st.number_input("Var Smoothing (Log Scale)", min_value=-15, max_value=-1, value=-9, step=1, key="var_smoothing")

                        # Convert var_smoothing from log scale to regular scale
                        var_smoothing_value = 10 ** var_smoothing

                        # Split the dataset into training and testing sets
                        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

                        # Initialize the Gaussian Naive Bayes classifier with hyperparameters
                        model = GaussianNB(var_smoothing=var_smoothing_value)

                        # Train the model on the training data
                        model.fit(X_train, Y_train)

                        # Evaluate the accuracy
                        accuracy = model.score(X_test, Y_test)

                        # Display the accuracy in the app
                        st.write(f"Accuracy: {accuracy * 100.0:.3f}%")

                    elif selected_model_name == "AdaBoost":
                        filename = 'C:/Users/USER/Documents/lab105/lab3/Heart_Disease_Prediction.csv'
                        dataframe = read_csv(filename)
                        array = dataframe.values
                        X = array[:, 0:13]
                        Y = array[:, 13]
                        @st.cache_data
                        def heart_disease(uploaded_file):
                            names = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
                                    'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium', 'Heart Disease']
                            dataframe = pd.read_csv(uploaded_file, names=names, header=0)
                            return dataframe
                        if "random_seed" not in st.session_state:
                            st.session_state.random_seed = 42  # Default seed value
                        if "test_size" not in st.session_state:
                            st.session_state.test_size = 0.2 

                        test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test_size2")
                        random_seed = st.slider("Random Seed", 1, 100, 7, key="random_seed2")
                        n_estimators = st.slider("Number of Estimators", 1, 100, 50)

                        # Split the dataset into training and testing sets
                        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

                        # Create an AdaBoost classifier
                        model = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_seed)

                        # Train the model on the training data
                        model.fit(X_train, Y_train)

                        # Evaluate the accuracy
                        accuracy = model.score(X_test, Y_test)

                        # Display the accuracy in the app
                        st.write(f"Accuracy: {accuracy * 100.0:.3f}%")
                        
                    elif selected_model_name == "Logistic Regression":
                        if "random_seed" not in st.session_state:
                            st.session_state.random_seed = 42  # Default seed value
                        if "test_size" not in st.session_state:
                            st.session_state.test_size = 0.2 
                        filename = 'C:/Users/USER/Documents/lab105/lab3/Heart_Disease_Prediction.csv'
                        dataframe = read_csv(filename)
                        array = dataframe.values
                        X = array[:, 0:13]
                        Y = array[:, 13]
                        @st.cache_data
                        def heart_disease(uploaded_file):
                            names = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
                                    'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium', 'Heart Disease']
                            dataframe = pd.read_csv(uploaded_file, names=names, header=0)
                            return dataframe
                            
                        test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test4")
                        random_seed = st.slider("Random Seed", 1, 100, 7, key="seed4")
                        max_iter = st.slider("Max Iterations", 100, 500, 200)
                        solver = st.selectbox("Solver", options=["lbfgs", "liblinear", "sag", "saga", "newton-cg"])
                        C = st.number_input("Inverse of Regularization Strength", min_value=0.01, max_value=10.0, value=1.0)

                # Split the dataset into training and testing sets
                        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

                        # Create a Logistic Regression model
                        model = LogisticRegression(max_iter=max_iter, solver=solver, C=C)

                        # Train the model on the training data
                        model.fit(X_train, Y_train)

                        # Evaluate the accuracy
                        accuracy = model.score(X_test, Y_test)

                        # Display the accuracy in the app
                        st.write(f"Accuracy: {accuracy * 100.0:.3f}%")

                    elif selected_model_name == "Random Forest":
                        if "random_seed" not in st.session_state:
                            st.session_state.random_seed = 42  # Default seed value
                        if "test_size" not in st.session_state:
                            st.session_state.test_size = 0.2 
                        filename = 'C:/Users/USER/Documents/lab105/lab3/Heart_Disease_Prediction.csv'
                        dataframe = read_csv(filename)
                        array = dataframe.values
                        X = array[:, 0:13]
                        Y = array[:, 13]
                        @st.cache_data
                        def heart_disease(uploaded_file):
                            names = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
                                    'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium', 'Heart Disease']
                            dataframe = pd.read_csv(uploaded_file, names=names, header=0)
                            return dataframe
                        
                        test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test7")
                        random_seed = st.slider("Random Seed", 1, 100, 7, key="seed7")
                        n_estimators = st.slider("Number of Estimators (Trees)", 10, 200, 100)
                        max_depth = st.slider("Max Depth of Trees", 1, 50, None)  # Allows None for no limit
                        min_samples_split = st.slider("Min Samples to Split a Node", 2, 10, 2)
                        min_samples_leaf = st.slider("Min Samples in Leaf Node", 1, 10, 1)

                        # Split the dataset into training and testing sets
                        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

                        # Create a Random Forest classifier
                        rfmodel = RandomForestClassifier(
                            n_estimators=n_estimators,
                            random_state=random_seed,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf
                        )

                        # Train the model
                        rfmodel.fit(X_train, Y_train)

                        # Evaluate the accuracy
                        accuracy = rfmodel.score(X_test, Y_test)

                        # Display the accuracy in the app
                        st.write(f"Accuracy: {accuracy * 100.0:.3f}%")

                    elif selected_model_name == "MLP Classifier":
                        if "random_seed" not in st.session_state:
                            st.session_state.random_seed = 42  # Default seed value
                        if "test_size" not in st.session_state:
                            st.session_state.test_size = 0.2 
                        filename = 'C:/Users/USER/Documents/lab105/lab3/Heart_Disease_Prediction.csv'
                        dataframe = read_csv(filename)
                        array = dataframe.values
                        X = array[:, 0:13]
                        Y = array[:, 13]
                        @st.cache_data
                        def heart_disease(uploaded_file):
                            names = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
                                    'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium', 'Heart Disease']
                            dataframe = pd.read_csv(uploaded_file, names=names, header=0)
                            return dataframe
                        
                        test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test5")
                        random_seed = st.slider("Random Seed", 1, 100, 7, key="seed5")
                        hidden_layer_sizes = st.text_input("Hidden Layer Sizes (e.g., 65,32)", "65,32")
                        activation = st.selectbox("Activation Function", options=["identity", "logistic", "tanh", "relu"])
                        max_iter = st.slider("Max Iterations", 100, 500, 200, key="max5")

                        # Convert hidden_layer_sizes input to tuple
                        hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(',')))

                        # Split the dataset into training and testing sets
                        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

                        # Create an MLP-based model
                        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, 
                                            solver='adam', max_iter=max_iter, random_state=random_seed)

                        # Train the model
                        model.fit(X_train, Y_train)

                        # Evaluate the accuracy
                        accuracy = model.score(X_test, Y_test)

                        # Display the accuracy in the app
                        st.write(f"Accuracy: {accuracy * 100.0:.3f}%")

                    # Save tuned model
                    if st.button(f"Save Tuned {selected_model_name} Model"):
                        buffer = io.BytesIO()
                        joblib.dump(model, buffer)
                        buffer.seek(0)
                        st.download_button(
                            label=f"Download Tuned {selected_model_name} Model",
                            data=buffer,
                            file_name=f"{selected_model_name.replace(' ', '_')}_tuned_model.pkl",
                            mime="application/octet-stream"
                        )
    elif problem_type == "Regression":
        if problem_type2  == "Sampling Techniques":
                filename = 'C:/Users/USER/Documents/lab105/lab3/forestfires.csv'
                dataframe = read_csv(filename)
                print(dataframe.head())

                array = dataframe.values
                X = array[:, 0:12]
                Y = array[:, 12]
                # ---- SAMPLING TECHNIQUE SECTION ----
                st.header("Sampling Technique")
                st.write("Upload your dataset and split it into training and testing sets.")

                # File uploader for CSV
                uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
                if uploaded_file:
                    st.session_state.uploaded_file = uploaded_file
                    dataframe = forest_fire(uploaded_file)
                    st.session_state.dataframe = dataframe

                    st.write("Preview of the processed dataset:")
                    st.dataframe(dataframe.head())

                    # Handle missing values with imputation
                    imputer = SimpleImputer(strategy='mean')  # You can change to 'median' or 'most_frequent'
                    X = dataframe.drop(columns=['area']).values  # Feature matrix
                    X = imputer.fit_transform(X)  # Impute missing values
                    Y = dataframe['area'].values  # Target column

                    # Set the test size using a slider
                    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test_size")
                    random_seed = st.slider("Random Seed", 1, 100, 7, key="random_seed")

                    # Split the dataset into test and train
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

                    # Save train-test splits to session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.Y_train = Y_train
                    st.session_state.Y_test = Y_test
                    st.session_state.sampling_complete = True  # Mark sampling as complete

                    st.write("Data has been split into training and test sets.")

                else:
                    st.info("Please upload a CSV file to proceed.")

                if st.session_state.sampling_complete:
                    X_train = st.session_state.X_train
                    X_test = st.session_state.X_test
                    Y_train = st.session_state.Y_train
                    Y_test = st.session_state.Y_test
                    st.subheader("Model Evaluation using MAE")

                    # List of regression models
                    models = [
                        ("Decision Tree Regressor", DecisionTreeRegressor()),
                        ("Elastic Net", ElasticNet()),
                        (" Regressor", AdaBoostRegressor(n_estimators=50, random_state=st.session_state.random_seed)),
                        ("K-Nearest Neighbors", KNeighborsRegressor(n_neighbors=5)),
                        ("Lasso Regression", Lasso()),
                        ("Ridge Regression", Ridge()),
                        ("Linear Regression", LinearRegression()),
                        ("MLP Regressor", MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000)),
                        ("Random Forest Regressor", RandomForestRegressor(n_estimators=100)),
                        ("Support Vector Regressor (SVR)", SVR(kernel="linear"))
                    ]

                    # Store results in a dictionary
                    mae_results = {}

                    # Loop through models, train, and evaluate MAE
                    for model_name, model in models:
                        model.fit(X_train, Y_train)
                        predictions = model.predict(X_test)
                        mae = mean_absolute_error(Y_test, predictions)
                        mae_results[model_name] = mae  # Store MAE

                    # Sort models by MAE (ascending order, lower is better)
                    sorted_results = sorted(mae_results.items(), key=lambda x: x[1])
                    st.session_state.regression_results = sorted_results

                    results_df = pd.DataFrame(sorted_results, columns=["Model", "Test MAE"])

                    # Display the raw DataFrame
                    st.write("Model MAE Comparison:")
                    st.write(results_df)

                    # Highlight the best performing model
                    best_model = results_df.loc[results_df["Test MAE"].idxmin()]

                    # Create a bar chart for model MAE
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Dynamically assign colors
                    min_mae = results_df["Test MAE"].min()
                    max_mae = results_df["Test MAE"].max()
                    colors = ["lightgreen" if mae == min_mae else "lightcoral" if mae == max_mae else "steelblue" for mae in results_df["Test MAE"]]

                    sns.barplot(x="Model", y="Test MAE", data=results_df, ax=ax, palette=colors)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                    ax.set_title("Model MAE Comparison")
                    st.pyplot(fig)
                    st.info(f"The best performing model is: {best_model['Model']} with MAE: {best_model['Test MAE']:.3f}")
                
        elif problem_type2  == "Hyperparameter Tuning":
                    st.subheader("Hyperparameter Tuning")

                                    # Algorithm selection dropdown
                    selected_model_name = st.selectbox("Select a Machine Learning Algorithm", [
                        "Support Vector Regressor", "K-Nearest Neighbors", "Elastic Net", "Lasso and Ridge Regressor"
                    ])

                    # Common split logic
                    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
                    random_seed = st.slider("Random Seed", 1, 100, 42)
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

                    # Initialize `model` for saving
                    model = None

                    if selected_model_name == "Support Vector Regressor":
                        filename = 'C:/Users/USER/Documents/lab105/lab3/forestfires.csv'
                        dataframe = read_csv(filename)
                        array = dataframe.values
                        X = array[:, 0:12]
                        Y = array[:, 12]
                        # User-defined parameters for SVR
                        kernel = st.selectbox("Kernel", options=['linear', 'poly', 'rbf', 'sigmoid'], index=2)
                        C = st.slider("Regularization Parameter (C)", min_value=0.01, max_value=100.0, value=1.0, step=0.01)
                        epsilon = st.slider("Epsilon", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
                        
                        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

                        # Train and evaluate SVR model
                        model = SVR(kernel=kernel, C=C, epsilon=epsilon)
                        model.fit(X_train, Y_train)

                        svr_model = SVR(kernel=kernel, C=C, epsilon=epsilon)
                        svr_model.fit(X_train, Y_train)
                        predictions = svr_model.predict(X_test)

                        # Calculate and display MAE
                        mae = mean_absolute_error(Y_test, predictions)
                        st.write(f"SVM Mean Absolute Error (MAE): **{mae:.3f}**")


                    elif selected_model_name == "K-Nearest Neighbors":
                        filename = 'C:/Users/USER/Documents/lab105/lab3/forestfires.csv'
                        dataframe = read_csv(filename)
                        array = dataframe.values
                        X = array[:, 0:12]
                        Y = array[:, 12]
                        # User inputs
                        n_neighbors = st.slider("Number of Neighbors", 1, 20, 5, 1)
                        weights = st.selectbox("Weights", ["uniform", "distance"])
                        algorithm = st.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"])

                        # Train the K-NN model
                        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
                        model.fit(X_train, Y_train)

                    elif selected_model_name == "Elastic Net":
                        filename = 'C:/Users/USER/Documents/lab105/lab3/forestfires.csv'
                        dataframe = read_csv(filename)
                        array = dataframe.values
                        X = array[:, 0:12]
                        Y = array[:, 12]
                        # User-defined parameters for Elastic Net
                        alpha = st.slider("Alpha (Regularization Strength)", 0.0, 5.0, 1.0, 0.1)
                        l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5, 0.01)
                        max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100)

                        # Train the Elastic Net model
                        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=random_seed)
                        model.fit(X_train, Y_train)

                    elif selected_model_name == "Lasso and Ridge Regressor":
                        filename = 'C:/Users/USER/Documents/lab105/lab3/forestfires.csv'
                        dataframe = read_csv(filename)
                        array = dataframe.values
                        X = array[:, 0:12]
                        Y = array[:, 12]
                        # User-defined parameters for Lasso and Ridge
                        alpha = st.slider("Regularization Parameter (alpha)", 0.01, 10.0, 1.0, 0.01)
                        max_iter = st.slider("Maximum Iterations", 100, 1000, 1000, 100)

                        # Train the Lasso model
                        lasso_model = Lasso(alpha=alpha, max_iter=max_iter, random_state=random_seed)
                        lasso_model.fit(X_train, Y_train)

                        # Train the Ridge model
                        ridge_model = Ridge(alpha=alpha, max_iter=max_iter, random_state=random_seed)
                        ridge_model.fit(X_train, Y_train)

                        # Let the user choose which model to save
                        model_choice = st.radio("Select Model to Save", ["Lasso", "Ridge"])
                        model = lasso_model if model_choice == "Lasso" else ridge_model

                    # Predict and calculate metrics
                    if model is not None:
                        Y_pred = model.predict(X_test)
                        mae = mean_absolute_error(Y_test, Y_pred)
                        residuals = Y_test - Y_pred
                        std_dev = np.std(residuals)

                        # Display results
                        st.write(f"Mean Absolute Error (MAE): {mae:.3f} ± {std_dev:.3f}")

                        # Save the trained model
                        if st.button(f"Save Tuned {selected_model_name} Model"):
                            buffer = io.BytesIO()
                            joblib.dump(model, buffer)
                            buffer.seek(0)
                            st.download_button(
                                label=f"Download Tuned {selected_model_name} Model",
                                data=buffer,
                                file_name=f"{selected_model_name.replace(' ', '_')}_tuned_model.pkl",
                                mime="application/octet-stream"
                            )
                    else:
                        st.warning("Please select and train a model before saving.")
if __name__ == "__main__":
    main()