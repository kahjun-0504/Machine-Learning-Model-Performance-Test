import pandas as pd
from sklearn.calibration import LabelEncoder
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
import numpy as np

# Title
st.title("Machine Learning Model")
st.write("""
         ## CNS3102 Python Programming
         """)

dataset_name = st.sidebar.selectbox("Select Dataset",("Medical Dataset","Banana Quality"))
classifier_name = st.sidebar.selectbox("Select Classifier",("Random Forest","SVM","Logistic Regression"))
label_encoder = LabelEncoder()
def get_dataset(dataset_name):
    if dataset_name == "Medical Dataset":
        df = pd.read_csv('Medicaldataset.csv')
        # Drop the 'Age' and 'Gender' columns from heart_disease_df and update the DataFrame
        df.drop(columns=['Gender'], inplace=True)
        X = df.drop('Result',axis=1)
        y = df['Result']
        
        # Convert the 'Result' column to numerical values using LabelEncoder for Heart Attack
        df['Result'] = label_encoder.fit_transform(df['Result'])

        # Verify the changes
        print(df.head(150))
        print(df['Result'].unique())  # Check the unique values to see the encoded labels

    else:
        df = pd.read_csv('banana_quality.csv')
        X = df.drop('Quality',axis=1)
        y = df['Quality']
        
        # Convert the 'Result' column to numerical values using LabelEncoder for Heart Attack
        df['Quality'] = label_encoder.fit_transform(df['Quality'])

        # Verify the changes
        print(df.head(150))
        print(df['Quality'].unique())  # Check the unique values to see the encoded labels
    st.write(df.head())
    return X,y

X, y = get_dataset(dataset_name)
st.write(f"## {dataset_name}")


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

# Function to train and evaluate classifiers
def evaluate_classifier(clf, X_train, y_train, X_test, y_test):
    start_train = time.time()
    clf.fit(X_train, y_train)
    end_train = time.time()

    start_test = time.time()
    y_pred = clf.predict(X_test)
    end_test = time.time()

    accuracy = accuracy_score(y_test, y_pred) * 100  # Convert accuracy to percentage
    train_time = (end_train - start_train) * 1000
    test_time = (end_test - start_test) * 1000

    return accuracy, train_time, test_time

# Evaluate classifiers
results = {
    'Classifier': [],
    'Accuracy (%)': [],
    'Train Time (ms)': [],
    'Test Time (ms)': []
}

for name, clf in classifiers.items():
    accuracy, train_time, test_time = evaluate_classifier(clf, X_train, y_train, X_test, y_test)
    results['Classifier'].append(name)
    results['Accuracy (%)'].append(accuracy)
    results['Train Time (ms)'].append(train_time)
    results['Test Time (ms)'].append(test_time)

# Create a DataFrame to display the results
results_df = pd.DataFrame(results)

# Define the plotting function
def plot_model_vs_accuracy(dataset_name):
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#FF9999', '#66B2FF', '#99FF99']  # Color palette
    bars = ax.bar(results_df['Classifier'], results_df['Accuracy (%)'], color=colors)

    # Add data labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_title(f'Model vs Accuracy ({dataset_name})', fontsize=16)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the plot in Streamlit
    st.pyplot(fig)

# Call the function to display the plot in Streamlit
plot_model_vs_accuracy(dataset_name)


# Plot Train Time vs Model
def plot_train_time_vs_model(dataset_name):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define a color palette
    colors = ['#FF9999', '#66B2FF', '#99FF99']  # You can customize the colors here

    bars = ax.bar(results_df['Classifier'], results_df['Train Time (ms)'], color=colors)

    # Add data labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f} ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    ax.set_title(f'Train Time vs Model ({dataset_name})', fontsize=16)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Train Time (ms)', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Display the plot in Streamlit
    st.pyplot(fig)


plot_train_time_vs_model(dataset_name)


# Plot Test Time vs Model
def plot_test_time_vs_model(dataset_name):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define a color palette
    colors = ['#FF9999', '#66B2FF', '#99FF99']  # You can customize the colors here

    bars = ax.bar(results_df['Classifier'], results_df['Test Time (ms)'], color=colors)

    # Add data labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f} ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_title(f'Test Time vs Model ({dataset_name})', fontsize=16)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Test Time (ms)', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Display the plot in Streamlit
    st.pyplot(fig)

plot_test_time_vs_model(dataset_name)

# Display the selected classifier's results
def get_classifier(classifier_name):
    selected_classifier = results_df[results_df['Classifier'] == classifier_name]
    if not selected_classifier.empty:
        accuracy = selected_classifier['Accuracy (%)'].values[0]
        train_time = selected_classifier['Train Time (ms)'].values[0]
        test_time = selected_classifier['Test Time (ms)'].values[0]

        st.write(f"## Results for {classifier_name}")
        st.write(f"Classifier: {classifier_name}")
        st.write(f"Accuracy: {accuracy:.2f}%")
        st.write(f"Train Time (ms): {train_time:.2f} ms")
        st.write(f"Test Time (ms): {test_time:.2f} ms")
    else:
        st.write("Please select a classifier.")

# Call the function to display the selected classifier's results
get_classifier(classifier_name)
