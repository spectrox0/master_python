# Name Student: Alejandro Josue Velazco Rodriguez
# EIP
from decimal import Decimal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from pydantic import BaseModel, condecimal
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch


# Define the data model for prediction inputs using pydantic
class IrisFeatures(BaseModel):
    sepal_length: Decimal = condecimal(gt=0, max_digits=5, decimal_places=3)
    sepal_width: Decimal = condecimal(gt=0, max_digits=5, decimal_places=3)
    petal_length: Decimal = condecimal(gt=0, max_digits=5, decimal_places=3)
    petal_width: Decimal = condecimal(gt=0, max_digits=5, decimal_places=3)

# Load the Iris dataset
def load_iris_data() -> pd.DataFrame:
    iris: Bunch = load_iris()  # Explicitly type the return value of load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df

def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    X = df.drop('species', axis=1)
    y = df['species']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_species(model: RandomForestClassifier, features: IrisFeatures) -> str:
    # Make a prediction
    prediction = model.predict([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]])
    
    # Return the species name directly if it is a string (like 'versicolor')
    if isinstance(prediction[0], str):
        return prediction[0]
    
    # If prediction is an index, use it to get the species name
    iris_data = load_iris()
    predicted_index = int(prediction[0])
    return iris_data.target_names[predicted_index]

# Display the Streamlit Dashboard
def display_dashboard():
    # Load and prepare the data
    df = load_iris_data()
    
    # Dashboard title
    st.title("🌼 Iris Dataset Dashboard")
    
    # Display the first few rows of the dataset
    st.write("### Explore the Dataset")
    st.write(df.head())
    
    # Plot the distribution of species
    st.write("### Species Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='species', data=df, palette="viridis", ax=ax)
    st.pyplot(fig)
    
    # Plot a pairplot of the features
    st.write("### Pairplot of Features")
    fig = sns.pairplot(df, hue='species', palette="viridis")
    st.pyplot(fig)
    
    # Select features for prediction
    st.write("### Species Prediction")
    st.sidebar.write("#### Adjust the features to predict the species:")

    sepal_length = st.sidebar.slider('Sepal length (cm)', float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
    sepal_width = st.sidebar.slider('Sepal width (cm)', float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
    petal_length = st.sidebar.slider('Petal length (cm)', float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
    petal_width = st.sidebar.slider('Petal width (cm)', float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))
    
    # Validate input using pydantic
    try:
        features = IrisFeatures(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width
        )
        
        # Train model and predict species
        model = train_model(df)
        prediction_species = predict_species(model, features)
        
        # Display the prediction
        st.write(f"### Prediction: The species is **{prediction_species}**")

        # Add a scatter plot to visualize the prediction in the context of the dataset
        st.write("### Visualizing Your Prediction in the Dataset")
        fig, ax = plt.subplots()
        sns.scatterplot(x='petal length (cm)', y='petal width (cm)', hue='species', data=df, palette="viridis", ax=ax)
        ax.scatter(features.petal_length, features.petal_width, color='red', s=100, label='Your Sample')
        ax.legend()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    display_dashboard()