import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score, accuracy_score

def main():
    st.title('Binary classification web-app using streamlit')
    st.sidebar.title('Binary classfication')
    st.markdown("Are your mushrooms eatable or poisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms eatable or poisonous? 🍄")

# Lets load the mushroom data using pandas dataframe and transform it

@st.cache_data(persist=True) # This is the decorator function not to load the data everytime we change the code and save the file

# Function to load the data into pandas dataframe
def load_data():         
    dataframe = pd.read_csv('mushrooms.csv')
    label = LabelEncoder()
    for col in dataframe.columns:
        dataframe[col] = label.fit_transform(dataframe[col])
    return dataframe

@st.cache_data(persist=True)

# Function to split the data using train_test_split method
def split_data(mushroom_df):
    y = mushroom_df['type']
    X = mushroom_df.drop(columns=['type'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, split_data=0.2, random_state=True)
    return X_train, X_test, y_train, y_test

#Funtion to plot the metrics
def plot_metrics(metrics_list):
    if 'Confusion Metrics' in metrics_list:
        st.subheader('Displaying the Confusion metrics')
        # ConfusionMatrixDisplay(y_test, prediction_svc, display_labels = class_name)
        st.pyplot()

    if 'ROC Curve' in metrics_list:
        st.subheader('Displaying the ROC Curve')
        # RocCurveDisplay()
        st.pyplot()

    if 'Precision Recall curve' in metrics_list:
        st.subheader('Displaing Precision Recall curve')
        # PrecisionRecallDisplay()
        st.pyplot()
    
    



mushroom_df = load_data()
class_name = ['eatable', 'poisonous']


if st.sidebar.checkbox("Show raw data", False):
    st.subheader("Mushroom Data set (classification)")
    st.write(mushroom_df)


if __name__ == '__main__':
    main()
