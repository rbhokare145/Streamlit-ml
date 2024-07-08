import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Lets define the main funtion with title and markdown information
def main():
    st.title('Binary classification web-app using streamlit')
    st.sidebar.title('Binary classfication')
    st.markdown("Are your mushrooms eatable or poisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms eatable or poisonous? 🍄")

# Lets load the mushroom data using pandas dataframe and transform it
# Function to load the data into pandas dataframe

@st.cache_data(persist=True) # This is the decorator function not to load the data everytime we change the code and save the file
def load_data():         
    dataframe = pd.read_csv('mushrooms.csv')
    label = LabelEncoder()
    for col in dataframe.columns:
        dataframe[col] = label.fit_transform(dataframe[col])
    return dataframe

# Function to split the data using sikitlearn train_test_split method 

@st.cache_data(persist=True)
def split_data(df):
    y = df['type']
    X = df.drop(columns=['type'])
    X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size = 0.2, shuffle=True)
    return X_train, X_test, y_train, y_test

# Funtion to plot the model evaluation metrics 
def plot_metrics(metrics_list):
    st.set_option('deprecation.showPyplotGlobalUse', False) # To disable warning with pyplot
    if 'Confusion Metrics' in metrics_list:
        st.subheader('Displaying the Confusion metrics')
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        st.pyplot()

    if 'ROC Curve' in metrics_list:
        st.subheader('Displaying the ROC Curve')
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        st.pyplot()

    if 'Precision Recall Curve' in metrics_list:
        st.subheader('Displaing Precision Recall curve')
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
        st.pyplot()
    
    

mushroom_df = load_data()
X_train, X_test, y_train, y_test = split_data(mushroom_df)
class_name = ['eatable', 'poisonous']
st.sidebar.subheader("Select Classifier")
classifier = st.sidebar.selectbox("classifier", ("Support Vector Machine(SVM)",
                                                 "Logistic Regression",
                                                 "Random Forest"))

if classifier == "Support Vector Machine(SVM)" :
    st.sidebar.subheader("Select Model Hyperparameter")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step = 0.01, key = 'C')
    Kernal = st.sidebar.radio("Kernal", ("rbf", "linear"), key ='Kernal')
    gamma = st.sidebar.radio("Gamma (Kernal coefficient)", ("scale", "auto"), key = 'Gamma')

    metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Metrics', 'ROC Curve', 'Precision Recall Curve'))

    if st.sidebar.button("Classify", key = "Classify"):
        st.subheader("Support Vector Machine(SVM) Result")
        model = SVC(C=C, kernel=Kernal, gamma=gamma)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write("Accuracy:", accuracy)
        # st.write("Prediction:", y_pred)
        st.write("Precision:", precision_score(y_test, y_pred, labels=class_name).round(2))
        st.write("Recall:", recall_score(y_test, y_pred, labels=class_name).round(2))
        plot_metrics(metrics)


if classifier == "Logistic Regression" :
    st.sidebar.subheader("Select Model Hyperparameter")
    C_LR = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step = 0.01, key = 'C_LR')
    max_iter = st.sidebar.slider("Max number of iteration", 100, 500, key = 'max_iter')
    metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Metrics', 'ROC Curve', 'Precision Recall Curve'))

    if st.sidebar.button("Classify", key = "Classify"):
        st.subheader("Logistic Regression")
        model = LogisticRegression(C=C_LR, max_iter=max_iter)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write("Accuracy:", accuracy)
        # st.write("Prediction:", y_pred)
        st.write("Precision:", precision_score(y_test, y_pred, labels=class_name).round(2))
        st.write("Recall:", recall_score(y_test, y_pred, labels=class_name).round(2))
        plot_metrics(metrics)


if classifier == "Random Forest":
    st.sidebar.subheader("Select Model Hyperparameter")
    n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimator')
    max_depth = st.sidebar.number_input("The Max depth of the trees", 1, 20, step=1, key='max_depth')
    bootstrap = st.sidebar.radio("Bootstrap sample while building tree", ('True', 'False'), key='bootstrap')
    metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Metrics', 'ROC Curve', 'Precision Recall Curve'))
    if st.sidebar.button("Classify", key = "Classify"):
        st.subheader("Random Forest")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write("Accuracy:", accuracy)
        # st.write("Prediction:", y_pred)
        st.write("Precision:", precision_score(y_test, y_pred, labels=class_name).round(2))
        st.write("Recall:", recall_score(y_test, y_pred, labels=class_name).round(2))
        plot_metrics(metrics)


if st.sidebar.checkbox("Show raw data", False):
    st.subheader("Mushroom Data set (classification)")
    st.write(mushroom_df)


# The code block beneath __main__ will execute only if the script run directly and not when imported as module
if __name__ == '__main__':
    main()
