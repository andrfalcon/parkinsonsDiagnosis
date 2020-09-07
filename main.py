# Description: This is a web app that detects if a user has Parkinson's disease using ML and Python

# Import the libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import requests
from io import BytesIO
import s3fs
import streamlit as st

# Create a title and sub-title
st.write("""
# Parkinson's Detection
Detect if someone has Parkinson's using machine learning and Python!
""")

# Open and display thumbnail image
response = requests.get('https://parkinsons-assets.s3-us-west-1.amazonaws.com/parkinsonsLogo.jpg')
image = Image.open(BytesIO(response.content))
st.image(image, caption='ML', use_column_width=True)

# Get the data
df = pd.read_csv('s3://parkinsons-assets/parkinsonsData.csv')

# Set a sub header
st.subheader('Data Information:')

# Show the data as a table
st.dataframe(df)

# Show statistics on the data
st.write(df.describe())

# Show the data as a chart
chart = st.bar_chart(df)

# Create the feature and target data set
X = df.drop(['name'], 1)
X = np.array(X.drop(['status'],1))
y = np.array(df['status'])

# Split the data into 80% training and 20% testing data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Transform the feature data to be values between 0 and 1
sc = MinMaxScaler(feature_range=(0,1))
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Get the feature input from the user
def get_user_input():
    fo_hz = st.sidebar.slider('fo_hz', 80.0, 275.0, 125.0)
    fhi_hz = st.sidebar.slider('fhi_hz', 90.0, 600.0, 250.0)
    flo_hz = st.sidebar.slider('flo_hz', 50.0, 275.0, 130.0)
    jitter_percent = st.sidebar.slider('jitter_percent', 0.0, 1.0, 0.5)
    jitter_abs = st.sidebar.slider('jitter_abs', 0.0, 1.0, 0.5)
    rap = st.sidebar.slider('rap', 0.0, 1.0, 0.5)
    ppq = st.sidebar.slider('ppq', 0.0, 1.0, 0.5)
    ddp = st.sidebar.slider('ddp', 0.0, 1.0, 0.5)
    shimmer = st.sidebar.slider('shimmer', 0.0, 1.25, 0.75)
    shimmer_db = st.sidebar.slider('shimmer_db', 0.0, 1.5, 0.75)
    apq_three = st.sidebar.slider('apq_three', 0.0, 1.0, 0.5)
    apq_five = st.sidebar.slider('apq_five', 0.0, 1.0, 0.5)
    apq = st.sidebar.slider('apq', 0.0, 1.5, 0.75)
    dda = st.sidebar.slider('dda', 0.0, 2.0, 1.0)
    nhr = st.sidebar.slider('nhr', 0.0, 0.5, 0.25)
    hnr = st.sidebar.slider('hnr', 5.0, 40.0, 17.5)
    rpde = st.sidebar.slider('rpde', 0.0, 1.0, 0.5)
    dfa = st.sidebar.slider('dfa', 0.0, 1.0, 0.5)
    spread_one = st.sidebar.slider('spread_one', -10.0, 0.0, -5.0)
    spread_two = st.sidebar.slider('spread_two', 0.0, 1.0, 0.5)
    d_two = st.sidebar.slider('d_two', 0.0, 5.0, 2.5)
    ppe = st.sidebar.slider('ppe', 0.0, 1.0, 0.5)


    # Store a dictionary into a variable
    user_data = {
        'fo_hz' : fo_hz,
        'fhi_hz' : fhi_hz,
        'flo_hz' : flo_hz,
        'jitter_percent': jitter_percent,
        'jitter_abs' : jitter_abs,
        'rap' : rap,
        'ppq' : ppq,
        'ddp' : ddp,
        'shimmer' : shimmer,
        'shimmer_db' : shimmer_db,
        'apq_three' : apq_three,
        'apq_five' : apq_five,
        'apq' : apq,
        'dda' : dda,
        'nhr' : nhr,
        'hnr' : hnr,
        'rpde' : rpde,
        'dfa' : dfa,
        'spread_one' : spread_one,
        'spread_two' : spread_two,
        'd_two' : d_two,
        'ppe' : ppe,
    }

    # Transform the data into a data frame
    features = pd.DataFrame(user_data, index = [0])
    return features

# Store the user input into a variable
user_input = get_user_input()

# Set a subheader and display the users' input
st.subheader('User Input:')
st.write(user_input)

# Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(x_train, y_train)

# Show the models' metrics
st.subheader('Model Test Accuracy Score:')
st.write( str(accuracy_score(y_test, RandomForestClassifier.predict(x_test)) * 100) + '%' )

# Store the models' predictions in a variable
prediction = RandomForestClassifier.predict(user_input)

# Set a sub header and display the classification
st.subheader('Classification: ')
st.write(prediction)

diagnosis_certainty = str(accuracy_score(y_test, RandomForestClassifier.predict(x_test)) * 100) + '%'

if prediction == 1:
    diagnosis_statement = "There is a {} chance you have Parkinson's disease. God bless you <3".format(diagnosis_certainty)
elif prediction == 0:
    diagnosis_statement = "There is a {} chance you do not have Parkinson's Disease. God bless you <3".format(diagnosis_certainty)

st.write(diagnosis_statement)