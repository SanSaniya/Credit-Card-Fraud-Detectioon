import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import base64

#import data
data = pd.read_csv('dataset/creditcard.csv')


#seperate legitimate and fraud data
legit = data[data.Class==0]
fraud = data[data.Class==1]

#undersample legitimate tx to balance the class
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis = 0)

#split data into training and testing datasets
X = data.drop(columns="Class", axis =1)
Y = data['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)


#initializing shap & loading model & scaleing data
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")
explainer = shap.Explainer(model, scaler.transform(X_train)) 

#train logistic regression model    
#model = LogisticRegression(max_iter=100)
#model.fit(X_train,Y_train)

#evaluate model performance
train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test), Y_test)



#web-app
st.title("Credit Card fraud detection Model")
st.write("Simulate a transaction by adjusting the values below:")

#User input
time_input = st.number_input("⏱️ Time Since Last Transaction (seconds)", value=10, step=60)
amount_input = st.number_input("💳 Transaction Amount ($)", value=2000000.00, step=1.0)





# now i add default values from v1 to v28 bcuz user don't know that.. 
# i filled mean values here.
# since my model is already trained so it won't affect the accuracy either
#all columns except for class time and amount get the dafult mean value
default_values = legit.drop(columns=["Class", "Time", "Amount"]).mean().to_dict()



# Time
input_data = [time_input]  
# V1 to V28
input_data.extend([default_values[f"V{i}"] for i in range(1, 29)])  
# Amount
input_data.append(amount_input)  
# ['Time', 'V1', ..., 'Amount']
#feature_names = X.columns.tolist()  

# Dynamically create input fields for each feature
#input_df_splited = []
#for feature in feature_names:
#    value = st.number_input(f"{feature}", value=0.0, step=0.01)
#    input_df_splited.append(value)
    
#input_df = st.text_input('Enter All 29 feature values')
#input_df_splited = input_df.split(',')


#func to plot shap explanation as an img
def get_shap_plot(input_data):
    shap_values = explainer(input_data)         #Genertae shap values
    
    plt.figure(figsize=(10, 5))
    shap.plots.bar(shap_values[0], show=False)       #generate the plot
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")      # Save the plot to buffer
    buf.seek(0)
    plt.close()
    img_str = base64.b64encode(buf.read()).decode("utf-8") # Encode image as base64 string
    return img_str


submit = st.button("Submit")

if submit:
    try:
        features = np.asarray(input_data, dtype=np.float64).reshape(1,-1)        # Convert user input to numpy array
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)              #Reshape & Predict the outcome using the model

    
        if prediction[0]==0:
            st.write("✅ Legitimate Transaction")
        else:
            st.write("⚠️ Fraudulant Transaction")
        #func to plot shap explanation as an img

        shap_img_str = get_shap_plot(scaled_features)
        st.image(f"data:image/png;base64,{shap_img_str}", use_container_width=True)

    except Exception as e:
        st.warning(f"⚠️ Something went wrong. Error: {e}")