import numpy as np
import pickle
import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
import predict

loaded_model = pickle.load(open('assets/trained_model.sav', 'rb'))

def stroke_predictio(input_data):
   
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict_proba(input_data_reshaped)[:, 1][0]
    prediction_proba = loaded_model.predict_proba(input_data_reshaped)
    print( prediction)
    return prediction_proba, prediction

def main():
    st.title('Stroke Prediction')

    image = Image.open('assets/87343087.jpg')
    st.image(image, caption='Stroke Prediction')
    st.write("""
    The World Health Organization (WHO) identifies strokes as the second leading cause of death globally. A stroke happens when a person’s blood supply to their brain is interrupted or reduced, causing brain cells to die within minutes. It prevents the brain tissue from getting the oxygen and nutrients that it needs and is responsible for approximately 11% of total deaths.

    The website aims at classifying the stroke based on the input parameters like gender, age, various diseases, and smoking status. Since, the project is related to medical domain multiple models were trained and their performance was compared considering the sensitivity, accuracy, as well as specificity scores in the course: CSL2050 Pattern Recognition and Machine Learning under Prof. Richa Singh.
    """)

    with st.sidebar:
        st.sidebar.header('User Input')
        gender = st.selectbox("Gender",('Male', 'Female', 'Other'))
        _gender = gender
        if(gender == 'Male'):
            gender = 1
        elif(gender == 'Female'):
            gender = 0
        else:
            gender = 2

        age = st.slider('Age', 1, 120)
        _age = age
        age = (age-43.22661448140902)/(22.61043402711303)

        hypertension = st.radio("Hypertension",('Yes', 'No'))
        _hypertension = hypertension
        if(hypertension == 'Yes'):
            hypertension = 1
        else:
            hypertension = 0

        heart_disease = st.radio("Heart Disease",('Yes', 'No'))
        _heart_disease = heart_disease
        if(heart_disease == 'Yes'):
            heart_disease = 1
        else:
            heart_disease = 0

        ever_married = st.radio("Ever Married",('Yes', 'No'))
        _ever_married = ever_married
        if(ever_married == 'Yes'):
            ever_married = 1
        else:
            ever_married = 0

        work_type = st.selectbox("Work Type",('Government Job', 'Never Worked', 'Private', 'Self-employed', 'Children'))
        _work_type = work_type
        if(work_type == 'Government Job'):
            work_type = 0
        elif(work_type == 'Never Worked'):
            work_type = 1
        elif(work_type == 'Private'):
            work_type = 2
        elif(work_type == 'Self-employed'):
            work_type = 3
        else:
            work_type = 4
        
        Residence_type = st.radio("Residence Type",('Rural', 'Urban'))
        _Residence_type = Residence_type
        if(Residence_type == "Rural"):
            Residence_type = 0
        else:
            Residence_type = 1

        avg_glucose_level = st.slider('Average Glucose Level',1,350)
        _avg_glucose_level = avg_glucose_level
        avg_glucose_level = (avg_glucose_level-106.14767710371795)/(45.27912905705893)

        bmi = st.slider('BMI',5,100)
        _bmi = bmi
        bmi = (bmi-28.90337865973328)/(7.698534094073452)

        smoking_status = st.selectbox("Smoking Status",('Formerly Smoked', 'Never Smoked', 'Smokes', 'Unknown'))
        _smoking_status = smoking_status
        if(smoking_status == "Formerly Smoked"):
            smoking_status = 1
        elif(smoking_status == "Never Smoked"):
            smoking_status = 2
        elif(smoking_status == "Smokes"):
            smoking_status = 3
        else:
            smoking_status = 0

        diagnosis = 0
        
        if st.button('Stroke Test Result'):
            prediction_proba, diagnosis = stroke_predictio([gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status])
            diagnosis=diagnosis*100
    data = {'Gender': _gender,
            'Age': _age,
            'Hypertension': _hypertension,
            'Heart Disease': _heart_disease,
            'Ever Married': _ever_married,
            'Work Type': _work_type,
            'Residence Type' : _Residence_type,
            'Avg Glucose Level': _avg_glucose_level,
            'BMI': _bmi,
            'Smoking Status': _smoking_status
        }
    features = pd.DataFrame(data, index=[0])

    input_df = pd.DataFrame(features)
    strokes = pd.DataFrame(columns=["Gender", "Age", "Hypertension", "Heart Disease", "Ever Married", "Work Type", "Residence Type", "Avg Glucose Level", "BMI", "Smoking Status"])

    df = pd.concat([input_df,strokes],axis=0)

    df_input = df[:1]
    st.subheader('Input features')
    st.write(df_input)

    if(diagnosis == 0):
        st.info("Please press 'Stroke Test Result' button for prediction!!")
    elif(diagnosis >= 75):
        st.error(f'You have {diagnosis}% chance of having a stroke. Please consult a Neurologist.')
        st.subheader('Prediction Probabilities')
        st.write(prediction_proba)
    elif(diagnosis >= 40 and diagnosis < 75):
        st.warning(f'You have {diagnosis}% chance of having a stroke. It is advised to take precautions.')
        st.subheader('Prediction Probabilities')
        st.write(prediction_proba)
    else:
        st.success(f'You have {diagnosis}% chance of having a stroke.')
        st.subheader('Prediction Probabilities')
        st.write(prediction_proba)

def page2():
    TITLE = "Brain Stroke Detector"
    STROKE_STYLE = "padding: 20px; background-color: #f44336; color: white; margin-bottom: 15px; text-align: center; text-style: bold; font-size: 24px;"
    STROKE_MESSAGE = "Stroke"
    HEALTHY_STYLE = "padding: 20px; background-color: #4cbb17; color: white; margin-bottom: 15px; text-align: center; text-style: bold; font-size: 24px;"
    HEALTHY_MESSAGE = "Healthy"
    ERROR_STYLE = "padding: 20px; background-color: #ffc300; color: white; margin-bottom: 15px; text-align: center; text-style: bold; font-size: 24px;"

    def print_outcome(outcome):
        """Outcome printing function"""
        if outcome:
            st.write(f'<div style="{STROKE_STYLE}">{STROKE_MESSAGE}</div>', unsafe_allow_html=True)
        else:
            st.write(f'<div style="{HEALTHY_STYLE}">{HEALTHY_MESSAGE}</div>', unsafe_allow_html=True)   
    def print_error(error):
        """Error print function"""
        st.write(f'<div style="{ERROR_STYLE}">Error: {error}</div>', unsafe_allow_html=True)
    def fin():
        """Main function"""
        # st.set_page_config(page_title=TITLE)
        st.title(TITLE)
        uploaded_img = st.file_uploader("Upload a CT scan to be analyzed")
        if uploaded_img is not None:
            try:
                stroke = predict.predict(uploaded_img)
                print_outcome(stroke)
                st.image(uploaded_img, caption='Uploaded CT scan')
            except Exception as e:
                print_error(e)
    fin() 


    # Add the content of page 2 here

# Create a dictionary of pages
pages = {
    "Page 1": main,
    "Page 2": page2
}

# Create a selectbox for navigation
page = st.sidebar.selectbox("Choose a page", list(pages.keys()))

# Display the selected page
pages[page]()