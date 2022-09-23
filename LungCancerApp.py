#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import streamlit as st
import pandas as pd
import sklearn


# In[2]:


pickle_in1 = open("clf.pkl","rb")
classifier = pickle.load(pickle_in1)


# In[3]:


def prediction(Obesity,Coughing_of_Blood,Passive_smoker,Alcohol_use,Balanced_diet,Wheezing,Dust_allergy,Swallowing_difficulty,Fatigue,Shortness_of_breath,Genetic_risk,Chest_pain,Air_pollution,Snoring,Chronic_lung_disease):
    dic = {'Obesity':[Obesity],'Coughing_of_Blood':[Coughing_of_Blood],'Passive_smoker':[Passive_smoker],'Alcohol_use':[Alcohol_use],'Balanced_diet':[Balanced_diet],'Wheezing':[Wheezing],'Dust_allergy':[Dust_allergy],'Swallowing_difficulty':[Swallowing_difficulty],'Fatigue':[Fatigue],'Shortness_of_breath':[Shortness_of_breath],'Genetic_risk':[Genetic_risk],'Chest_pain':[Chest_pain],'Air_pollution':[Air_pollution],'Snoring':[Snoring],'Chronic_lung_disease':[Chronic_lung_disease]}
    df = pd.DataFrame(dic)
    pca = pickle.load(open("pca.pkl","rb"))
    processed = pca.transform(df)
    #making predictions
    prediction = classifier.predict(processed)
    return prediction


# In[7]:


def main():
    #designing the webpage
    logo_url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTIOY6jHYXMu_rQFkCLTwyCACloQElq76KUE4wuY-BrugEph2sPZ53w6ko&s=10'
    st.image(logo_url, width=100) #adding logo
    st.title("Lung Cancer Risk Level Prediction") #creating a title
    st.subheader("Know your lung cancer risk level today!") #adding a subtitle
    st.markdown(""" <style> #MainMenu {visibility: hidden;} footer {visibility:hidden;} </style> """, unsafe_allow_html=True) #hiding the menu button
    padding = 0 #reducing the padding
    st.markdown(f""" <style> .reportview-container .main .block-container{{padding-top: {padding}rem; padding-right: {padding}rem; padding-left: {padding}rem; padding-bottom: {padding}rem;}} </style> """, unsafe_allow_html=True)
    #creating the input tabs
    Obesity = st.slider('How obese are you?',min_value=1, max_value=7,step=1)
    Coughing_of_Blood = st.slider('Have you ever coughed blood? How often does this happen?',min_value=1, max_value=9,step=1)
    Passive_smoker = st.slider("How exposed would you say you are to other people's tobacco smoke?",min_value=1, max_value=8,step=1)
    Alcohol_use = st.slider("What's your alcohol intake like? Rate it on a scale of 1-8.",min_value=1, max_value=8,step=1)
    Balanced_diet = st.slider('How balanced is your diet',min_value=1, max_value=7,step=1)
    Wheezing = st.slider('Do you wheeze a lot? How severe would you say it is?',min_value=1, max_value=8,step=1)
    Dust_allergy = st.slider('Are you allergic to dust? If so, how severe is your allergy?',min_value=1, max_value=8,step=1)
    Swallowing_difficulty = st.slider('Any difficulty with swallowing? Rate it.',min_value=1, max_value=8,step=1)
    Fatigue = st.slider('How fatigued would you say you get?',min_value=1, max_value=9,step=1)
    Shortness_of_breath = st.slider('Do you experience shortness of breath? If so, how often/severe?',min_value=1, max_value=9,step=1)
    Genetic_risk = st.slider('Can you rate your genetic risk of lung cancer? You may need to carry out genetic testing for this.',min_value=1, max_value=7,step=1)
    Chest_pain = st.slider('Do you experience chest pain? How severe is it?',min_value=1, max_value=9,step=1)
    Air_pollution = st.slider('How polluted is the air you breathe?',min_value=1, max_value=8,step=1)
    Snoring = st.slider("How severe is your snoring? You may ask living partners or family members for their input.",min_value=1, max_value=7,step=1)
    Chronic_lung_disease = st.slider('Do you have any form of chronic lung disease? If so, how severe is it?',min_value=1, max_value=7,step=1)
    result="" #empty string
    #creating the predict button and output message
    if st.button("Predict"): 
        result = prediction(Obesity,Coughing_of_Blood,Passive_smoker,Alcohol_use,Balanced_diet,Wheezing,Dust_allergy,Swallowing_difficulty,Fatigue,Shortness_of_breath,Genetic_risk,Chest_pain,Air_pollution,Snoring,Chronic_lung_disease)
        if result[0] == 0:
            st.success("Your risk level is low.") 
        elif result[0] == 1:
            st.success("Your risk level is medium.")
        else:
            st.success("Your risk level is high!")
        
main()


# In[ ]:




