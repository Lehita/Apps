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


def prediction(Obesity,Coughing_of_Blood,Genetic_risk,Passive_smoker,Balanced_diet,Swallowing_difficulty,Wheezing,Dust_allergy,Fatigue,Alcohol_use,Shortness_of_breath,Chronic_lung_disease,Chest_pain,Frequent_cold,Snoring):
    if Genetic_risk == "Yes":
        Genetic_risk = 1
    else:
        Genetic_risk = 0
    dic = {'Obesity':[Obesity], 'Coughing_of_Blood':[Coughing_of_Blood], 'Genetic_risk':[Genetic_risk], 'Passive_smoker':[Passive_smoker], 'Balanced_diet':[Balanced_diet], 'Swallowing_difficulty':[Swallowing_difficulty], 'Wheezing':[Wheezing], 'Dust_allergy':[Dust_allergy], 'Fatigue':[Fatigue], 'Alcohol_use':[Alcohol_use], 'Shortness_of_breath':[Shortness_of_breath], 'Chronic_lung_disease':[Chronic_lung_disease], 'Chest_pain':[Chest_pain], 'Frequent_cold':[Frequent_cold], 'Snoring':[Snoring]}
    df = pd.DataFrame(dic) #creating the dataframe
    pca = pickle.load(open("pca.pkl","rb")) #transforming the dataframe using PCA
    processed = pca.transform(df)
    #making predictions
    prediction = classifier.predict(processed)
    return prediction


# In[4]:


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
    Obesity = st.slider('Are you obese? If yes, how obese are you?',min_value=1, max_value=7,step=1)
    Coughing_of_Blood = st.slider('Have you ever coughed blood? If yes, how much blood? Rate it.',min_value=1, max_value=9,step=1)
    Passive_smoker = st.slider("How often are you exposed to other people's smoke?",min_value=1, max_value=8,step=1)
    Alcohol_use = st.slider("What's your alcohol intake like? Rate it on a scale of 1-8.",min_value=1, max_value=8,step=1)
    Balanced_diet = st.slider('How balanced is your diet on a scale of 1-7?',min_value=1, max_value=7,step=1)
    Wheezing = st.slider('How often do you wheeze? And how severe would you say it is?',min_value=1, max_value=8,step=1)
    Dust_allergy = st.slider('Rate your level of exposure to dust on a scale of 1-8.',min_value=1, max_value=8,step=1)
    Swallowing_difficulty = st.slider('How often do you find it difficult to swallow? And how severe is it?',min_value=1, max_value=8,step=1)
    Fatigue = st.slider('How often do you experience fatigue?',min_value=1, max_value=9,step=1)
    Shortness_of_breath = st.slider('Have you ever been short of breath? Rate your experience based on severity and frequency of occurrence.',min_value=1, max_value=9,step=1)
    Genetic_risk = st.selectbox('Do you have family history of lung cancer?',("Yes","No"), index=1)
    Chest_pain = st.slider('Do you experience chest pain? If yes, how painful was the last occurrence?',min_value=1, max_value=9,step=1)
    Frequent_cold = st.slider('How often do you catch a cold?',min_value=1, max_value=7,step=1)
    Snoring = st.slider("How loudly do you snore? Inquire from living partners or family members.",min_value=1, max_value=7,step=1)
    Chronic_lung_disease = st.slider('How often do you experience chronic lung diseases (e.g. asthma, bronchitis, pneumonia, etc.)',min_value=1, max_value=7,step=1)
    
    #creating the predict button and output message
    if st.button("Predict"): 
        result = prediction(Obesity,Coughing_of_Blood,Genetic_risk,Passive_smoker,Balanced_diet,Swallowing_difficulty,Wheezing,Dust_allergy,Fatigue,Alcohol_use,Shortness_of_breath,Chronic_lung_disease,Chest_pain,Frequent_cold,Snoring)
        if result == 0:
            st.success("Your risk level is low.") 
        elif result == 1:
            st.success("Your risk level is medium.")
        else:
            st.success("Your risk level is high!")
        
main()


# In[ ]:




