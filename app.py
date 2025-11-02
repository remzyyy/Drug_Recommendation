# Import necessary libraries
import os
import joblib
import pandas as pd
import re
import nltk
import streamlit as st
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import plotly.express as px

# Download NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# Define file paths
MODEL_PATH = 'model/passmodel.pkl'
TOKENIZER_PATH = 'model/tfidfvectorizer.pkl'
DATA_PATH = 'data/custom_dataset.csv'

# Load vectorizer and model
vectorizer = joblib.load(TOKENIZER_PATH)
model = joblib.load(MODEL_PATH)

# Initialize NLTK tools
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Configure Streamlit page
st.set_page_config(
    page_title='DPDR',
    page_icon='👨‍⚕️',
    layout='wide'
)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Predict button styling */
    .stButton button {
        background-color: #89b4fa; /* Blue */
        color: #1e1e2e; /* Base (dark text) */
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #000000; /* Dark Black */
        color: #ffffff; /* White text for contrast */
    }
    .stButton button:active {
        background-color: #89dceb; /* Sky */
    }
    
    /* Condition Predicted styling */
    .condition-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #FFA500; /* Vibrant Orange */
        color: #1e1e2e; /* Base (dark text) */
        margin: 10px 0;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        font-weight: bold; /* Make text bold for emphasis */
    }

    /* Top Recommended Drugs styling */
    .drug-card-1 {
        padding: 15px;
        border-radius: 8px;
        background-color: #b4befe; /* Lavender */
        color: #1e1e2e; /* Base (dark text) */
        margin: 10px 0;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    }
    .drug-card-2 {
        padding: 15px;
        border-radius: 8px;
        background-color: #a6e3a1; /* Green */
        color: #1e1e2e; /* Base (dark text) */
        margin: 10px 0;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    }
    .drug-card-3 {
        padding: 15px;
        border-radius: 8px;
        background-color: #f38ba8; /* Red */
        color: #1e1e2e; /* Base (dark text) */
        margin: 10px 0;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    }
    .drug-card-4 {
        padding: 15px;
        border-radius: 8px;
        background-color: #f2cdcd; /* Flamingo */
        color: #1e1e2e; /* Base (dark text) */
        margin: 10px 0;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Title and header
st.title("💉 An Intelligent System for Disease Prediction and Drug Recommendation 💉")
st.markdown("---")

# Sidebar for additional options
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("Configure the model and data paths here.")
    MODEL_PATH = st.text_input("Model Path", value='model/passmodel.pkl')
    TOKENIZER_PATH = st.text_input("Tokenizer Path", value='model/tfidfvectorizer.pkl')
    DATA_PATH = st.text_input("Data Path", value='data/custom_dataset.csv')

# Predefined list of symptoms
symptoms = [
    "Acne", "Anxiety", "Arthritis", "Asthma", "Back pain", "Bipolar disorder", 
    "Birth control", "Bronchitis", "Chronic pain", "Cold", "Constipation", 
    "Cough", "Depression", "Diabetes", "Diarrhea", "Eczema", "Fatigue", 
    "Fever", "Flu", "Gastroesophageal reflux disease (GERD)", "Headache", 
    "High blood pressure", "High cholesterol", "Insomnia", "Migraine", 
    "Nausea", "Obesity", "Pain", "Pneumonia", "Psoriasis", "Sinusitis", 
    "Skin rash", "Stress", "Thyroid disorder", "Urinary tract infection (UTI)", 
    "Vomiting", "Weight loss", "Allergies", "Bladder infection", 
    "Chest pain", "Dizziness", "Ear infection", "Eye infection", 
    "Fibromyalgia", "Heartburn", "Hemorrhoids", "Indigestion", 
    "Irritable bowel syndrome (IBS)", "Joint pain", "Kidney stones", 
    "Muscle pain", "Seasonal allergies", "Sore throat", "Swelling", 
    "Toothache", "Vaginal infection", "Wheezing"
]

# Input section with multiselect for symptoms
st.header("📝 Enter Patient Symptoms")

# Radio button to choose input method
input_method = st.radio(
    "Choose how you want to enter symptoms:",
    options=["Select from predefined list", "Type your own text"],
    index=0  # Default to the first option
)

# Initialize raw_text
raw_text = ""

# Option 1: Select from predefined list
if input_method == "Select from predefined list":
    selected_symptoms = st.multiselect("Choose the symptoms:", symptoms)
    raw_text = ", ".join(selected_symptoms)

# Option 2: Type your own text
elif input_method == "Type your own text":
    raw_text = st.text_area("Describe the patient's symptoms or condition here:", height=100)

# Display the final input text (for debugging or confirmation)
st.markdown(f"**Input Text:** {raw_text}")

# Function to clean and preprocess text
def clean_text(raw_review):
    # 1. Remove HTML tags
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    
    # 2. Keep only letters and replace non-alphabetic characters with spaces
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    
    # 3. Convert to lowercase and split into words
    words = letters_only.lower().split()
    
    # 4. Remove stopwords
    meaningful_words = [w for w in words if w not in stop]
    
    # 5. Lemmatize words
    lemmatized_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    
    # 6. Join words back into a single string
    return ' '.join(lemmatized_words)

# Function to extract top recommended drugs
def top_drugs_extractor(condition, df):
    # Filter and sort drugs based on rating and usefulness
    df_top = df[(df['rating'] >= 9) & (df['usefulCount'] >= 90)].sort_values(
        by=['rating', 'usefulCount'], ascending=[False, False]
    )
    
    # Extract top 4 unique drugs for the condition
    drug_lst = df_top[df_top['condition'] == condition]['drugName'].head(4).tolist()
    drug_lst = list(set(drug_lst))
    
    return drug_lst

# Function to predict condition and recommend drugs
def predict(raw_text):
    global predicted_cond, top_drugs
    
    if raw_text != "":
        # Clean and preprocess input text
        clean_text_result = clean_text(raw_text)
        clean_lst = [clean_text_result]
        
        # Transform text using the vectorizer
        tfidf_vect = vectorizer.transform(clean_lst)
        
        # Predict condition
        prediction = model.predict(tfidf_vect)
        predicted_cond = prediction[0]
        
        # Load data and extract top drugs
        df = pd.read_csv(DATA_PATH)
        top_drugs = top_drugs_extractor(predicted_cond, df)

# Predict button
predict_button = st.button("🔍 Predict")

# Display results when the button is clicked
if predict_button:
    # Check if input is empty
    if not raw_text.strip():
        st.warning("⚠️ Please enter symptoms or select from the predefined list before predicting.")
    else:
        with st.spinner("🧠 Analyzing the condition and generating recommendations..."):
            predict(raw_text)
        
        st.markdown("---")
        st.markdown("### 🎯 Condition Predicted")
        st.markdown(f"<div class='condition-card'><h3>{predicted_cond}</h3></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 💊 Top Recommended Drugs")

        # Check if recommended drugs are less than 4
        if len(top_drugs) < 4:
            st.warning(f"⚠️ Only {len(top_drugs)} recommended drugs are available for this condition.")

        # Display recommended drugs
        for i, drug in enumerate(top_drugs):
            if i == 0:
                st.markdown(f"<div class='drug-card-1'><h4>{i + 1}. {drug}</h4></div>", unsafe_allow_html=True)
            elif i == 1:
                st.markdown(f"<div class='drug-card-2'><h4>{i + 1}. {drug}</h4></div>", unsafe_allow_html=True)
            elif i == 2:
                st.markdown(f"<div class='drug-card-3'><h4>{i + 1}. {drug}</h4></div>", unsafe_allow_html=True)
            elif i == 3:
                st.markdown(f"<div class='drug-card-4'><h4>{i + 1}. {drug}</h4></div>", unsafe_allow_html=True)
        
        # Visualize top drugs using a bar chart with matching colors
        st.markdown("---")
        st.markdown("### 📊 Drug Recommendations Visualization")
        df_drugs = pd.DataFrame({"Drug": top_drugs, "Rank": range(len(top_drugs), 0, -1)})  # Reverse ranks
        color_map = {}
        if len(top_drugs) >= 1:
            color_map[top_drugs[0]] = "#b4befe"  # Lavender
        if len(top_drugs) >= 2:
            color_map[top_drugs[1]] = "#a6e3a1"  # Green
        if len(top_drugs) >= 3:
            color_map[top_drugs[2]] = "#f38ba8"  # Red
        if len(top_drugs) >= 4:
            color_map[top_drugs[3]] = "#f2cdcd"  # Flamingo
            
        fig = px.bar(df_drugs, x="Rank", y="Drug", title="Top Recommended Drugs", 
                     labels={"Drug": "Drug Name", "Rank": "Rank"},
                     orientation='h',  # Horizontal bar chart
                     color="Drug",  # Color by drug name
                     color_discrete_map=color_map)  # Map colors to drugs
        
        fig.update_layout(yaxis={'categoryorder':'total ascending'})  # Sort by rank
        st.plotly_chart(fig, use_container_width=True)

# Warning and thank you message
st.markdown("---")
st.markdown("### ⚠️🚧 **Disclaimer** 🚧⚠️")
st.warning("""
    **This is just a prototype and not a substitute for professional medical advice.**  
    The predictions and recommendations provided by this app are based on a machine learning model and may not always be accurate.  
    Always consult a qualified healthcare provider for medical diagnosis and treatment.
""")
st.markdown("---")

# Expandable section for additional information
with st.expander("ℹ️ About This App"):
    st.write("""
        This app uses a machine learning model to predict diseases based on symptoms and recommends the top drugs for the predicted condition.
        - **Symptoms**: Select from a predefined list of symptoms.
        - **Prediction**: The model analyzes the symptoms and predicts the most likely condition.
        - **Drug Recommendations**: Based on the predicted condition, the app recommends the top drugs with high ratings and usefulness.
    """)
