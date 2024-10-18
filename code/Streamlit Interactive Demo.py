import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import torch
import pickle

# Load your data
df = pd.read_csv('pycon_2024_sessions.csv')  # Assuming you have the DataFrame from the previous step
df['Title'] = df['Title'].str.replace("—PyCon AU 2024", "")

# Load pickle
with open('related_articles_scores.pkl', 'rb') as f:
    related_articles = pickle.load(f)

st.set_page_config(layout="wide")

# Streamlit UI to select a title
selected_title = st.selectbox("Select a session title", df['Title'].values)

# get selected id
selected_id = df[df['Title'] == selected_title].index[0]

related_ids = [item['corpus_id'] for item in related_articles[selected_id]]

# Find the top N most similar descriptions
N = st.slider("Number of similar sessions to display", 1, 10, 5)
top_n_indices = related_ids[1:N + 1]  # Skip the first as it's the selected one

# Display the most similar descriptions
col1, col2 = st.columns(2)

with col1:
    st.write(f"**{selected_title}**\n\n{df.iloc[selected_id]['Description']}")
with col2:  
    for rank, index in enumerate(top_n_indices):
        index = int(index)  # Convert tensor index to int
        st.write(f"**{rank+1}: {df.iloc[index]['Title']}**")
        st.write(df.iloc[index]['Description'])
        st.write("----")
