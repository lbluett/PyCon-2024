import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import pickle

@st.cache_data
def load_speeches():
    # Load your data
    df = pd.read_csv('pycon_2024_sessions.csv')
    df['Title'] = df['Title'].str.replace("—PyCon AU 2024", "")
    return df
# Load your data
df_speeches = load_speeches()

@st.cache_data
def load_scores():
    # Load pickle
    with open('related_articles_scores.pkl', 'rb') as f:
        return pickle.load(f)

related_articles = load_scores()

# Streamlit UI to select a title
selected_title = st.selectbox("Select a session title", df_speeches['Title'].values)

# get selected id
selected_id = df_speeches[df_speeches['Title'] == selected_title].index[0]

# related_ids = [item['corpus_id'] for item in related_articles[selected_id]]
selected_scores = related_articles[selected_id]

# Find the top N most similar descriptions
N = st.slider("Number of similar sessions to display", 1, 10, 5)
selected_scores = selected_scores[1:N+1]

sorted_scores = sorted(selected_scores, key=lambda x: x['cross_score'], reverse=True)
# Display the most similar descriptions
col1, col2 = st.columns(2)

with col1:
    st.write(f"**{selected_title}**\n\n{df_speeches.iloc[selected_id]['Description']}")
with col2:  
    for rank, speech in enumerate(sorted_scores):
        index = int(speech['corpus_id'])  # Convert tensor index to int
        st.write(f"**{rank+1}: {df_speeches.iloc[index]['Title']}**")
        st.write(df_speeches.iloc[index]['Description'])
        st.write(f"Cross-encoder score: {speech['cross_score']:.2f}")
        st.write("----")
