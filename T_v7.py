#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
import openai
import time
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Set your OpenAI API key
openai.api_key = 'f21c398476b146988044b391bd5256df'

# Function to get tweets based on keywords
REQUIRE_COLUMNS = ['conversation_id_str' , 'user_id_str', 'created_at', 'lang', 'full_text', 'bookmark_count',  
                   'favorite_count', 'quote_count', 'reply_count', 'retweet_count']

def get_tweets_by_explore(keywords="finance"):
    headers = {
        "x-rapidapi-key": "625a7d077amshfb106d4c100ee6ap1a8c14jsn87cd99088de2",
        "x-rapidapi-host": "twitter241.p.rapidapi.com"
    }

    url = "https://twitter241.p.rapidapi.com/search-v2"
    querystring = {"type": "Top", "count": "20", "query": keywords}
    response = requests.get(url, headers=headers, params=querystring)
    data_dict = response.json()
    print("Status Code:", response.status_code)
    
    while response.status_code != 200:
        time.sleep(3)
        response = requests.get(url, headers=headers, params=querystring)
        print("Status Code:", response.status_code)
        data_dict = response.json()
    
    result = [r['content'] for r in data_dict['result']['timeline']['instructions'][0]['entries']]

    csv_data = []
    for r in result:
        if 'itemContent' in r:
            if 'tweet_results' in r['itemContent']:
                if 'legacy' in r['itemContent']['tweet_results']['result']:
                    tweet = r['itemContent']['tweet_results']['result']['legacy']
                    csv_data.append(tweet)

    df = pd.DataFrame(csv_data)
    df = df.loc[:,REQUIRE_COLUMNS]
    df = df.drop_duplicates(subset=['conversation_id_str' , 'user_id_str', 'full_text', 'bookmark_count',  
                                    'favorite_count', 'quote_count', 'reply_count', 'retweet_count'], keep='first')

    df = df.reset_index()
    return df

#keywords = "banking in hong kong"
#df = get_tweets_by_explore(keywords) # Get Tweets Need times Please wait
#df


# In[2]:


from transformers import is_torch_tpu_available
from bertopic import BERTopic

# Function to analyze tweets and create a recommendation
def analyze_tweets(df):
    if df.empty:
        return "No tweets found for the given keywords."

    # Topic modeling using BERTopic
    vectorizer_model = CountVectorizer(stop_words='english')
    topic_model = BERTopic(vectorizer_model=vectorizer_model)
    topics, _ = topic_model.fit_transform(df['full_text'])

    df['topic'] = topics
    st.write(df)
    #topic_model.visualize_barchart()
    #topic_model.visualize_heatmap()
    #topic_model.visualize_hierarchy()

    # Count metrics
    retweet_count = df['retweet_count'].sum()
    comment_count = df['reply_count'].sum()

    # Create bar chart with bank's corporate colors
    #labels = ['Retweets', 'Comments']
    #values = [retweet_count, comment_count]

    #plt.figure(figsize=(8, 5))
    #bars = plt.bar(labels, values, color=['#800000', '#D3D3D3'])  # Burgundy and Grey
    #plt.title('Twitter Engagement Metrics', fontsize=16)
    #plt.xlabel('Metrics', fontsize=14)
    #plt.ylabel('Count', fontsize=14)

    # Add value labels on top of bars
    #for bar in bars:
        #yval = bar.get_height()
        #plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom', fontsize=12, color='black')

    # Save the plot to a file
    #plt.tight_layout()
    #plt.savefig('engagement_metrics.png')

    # Generate recommendation
    recommendation = generate_recommendation(retweet_count, comment_count)

    return recommendation


# In[3]:


# Function to generate recommendation using OpenAI's model
def generate_recommendation(retweet_count, comment_count):
    prompt = f"You are a marketing campaign expert who works in a bank. Based on the retweet count of {retweet_count} and comment count of {comment_count}, suggest possible marketing campaign strategies for a bank, which can create values to client while bring in sales to the bank. You may prioritise credit card and investment products."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

def main():
    # Streamlit UI
    st.set_page_config(page_title="Twitter Comment Analysis", layout="wide")

    # Set background image with Victoria Habour view
    st.markdown(
        '''
        <style>
        .stApp {
            background-image: url('	https://images.squarespace-cdn.com/content/v1/614e…/Victoria+Peak+Hong+Kong+%282%29.jpg?format=1500w');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            height: 100vh;
        }
        </style>
        ''',
        unsafe_allow_html=True
    )

    # Add HSBC logo
    # logo = Image.open('./hsbc-1.png')  # Replace with the actual path to the logo
    # st.image(logo, use_column_width=True)

    st.title("Twitter Comment Analysis for Marketing Campaigns")
    st.markdown("<h2 style='color: #800000;'>Analyze Twitter Comments Effectively</h2>", unsafe_allow_html=True)

    user_input = st.text_input("Enter keywords (e.g., Faker, Credit Card, etc):", placeholder="Type your keywords here...")

    if st.button("Analyze"):
        with st.spinner("Fetching tweets..."):
            df = get_tweets_by_explore(user_input)

            recommendation = analyze_tweets(df)

            #st.image('engagement_metrics.png')
            st.write("Recommendation for Marketing Campaign:")
            st.write(recommendation)

    # Add some professional icons (You can use FontAwesome or similar)
    st.markdown(
        '''
        <style>
        .icon {
            display: inline-block;
            margin: 0 10px;
        }
        </style>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        '''
        <div class="icon">
            <i class="fa fa-twitter" style="font-size: 24px; color: #800000;"></i>
        </div>
        <div class="icon">
            <i class="fa fa-chart-bar" style="font-size: 24px; color: #800000;"></i>
        </div>
        <div class="icon">
            <i class="fa fa-comments" style="font-size: 24px; color: #800000;"></i>
        </div>
        ''',
        unsafe_allow_html=True
    )


    hide = """
    <style>
    div[data-testid="stConnectionStatus"] {
        display: none !important;
    </style>
    """

    st.markdown(hide, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


# In[ ]:




