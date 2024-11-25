#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8


# In[2]:


#pip install PyPDF2


# In[ ]:

# In[3]:


#pip install bertopic


# In[1]:

# In[4]:


pip install openai


# In[ ]:


import streamlit as st
import requests
import numpy as np
from openai import OpenAI
from openai import AzureOpenAI
import os
import json
from PyPDF2 import PdfReader
from bertopic import BERTopic


# In[2]:

# Part 1: Fetch and Process Policy Document

# In[ ]:


def fetch_policy_content():
    """Fetch the content of the HKUST policy document."""
    url = "https://legal.hkust.edu.hk/files/Policy_Use_of_University_Titles_Names_and_Logos.pdf"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open("policy.pdf", "wb") as file:
                file.write(response.content)
            return "policy.pdf"
        else:
            st.error(f"Failed to fetch the document. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"An error occurred while fetching the document: {e}")
        return None


# In[ ]:


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"An error occurred while extracting text from the PDF: {e}")
        return None


# In[3]:

# In[ ]:


from openai import AzureOpenAI


# In[ ]:


openai_client = AzureOpenAI(
  api_key = "f21c398476b146988044b391bd5256df", # use your key here
  api_version = "2024-06-01", # apparently HKUST uses a deprecated version
  azure_endpoint = "https://hkust.azure-api.net" # per HKUST instructions
)


# In[ ]:


def get_embeddings(text_list, model="text-embedding-ada-002"):
    """Get embeddings for a list of texts."""
    try:
        response = openai_client.embeddings.create(
            input=text_list,
            model=model,
            encoding_format="float",
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        st.error(f"An error occurred while generating embeddings: {e}")
        return None

                 


# In[4]:

# Part 3: Cosine Similarity

# In[ ]:


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# In[5]:

# In[ ]:


def search_relevant_section(query_embedding, paragraph_embeddings, paragraphs, topics):
    """Perform a simple semantic search to find the relevant section in the policy document."""
    similarities = [cosine_similarity(query_embedding, paragraph_embedding) for paragraph_embedding in paragraph_embeddings]

    if not similarities:
        st.error("No valid paragraph embeddings were generated.")
        return None, None

    most_relevant_index = np.argmax(similarities)
    most_relevant_score = similarities[most_relevant_index]

    if most_relevant_score < 0.5:  # Set a threshold for relevance
        return None, most_relevant_score

    return paragraphs[most_relevant_index], most_relevant_score, topics[most_relevant_index]


# In[ ]:


# Use BERTopic to define topics
def define_topics(paragraphs):
    """Use BERTopic to extract topics from paragraphs."""
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(paragraphs)
    return topic_model.get_topic_info(), topics


# In[6]:

# Streamlit App

# In[ ]:


# Streamlit app
def main():
    st.title("HKUST Policy Q&A")
    st.write("Ask questions about the HKUST policy document.")

    if "policy_content" not in st.session_state:
        st.session_state["policy_content"] = None
    if "policy_embeddings" not in st.session_state:
        st.session_state["policy_embeddings"] = None
    if "policy_topics" not in st.session_state:
        st.session_state["policy_topics"] = None

    if st.session_state["policy_content"] is None:
        pdf_path = fetch_policy_content()
        if pdf_path:
            content = extract_text_from_pdf(pdf_path)
            if content:
                st.session_state["policy_content"] = content
                paragraphs = content.split("\n\n")  # Splitting by double newlines
                st.session_state["policy_embeddings"] = get_embeddings(paragraphs)
                topic_info, topics = define_topics(paragraphs)
                st.session_state["policy_topics"] = topics
                st.session_state["topic_info"] = topic_info

    user_query = st.text_input("Your Question:", placeholder="Type your question here...")
    if st.button("Submit"):
        if user_query:
            query_embedding = get_embeddings([user_query])
            if query_embedding and st.session_state["policy_content"]:
                most_relevant_section, similarity_score, relevant_topic = search_relevant_section(
                    query_embedding[0],  # Use the first (and only) embedding
                    st.session_state["policy_embeddings"],
                    st.session_state["policy_content"].split("\n\n"),
                    st.session_state["policy_topics"]
                )
                if most_relevant_section:
                    st.subheader("Response:")
                    st.write(f"**Topic**: {relevant_topic}")
                    st.write(most_relevant_section)
                    st.write(f"**Relevance Score**: {similarity_score:.2f}")
                else:
                    st.write("No relevant section found.")
            else:
                st.error("An error occurred while processing your query.")
        else:
            st.error("Please type a question!")


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:

# In[ ]:
