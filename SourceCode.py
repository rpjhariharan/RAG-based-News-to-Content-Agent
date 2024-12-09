import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import chromadb
from chromadb.config import Settings
import openai
from datetime import datetime, timedelta
import bcrypt

# Set Streamlit page configuration as the very first Streamlit command
st.set_page_config(page_title="RAG-based Content Generator", layout="wide")

# Access API Keys and Credentials from Streamlit Secrets
NEWS_API_KEY = st.secrets["news_api_key"]
OPENAI_API_KEY = st.secrets["openai_api_key"]
IMGFLIP_USERNAME = st.secrets.get("imgflip_username", "")
IMGFLIP_PASSWORD = st.secrets.get("imgflip_password", "")
VIDEO_API_KEY = st.secrets.get("video_api_key", "")  # For video generation API

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Initialize local Chroma DB for vector storage
client = chromadb.Client(Settings(
    persist_directory=".chromadb"  # Directory to persist the database
))
collection = client.get_or_create_collection("news_articles")

# Function to generate embeddings using OpenAI's Embedding API
def get_openai_embeddings(texts):
    try:
        response = openai.Embedding.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        embeddings = [data['embedding'] for data in response['data']]
        return embeddings
    except Exception as e:
        st.error(f"Error generating embeddings with OpenAI: {e}")
        return []

def get_openai_embedding(text):
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embedding = response['data'][0]['embedding']
        return embedding
    except Exception as e:
        st.error(f"Error generating embedding with OpenAI: {e}")
        return []

# Define NewsAPI only
NEWS_APIS = {
    "NewsAPI": {
        "url": "https://newsapi.org/v2/everything",
        "headers": {},
        "params": lambda query, limit: {"q": query, "language": "en", "pageSize": limit, "apiKey": NEWS_API_KEY},
        "parse": lambda data: [
            {**article, "source": "NewsAPI"} for article in data.get("articles", [])
        ]
    }
}

def fetch_news_autonomously(query, limit=5):
    articles = []
    for source in NEWS_APIS:
        if len(articles) >= limit:
            break
        fetched = fetch_from_source(source, query, limit - len(articles))
        articles.extend(fetched)
    return articles

def fetch_from_source(source, query, limit=5):
    api = NEWS_APIS[source]
    try:
        headers = api.get("headers", {})
        params = api["params"](query, limit)
        response = requests.get(api["url"], headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            articles = api["parse"](data)
            return articles[:limit] if articles else []
        else:
            st.warning(f"Failed to fetch from {source}: Status code {response.status_code}")
            return []
    except Exception as e:
        st.warning(f"Error fetching from {source}: {e}")
        return []

def sanitize_metadata(metadata):
    sanitized = {}
    for key, value in metadata.items():
        if value is None:
            sanitized[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized

def upsert_articles_to_chroma(articles):
    try:
        texts = [a.get('content', '') for a in articles]
        embeddings = get_openai_embeddings(texts)
        if not embeddings:
            st.error("Failed to generate embeddings for the articles.")
            return
        ids = [f"doc_{i}_{int(datetime.now().timestamp())}" for i in range(len(articles))]  # Unique IDs
        metadatas = [sanitize_metadata(a) for a in articles]
        collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
    except Exception as e:
        st.error(f"Error upserting articles to ChromaDB: {e}")

def retrieve_relevant_articles(query, k=3):
    try:
        query_embedding = get_openai_embedding(query)
        if not query_embedding:
            st.error("Failed to generate embedding for the query.")
            return [], []
        results = collection.query(query_embeddings=[query_embedding], n_results=k)
        if results and results["documents"]:
            docs = results["documents"][0]
            meta = results["metadatas"][0]
            return docs, meta
        return [], []
    except Exception as e:
        st.error(f"Error retrieving articles from ChromaDB: {e}")
        return [], []

def generate_image(prompt_text):
    if OPENAI_API_KEY:
        try:
            response = openai.Image.create(
                prompt=prompt_text,
                n=1,
                size="512x512"
            )
            img_url = response['data'][0]['url']
            return img_url
        except Exception as e:
            st.warning(f"OpenAI Image API unavailable, using placeholder image. Error: {e}")
    return "https://via.placeholder.com/512x512?text=Image+Unavailable"

def generate_meme(template_id, caption):
    if IMGFLIP_USERNAME and IMGFLIP_PASSWORD:
        api_url = "https://api.imgflip.com/caption_image"
        params = {
            "template_id": template_id,
            "username": IMGFLIP_USERNAME,
            "password": IMGFLIP_PASSWORD,
            "text0": caption,
            "text1": ""
        }
        try:
            r = requests.post(api_url, params=params)
            if r.status_code == 200:
                result = r.json()
                if result.get("success"):
                    return result["data"]["url"]
                else:
                    st.warning(f"Meme generation failed: {result.get('error_message')}")
            else:
                st.warning(f"Meme generation failed: {r.text}")
        except Exception as e:
            st.warning(f"Error generating meme: {e}")
    return "https://via.placeholder.com/512x512?text=Meme+Unavailable"

def generate_video(prompt_text):
    if VIDEO_API_KEY:
        try:
            response = requests.post(
                "https://api.synthesia.io/v1/videos",
                headers={"Authorization": f"Bearer {VIDEO_API_KEY}"},
                json={"script": prompt_text, "voice": "en-US", "resolution": "1080p"}
            )
            if response.status_code == 200:
                video_data = response.json()
                return video_data.get("video_url", "https://via.placeholder.com/512x512?text=Video+Unavailable")
            else:
                st.warning(f"Video generation API failed: {response.text}")
        except Exception as e:
            st.warning(f"Error generating video: {e}")
    return "https://via.placeholder.com/512x512?text=Video+Unavailable"

def summarize_and_rewrite(content, tone, platform):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes and rewrites content."},
                {"role": "user",
                 "content": f"Summarize the following content and rewrite it to match a {tone.lower()} tone suitable for {platform}: {content}"}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        summary = response.choices[0].message['content'].strip()
        return summary
    except Exception as e:
        st.warning(f"GPT-4 unavailable. Error: {e}")
        return content

def generate_fallback_content(query, tone, platform):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "You are a creative assistant that generates content based on user input."},
                {"role": "user", "content": f"Generate a {tone.lower()} {platform.lower()} post about {query}."}
            ],
            max_tokens=150,
            temperature=0.7,
        )
        fallback = response.choices[0].message['content'].strip()
        return fallback
    except Exception as e:
        st.warning(f"GPT-4 unavailable. Error: {e}")
        return f"Here's some content based on your interest in {query} with a {tone.lower()} tone, suitable for {platform}."

def suggest_hashtags(query, platform):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "You are an assistant that suggests relevant hashtags for social media posts."},
                {"role": "user", "content": f"Suggest 5 popular hashtags for a {platform} post about {query}."}
            ],
            max_tokens=50,
            temperature=0.5,
        )
        hashtags = response.choices[0].message['content'].strip()
        hashtags = ' '.join([tag if tag.startswith('#') else f"#{tag}" for tag in hashtags.split()])
        return hashtags
    except Exception as e:
        st.warning(f"GPT-4 unavailable. Error: {e}")
        return "#Trending #News #Updates"

def rate_limit_exceeded(username):
    last_reset = st.session_state.get(f"{username}_last_reset", datetime.now())
    if datetime.now() - last_reset > timedelta(days=1):
        st.session_state[f"{username}_count"] = 0
        st.session_state[f"{username}_last_reset"] = datetime.now()
    return st.session_state.get(f"{username}_count", 0) >= 10

def increment_rate_limit(username):
    if f"{username}_count" in st.session_state:
        st.session_state[f"{username}_count"] += 1
    else:
        st.session_state[f"{username}_count"] = 1

def authenticate():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = ''
        st.session_state.registered_users = {}

    if not st.session_state.authenticated:
        st.sidebar.title("Login / Register")
        auth_option = st.sidebar.selectbox("Select Option:", ["Login", "Register"])

        if auth_option == "Register":
            new_username = st.sidebar.text_input("New Username")
            new_password = st.sidebar.text_input("New Password", type="password")
            if st.sidebar.button("Register"):
                if new_username in st.session_state.registered_users:
                    st.sidebar.error("Username already exists. Please choose a different one.")
                elif not new_username or not new_password:
                    st.sidebar.error("Please enter both username and password.")
                else:
                    hashed_pw = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
                    st.session_state.registered_users[new_username] = hashed_pw
                    st.sidebar.success("Registered successfully! Please log in.")
        elif auth_option == "Login":
            username = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password", type="password")
            if st.sidebar.button("Login"):
                if username in st.session_state.registered_users:
                    hashed_pw = st.session_state.registered_users[username]
                    if bcrypt.checkpw(password.encode('utf-8'), hashed_pw):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.sidebar.success("Logged in successfully")
                    else:
                        st.sidebar.error("Invalid username or password")
                else:
                    st.sidebar.error("Invalid username or password")
    else:
        st.sidebar.title("Logged In")
        st.sidebar.write(f"**User:** {st.session_state.username}")
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = ''
            st.sidebar.success("Logged out successfully")

def main():
    st.title("RAG-based News-to-Content Agent")

    authenticate()

    if st.session_state.authenticated:
        username = st.session_state.username
        if rate_limit_exceeded(username):
            st.warning("You have reached your daily limit of content generations (10 per day). Please try again tomorrow.")
            st.stop()

        if 'history' not in st.session_state:
            st.session_state.history = []

        with st.sidebar:
            st.header("Popular Hashtags")
            platform_selected = st.selectbox("Select Platform for Hashtag Suggestions:", [
                "LinkedIn", "Instagram", "Reddit", "X (formerly Twitter)",
                "Facebook", "TikTok", "Pinterest", "Snapchat", "YouTube", "Medium"
            ])
            query_for_hashtags = st.text_input("Enter a topic to get popular hashtags:", "")
            if st.button("Get Hashtags"):
                if query_for_hashtags.strip():
                    hashtags = suggest_hashtags(query_for_hashtags, platform_selected)
                    st.write("### Suggested Hashtags:")
                    st.write(hashtags)
                else:
                    st.warning("Please enter a valid topic to get hashtags.")

        st.markdown("## Generate Content Based on News Articles")

        with st.form(key='content_generator_form'):
            query = st.text_input("Enter your topic or prompt:", "")
            tone = st.selectbox("Select Tone:", [
                "Formal", "Conversational", "Humorous", "Inspirational",
                "Sarcastic", "Optimistic", "Pessimistic", "Motivational",
                "Friendly", "Professional", "Witty", "Encouraging"
            ])
            format_type = st.selectbox("Select Content Format:", ["Text", "Image", "Meme", "Video"])
            platform = st.selectbox("Select Platform:", [
                "LinkedIn", "Instagram", "Reddit", "X (formerly Twitter)",
                "Facebook", "TikTok", "Pinterest", "Snapchat", "YouTube", "Medium"
            ])
            meme_template = None
            if format_type == "Meme":
                meme_templates = {
                    "Distracted Boyfriend": "112126428",
                    "Drake Hotline Bling": "181913649",
                    "Two Buttons": "87743020",
                    "Mocking Spongebob": "102156234",
                    "Change My Mind": "129242436"
                }
                meme_choice = st.selectbox("Select Meme Template:", list(meme_templates.keys()))
                meme_template = meme_templates[meme_choice]
            submit_button = st.form_submit_button(label="Generate Content")

        if submit_button:
            if not query.strip():
                st.warning("Please enter a valid prompt.")
                st.stop()

            with st.spinner("Fetching news articles..."):
                articles = fetch_news_autonomously(query, limit=5)
                if articles:
                    upsert_articles_to_chroma(articles)
                    st.success(f"Fetched {len(articles)} articles successfully.")
                else:
                    st.info("No articles found from the news sources. Generating fallback content based on your input.")
                    fallback_text = generate_fallback_content(query, tone, platform)
                    st.session_state.history.append({
                        "query": query,
                        "tone": tone,
                        "format_type": format_type,
                        "platform": platform,
                        "content": fallback_text,
                        "citations": []
                    })
                    increment_rate_limit(username)
                    st.markdown("### Generated Content:")
                    if format_type == "Text":
                        st.write(fallback_text)
                    elif format_type == "Image":
                        img_url = generate_image(f"Create a {tone.lower()} image based on: {fallback_text}")
                        st.image(img_url, use_container_width=True)
                        st.download_button("Download Image", requests.get(img_url).content, "image.png")
                    elif format_type == "Meme":
                        meme_caption = f"{tone} meme about {query}"
                        meme_url = generate_meme(meme_template, meme_caption)
                        st.image(meme_url, use_column_width=True)
                        st.download_button("Download Meme", requests.get(meme_url).content, "meme.png")
                    elif format_type == "Video":
                        video_url = generate_video(fallback_text)
                        st.video(video_url)
                        st.download_button("Download Video", requests.get(video_url).content, "video.mp4")
                    st.markdown("### Citations:")
                    st.write("No external sources were used to generate this content.")
                    st.stop()

            with st.spinner("Retrieving relevant articles..."):
                docs, meta = retrieve_relevant_articles(query, k=3)
                if not docs:
                    st.info("No relevant articles found in the database. Generating fallback content based on your input.")
                    fallback_text = generate_fallback_content(query, tone, platform)
                    st.session_state.history.append({
                        "query": query,
                        "tone": tone,
                        "format_type": format_type,
                        "platform": platform,
                        "content": fallback_text,
                        "citations": []
                    })
                    increment_rate_limit(username)
                    st.markdown("### Generated Content:")
                    if format_type == "Text":
                        st.write(fallback_text)
                    elif format_type == "Image":
                        img_url = generate_image(f"Create a {tone.lower()} image based on: {fallback_text}")
                        st.image(img_url, use_container_width=True)
                        st.download_button("Download Image", requests.get(img_url).content, "image.png")
                    elif format_type == "Meme":
                        meme_caption = f"{tone} meme about {query}"
                        meme_url = generate_meme(meme_template, meme_caption)
                        st.image(meme_url, use_column_width=True)
                        st.download_button("Download Meme", requests.get(meme_url).content, "meme.png")
                    elif format_type == "Video":
                        video_url = generate_video(fallback_text)
                        st.video(video_url)
                        st.download_button("Download Video", requests.get(video_url).content, "video.mp4")
                    st.markdown("### Citations:")
                    st.write("No external sources were used to generate this content.")
                    st.stop()
                else:
                    st.success(f"Retrieved {len(docs)} relevant articles.")

            combined_text = " ".join(docs)

            with st.spinner("Summarizing and rewriting content..."):
                final_text = summarize_and_rewrite(combined_text, tone, platform)
                st.session_state.history.append({
                    "query": query,
                    "tone": tone,
                    "format_type": format_type,
                    "platform": platform,
                    "content": final_text,
                    "citations": meta
                })
                increment_rate_limit(username)

            st.markdown("### Generated Content:")
            if format_type == "Text":
                st.write(final_text)
            elif format_type == "Image":
                img_url = generate_image(f"Create a {tone.lower()} image based on: {final_text}")
                st.image(img_url, use_container_width=True)
                st.download_button("Download Image", requests.get(img_url).content, "image.png")
            elif format_type == "Meme":
                meme_caption = f"{tone} meme about {query}"
                meme_url = generate_meme(meme_template, meme_caption)
                st.image(meme_url, use_column_width=True)
                st.download_button("Download Meme", requests.get(meme_url).content, "meme.png")
            elif format_type == "Video":
                video_url = generate_video(final_text)
                st.video(video_url)
                st.download_button("Download Video", requests.get(video_url).content, "video.mp4")

            if meta:
                st.markdown("### Citations:")
                for m in meta:
                    if m.get("url"):
                        st.markdown(f"- [Source: {m.get('title', 'Article')}]({m['url']})")
            else:
                st.markdown("### Citations:")
                st.write("No external sources were used to generate this content.")

            st.markdown("## Interactive Refinements")
            with st.expander("Refine Generated Content"):
                refinement = st.text_input(
                    "Enter your refinement prompt (e.g., 'Make it funnier' or 'Focus on AI ethics'):", "")
                refine_button = st.button("Refine Content")
                if refine_button and refinement.strip():
                    last_entry = st.session_state.history[-1]
                    original_content = last_entry["content"]
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system",
                                 "content": "You are a helpful assistant that refines content based on user instructions."},
                                {"role": "user", "content": f"{refinement}\n\nOriginal Content:\n{original_content}"}
                            ],
                            max_tokens=300,
                            temperature=0.7,
                        )
                        refined_content = response.choices[0].message['content'].strip()
                    except Exception as e:
                        st.warning(f"GPT-4 unavailable. Error: {e}")
                        refined_content = original_content

                    st.session_state.history.append({
                        "query": last_entry["query"],
                        "tone": last_entry["tone"],
                        "format_type": last_entry["format_type"],
                        "platform": last_entry["platform"],
                        "content": refined_content,
                        "citations": last_entry["citations"]
                    })

                    st.markdown("### Refined Content:")
                    if format_type == "Text":
                        st.write(refined_content)
                    elif format_type == "Image":
                        img_url = generate_image(f"Create a {tone.lower()} image based on: {refined_content}")
                        st.image(img_url, use_container_width=True)
                        st.download_button("Download Image", requests.get(img_url).content, "image.png")
                    elif format_type == "Meme":
                        meme_caption = f"{tone} meme about {query}"
                        meme_url = generate_meme(meme_template, meme_caption)
                        st.image(meme_url, use_column_width=True)
                        st.download_button("Download Meme", requests.get(meme_url).content, "meme.png")
                    elif format_type == "Video":
                        video_url = generate_video(refined_content)
                        st.video(video_url)
                        st.download_button("Download Video", requests.get(video_url).content, "video.mp4")
    else:
        st.info("Please log in to access the application.")

if __name__ == "__main__":
    main()
