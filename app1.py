import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import os
import spacy
from sentence_transformers import SentenceTransformer


# ----------------------
# Load data and embeddings
# ----------------------
@st.cache_resource
def load_data():
    df, embeddings= joblib.load("bookdata_combined_embeddings.pkl")
    df = df.reset_index(drop=True)
    return df, embeddings

df, combined_embeddings = load_data()
description_embeddings = joblib.load("description_embeddings.pkl")
model = SentenceTransformer("saved_models/sentence_model")

# ----------------------
# Cosine similarity matrix
# ----------------------
@st.cache_resource
def compute_similarity_matrix(embeddings):
    return cosine_similarity(embeddings)

cosine_sim_matrix = compute_similarity_matrix(combined_embeddings)

# ----------------------
# Book index by name
# ----------------------
def get_book_index(title):
    title = title.lower()
    matches = df[df['book_name'].str.lower() == title]
    if matches.empty:
        return None
    return matches.index[0]

# ----------------------
# Recommender Functions
# ----------------------

def cluster_recommender(book_title, top_n=5):
    idx = get_book_index(book_title)
    if idx is None:
        return None, None
    cluster = df.loc[idx, 'cluster_label']
    cluster_books = df[(df['cluster_label'] == cluster) & (df.index != idx)]
    return df.loc[idx], cluster_books.sample(min(top_n, len(cluster_books)))[['book_name', 'author', 'genre', 'rating']]

def genre_cluster_recommender(df, selected_genre, top_n=10):
    # Step 1: Filter books by the selected genre
    genre_books = df[df['genre'].str.lower() == selected_genre.lower()]

    if genre_books.empty:
        return None, None

    # Step 2: Find the most common cluster for this genre
    dominant_cluster = genre_books['cluster_label'].value_counts().idxmax()

    # Step 3: Get all books from that cluster (optionally within same genre)
    cluster_books = df[df['cluster_label'] == dominant_cluster]

    # Step 4: Optional - further filter by same genre (can skip this if you want variety)
    cluster_books = cluster_books[cluster_books['genre'].str.lower() == selected_genre.lower()]

    # Step 5: Clean and sort
    cluster_books['number_reviews'] = pd.to_numeric(cluster_books['number_reviews'], errors='coerce')
    cluster_books = cluster_books.dropna(subset=['rating', 'number_reviews'])
    cluster_books = cluster_books.sort_values(by=['rating', 'number_reviews'], ascending=False)

    return dominant_cluster, cluster_books.head(top_n)[['book_name', 'author', 'rating', 'number_reviews']]

def author_cluster_recommender(df, selected_author, top_n=10):
    # Step 1: Filter rows by the given author
    author_books = df[df['author'].str.lower() == selected_author.lower()]
    
    if author_books.empty:
        return None, None

    # Step 2: Get the most frequent cluster for this author
    dominant_cluster = author_books['cluster_label'].value_counts().idxmax()

    # Step 3: Get all books from that cluster
    cluster_books = df[df['cluster_label'] == dominant_cluster]

    # Optional: Filter to different authors for variety
    cluster_books = cluster_books[cluster_books['author'].str.lower() != selected_author.lower()]

    # Step 4: Sort by rating and number of reviews
    cluster_books['number_reviews'] = pd.to_numeric(cluster_books['number_reviews'], errors='coerce')
    cluster_books = cluster_books.dropna(subset=['rating', 'number_reviews'])
    cluster_books = cluster_books.sort_values(by=['rating', 'number_reviews'], ascending=False)

    return dominant_cluster, cluster_books.head(top_n)[['book_name', 'author', 'genre', 'rating', 'number_reviews']]

def hybrid_recommender(book_title, top_n=5):
    content_idx = get_book_index(book_title)
    if content_idx is None:
        return None, None
    # Content-based top 3
    sim_scores = list(enumerate(cosine_sim_matrix[content_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [score for score in sim_scores if score[0] != content_idx]
    content_top = [i[0] for i in sim_scores[:3]]

    # Cluster-based top 2
    cluster = df.loc[content_idx, 'cluster_label']
    cluster_books = df[(df['cluster_label'] == cluster) & (df.index != content_idx)]
    cluster_top = cluster_books.sample(min(2, len(cluster_books))).index.tolist()

    hybrid_indices = list(dict.fromkeys(content_top + cluster_top))  # Remove duplicates
    return df.loc[content_idx], df.iloc[hybrid_indices][['book_name', 'author', 'genre', 'rating']]

# load english language model and create nlp object from it
nlp = spacy.load("en_core_web_sm") 

#def load_spacy_model():
#    try:
#        return spacy.load("en_core_web_sm")
#    except OSError:
#       from spacy.cli import download
#       download("en_core_web_sm")
#       return spacy.load("en_core_web_sm")
#nlp = load_spacy_model()


#use this utility function to get the preprocessed text data
def preprocess(text):
    if pd.isna(text):  # Handles NaN or None safely
        return ""
    # remove stop words and lemmatize the text
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    
    return " ".join(filtered_tokens)

# ----------------------
# Streamlit Sidebar Navigation
# ----------------------
st.sidebar.title("📚 Book Recommender")
page = st.sidebar.radio("Choose a page:", ["Content-Based Filter", "Cluster-Based Filter", "Hybrid Filter", "Visualizations"])


# ----------------------
# Page 1: Content-Based
# ----------------------
if page == "Content-Based Filter":
    st.title("📘 Content-Based Recommender")

    st.subheader("Choose Recommendation Basis")
    method = st.selectbox("Recommend based on:", ["Book Title", "Book Description","Genre","Author"])

    if method == "Book Title":
        book_title = st.text_input("Enter a book Title:")

        if st.button("Get Recommendations"):
            if book_title.strip() == "":
                st.warning("Please enter a description.")
            else:
                idx = get_book_index(book_title)
                if idx is None:
                    st.error("Book not found in the database")
                    # Preprocess and encode the title
                    user_input_clean = preprocess(book_title)
                    title_embedding = model.encode([user_input_clean])
                    
                    # Use combined embeddings:
                    # Get the corresponding description as blank
                    desc_input = " "  # or use st.text_area for description
                    desc_embedding = model.encode(desc_input)
                    # Combine title + desc embeddings like you did earlier
                    #input_combined = np.hstack([title_embedding, desc_embedding]).reshape(1, -1)
                    input_combined = np.hstack([
                        title_embedding.reshape(1, -1),
                        desc_embedding.reshape(1, -1)
                    ])
                    # Calculate similarity with all combined embeddings
                    sim_scores = cosine_similarity(input_combined, combined_embeddings)[0]
                    # Get top results
                    top_indices = np.argsort(sim_scores)[::-1][:5]
                    recs = df.iloc[top_indices][['book_name', 'author', 'genre', 'rating']]
                    # Show results
                    st.subheader("🔍 Recommended by Title Similarity")
                    st.dataframe(recs)

                    # Optionally: show more from top genre
                    top_genre = df.iloc[top_indices[0]]['genre']
                    same_genre = df[(df['genre'] == top_genre) & (~df.index.isin(top_indices))]
                    if not same_genre.empty:
                        st.subheader(f"🎯 More from the '{top_genre}' genre")
                        st.dataframe(same_genre.sample(min(5, len(same_genre)))[['book_name', 'author', 'rating']])

                else:
                    book_info = df.loc[idx]
                    st.subheader("📖 Selected Book")
                    st.write(book_info[['book_name', 'author', 'genre', 'rating']])
                    # Use full combined embeddings (book name + desc)
                    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                    sim_scores = [score for score in sim_scores if score[0] != idx]
                    top_indices = [i[0] for i in sim_scores[:5]]
                    recs = df.iloc[top_indices][['book_name', 'author', 'genre', 'rating']]
                    st.subheader("🔍 Recommended by Title Similarity")
                    st.dataframe(recs)
                    # Below content_recommender results:
                    st.subheader("🎯 More from the same Genre")
                    same_genre = df[(df['genre'] == book_info['genre']) & (df.index != book_info.name)]
                    st.dataframe(same_genre.sample(min(5, len(same_genre)))[['book_name', 'author', 'rating']])

    elif method == "Book Description":
        user_input = st.text_area("Enter a book description:")
        # Compute cosine similarity on description embeddings
        if st.button("Find Similar Books"):
            if user_input.strip() == "":
                st.warning("Please enter a description.")
            else:
                # Step 1: Preprocess and encode the input description
                preprocessed_input = preprocess(user_input)  # Your custom function
                input_embedding = model.encode([preprocessed_input])  # Shape: (1, embedding_dim)
                # Step 2: Compute cosine similarity
                desc_sims = cosine_similarity(input_embedding, description_embeddings)[0]            
                # Step 3: Get top N similar books
                sim_scores = list(enumerate(desc_sims))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                top_indices = [i[0] for i in sim_scores[:5]]              
                # Step 4: Display recommended books
                recs = df.iloc[top_indices][['book_name', 'author', 'genre', 'rating']]
                st.subheader("📄 Recommended Books Based on Your Description")
                st.dataframe(recs)

    elif method == "Genre":
        selected_genre = st.selectbox("Choose a genre", df['genre'].unique())
        if st.button("Show Top Books"):
            # Filter books in the given genre
            genre_books = df[df['genre'].str.lower() == selected_genre.lower()]
            # Sort by rating (desc), then number_reviews (desc)
            sorted_books = genre_books.sort_values(by=['rating', 'number_reviews'], ascending=False)
            top_books = sorted_books.head(10)[['book_name', 'author', 'rating', 'number_reviews']]
            st.subheader(f"📚 Top 10 Books in '{selected_genre}' Genre")
            st.dataframe(top_books)

    elif method == "Author":
        selected_author = st.selectbox("Choose an author", df['author'].unique())
        if st.button("Show Top Books"):
            # Filter books for author
            author_books = df[df['author'].str.lower() == selected_author.lower()]
            # Sort by rating (desc), then number_reviews (desc)
            sorted_books = author_books.sort_values(by=['rating', 'number_reviews'], ascending=False)
            top_books = sorted_books.head(10)[['book_name', 'author', 'rating', 'number_reviews']]
            st.subheader(f"📚 Top 10 Books '{selected_author}' Author")
            st.dataframe(top_books)

# ----------------------
# Page 2: Cluster-Based
# ----------------------
elif page == "Cluster-Based Filter":
    st.title("📙 Cluster-Based Recommender")
    st.subheader("Choose Recommendation Basis")
    method = st.selectbox("Recommend based on:", ["Book Title","Genre","Author"])

    if method == "Book Title":
        book_title = st.text_input("Enter a book Title:")
        if st.button("Recommend from Cluster"):
            book_info, recs = cluster_recommender(book_title)
            if book_info is None:
                st.error("Book not found.")
            else:
                st.subheader("Selected Book")
                st.write(book_info[['book_name', 'author', 'genre', 'rating']])
                st.subheader("Books from Same Cluster")
                st.dataframe(recs)
        
    elif method == "Genre":
        selected_genre = st.selectbox("Choose a genre", df['genre'].unique())
        if st.button("Show Recommendations by Genre Cluster"):
            cluster_id, top_books = genre_cluster_recommender(df, selected_genre)
            if top_books is not None:
                st.subheader(f"📚 Top Books in '{selected_genre}' Genre (Cluster {cluster_id})")
                st.dataframe(top_books)
            else:
                st.warning("No books found for this genre.")
    
    elif method == "Author":
        selected_author = st.selectbox("Choose an Author", sorted(df['author'].unique()))

        if st.button("Recommend Books Based on Author's Cluster"):
            cluster_id, top_author_recs = author_cluster_recommender(df, selected_author)

            if top_author_recs is not None:
                st.subheader(f"✍️ Top Books from other Authors belonging to the same Cluster")
                st.dataframe(top_author_recs)
            else:
                st.warning("No recommendations found for the selected author.")


# ----------------------
# Page 3: Hybrid
# ----------------------
elif page == "Hybrid Filter":
    st.title("📗 Hybrid Recommender")
    book_title = st.text_input("Enter a book title:")
    if st.button("Get Hybrid Recommendations"):
        book_info, recs = hybrid_recommender(book_title)
        if book_info is None:
            st.error("Book not found.")
        else:
            st.subheader("Selected Book")
            st.write(book_info[['book_name', 'author', 'genre', 'rating']])
            st.subheader("Hybrid Recommendations")
            st.dataframe(recs)

# ----------------------
# Page 4: Visualizations
# ----------------------
elif page == "Visualizations":
    st.title("📊 Book Data Visualizations")

    st.subheader("Average Rating by Genre")
    genre_rating = df.groupby("genre")["rating"].mean().sort_values(ascending=False)
    st.bar_chart(genre_rating)

    st.subheader("Top Authors by Average Rating")
    top_authors = df.groupby("author")["rating"].mean().sort_values(ascending=False).head(10)
    st.bar_chart(top_authors)

    st.subheader("Cluster Size Distribution")
    cluster_counts = df['cluster_label'].value_counts().sort_index()
    st.bar_chart(cluster_counts)

    st.subheader("Rating Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["rating"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("📝 Top 10 Authors by Total Reviews")
    # Group and sort authors by total number of reviews
    top_authors = df.groupby('author')['number_reviews'].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    top_authors.plot(kind='bar', ax=ax)
    ax.set_title('Top 10 Authors by Total Reviews')
    ax.set_ylabel('Number of Reviews')
    ax.set_xlabel('Author')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
