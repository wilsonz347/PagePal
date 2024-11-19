import streamlit as st
import pandas as pd
import pickle
import random

st.set_page_config(page_title="PagePal")

def load_dataset():
    dataframe = pd.read_csv('Books_Data.csv')
    return dataframe

def load_model():
    with open('knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    return knn_model

def load_scaler():
    with open('scaler.pkl', 'rb') as f:
        scale = pickle.load(f)
    return scale

def load_vectorizer():
    with open('vectorizer.pkl', 'rb') as f:
        vector = pickle.load(f)
    return vector

df = load_dataset()
knn = load_model()
scaler = load_scaler()
vectorizer = load_vectorizer()

X = vectorizer.transform(df['Combined'])
X_scaled = scaler.transform(X)

def generate_books(num_books):
    selected_books = df.sample(n=num_books)

    books = []
    for _, book in selected_books.iterrows():
        books.append({
            'title': book['Book-Title'],
            'author': book['Book-Author'],
            'year': book['Year-Of-Publication'],
            'image_url': book['Image-URL-S']
        })

    return books

def get_knn_recommendations(title, n):
    title_index = df[df['Book-Title'] == title].index[0]
    distances, indices = knn.kneighbors(X_scaled[title_index], n_neighbors=n + 1)

    similarities = 1 - distances[0][1:]

    recommended_books = df.iloc[indices[0][1:]]

    recommendation = pd.DataFrame({
        'Book-Title': recommended_books['Book-Title'].values,
        'Book-Author': recommended_books['Book-Author'].values,
        'Publication_Date': recommended_books['Year-Of-Publication'].values,
        'image_url': recommended_books['Image-URL-S'],
        'Similarity': similarities
    })

    Recommendation = recommendation.sort_values('Similarity', ascending=False)

    return Recommendation


def get_recommendations_from_featured(selected_title, featured_books, num_recommendations):
    other_books = [book for book in featured_books if book['title'] != selected_title]

    recommendations = random.sample(other_books, min(num_recommendations, len(other_books)))

    return recommendations


def main():
    st.title("Book Recommendation System")

    if 'featured_books' not in st.session_state:
        st.session_state.featured_books = generate_books(num_books=5)

    st.header("Featured Books")
    cols = st.columns(5)
    for i, book in enumerate(st.session_state.featured_books):
        with cols[i]:
            st.image(book['image_url'], width=100)
            st.write(f"**{book['title']}**")
            st.write(f"by {book['author']}")
            st.write(f"Year: {book['year']}")

    st.header("Get Book Recommendations")

    featured_titles = [book['title'] for book in st.session_state.featured_books]

    book_title = st.selectbox("Select a book:", featured_titles)

    num_recommendations = st.slider("Number of recommendations:", 1, 4, 2)

    with st.container():
        if st.button("Get Recommendations"):
            recommendations = get_recommendations_from_featured(book_title, st.session_state.featured_books,
                                                                num_recommendations)

            st.subheader(f"Recommendations for '{book_title}':")
            for rec in recommendations:
                st.write(f"**{rec['title']}** by {rec['author']}")
                st.write(f"Year: {rec['year']}")
                st.image(rec['image_url'], width=100)
                st.write("---")

        if st.button("Generate New Books"):
            st.session_state.featured_books = generate_books(num_books=5)


if __name__ == '__main__':
    main()