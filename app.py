import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="PagePal", page_icon="📚")

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('knn_model.pkl', 'rb') as f:
    knn = pickle.load(f)

df = pd.read_csv('Books_Data.csv')

X = vectorizer.transform(df['Combined'])
X_scaled = scaler.transform(X)

def get_knn_recommendations(title, n):
    title_index = df[df['Book-Title'] == title].index[0]
    distances, indices = knn.kneighbors(X_scaled[title_index], n_neighbors=n + 1)

    similarities = 1 - distances[0][1:]

    recommended_books = df.iloc[indices[0][1:]]

    recommendation = pd.DataFrame({
        'Book-Title': recommended_books['Book-Title'].values,
        'Book-Author': recommended_books['Book-Author'].values,
        'Publication_Date': recommended_books['Year-Of-Publication'].values,
        'Image-URL': recommended_books['Image-URL-S'].values,
        'Similarity': similarities
    })

    Recommendation = recommendation.sort_values('Similarity', ascending=False)

    return Recommendation


def main():
    st.title("Book Recommendation System")

    st.warning("**Disclaimer:** Some images may not be available due to the age of the dataset.")

    if 'featured_books' not in st.session_state:
        st.session_state.featured_books = df.sample(3)

    featured_books = st.session_state.featured_books

    cols = st.columns(3)
    for i, row in enumerate(featured_books.iterrows()):
        with cols[i]:
            st.write(f"**Title:** {row[1]['Book-Title']}")
            st.write(f"**Author:** {row[1]['Book-Author']}")
            st.image(row[1]['Image-URL-S'], width=100)

    if st.button("Regenerate Featured Books"):
        st.session_state.featured_books = df.sample(3)
        st.rerun()

    featured_book_titles = featured_books['Book-Title'].tolist()

    selected_book = st.selectbox("Select a Book", featured_book_titles)

    num_recommendations = st.slider("Select Number of Recommendations", min_value=1, max_value=4, value=1)

    if st.button("Get Recommendations"):
        recommendations = get_knn_recommendations(selected_book, num_recommendations)

        st.subheader("Recommendations")
        rec_cols = st.columns(num_recommendations)

        for i in range(num_recommendations):
            with rec_cols[i]:
                st.write(f"**Title:** {recommendations.iloc[i]['Book-Title']}")
                st.write(f"**Author:** {recommendations.iloc[i]['Book-Author']}")
                st.image(recommendations.iloc[i]['Image-URL'], width=100)


if __name__ == "__main__":
    main()