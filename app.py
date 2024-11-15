import streamlit as st
import pandas as pd
import pickle

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

def generate_books():
    print("hello")


def get_knn_recommendations(title, n):
    title_index = df[df['Book-Title'] == title].index[0]
    distances, indices = knn.kneighbors(X_scaled[title_index], n_neighbors=n + 1)

    similarities = 1 - distances[0][1:]

    recommended_books = df.iloc[indices[0][1:]]

    recommendation = pd.DataFrame({
        'Book-Title': recommended_books['Book-Title'].values,
        'Book-Author': recommended_books['Book-Author'].values,
        'Publication_Date': recommended_books['Year-Of-Publication'].values,
        'Similarity': similarities
    })

    Recommendation = recommendation.sort_values('Similarity', ascending=False)

    return Recommendation


def recommendation_counts(prompt):
    while True:
        try:
            n = int(input(prompt))
            if n < 0:
                print("Invalid input. Please enter a positive integer.")
            else:
                return n
        except ValueError:
            print("Invalid input. Please enter a valid integer.")


def main():
    generate_books()

if __name__ == '__main__':
    main()