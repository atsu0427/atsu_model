import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_model():

    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return vectorizer

def recommend(vectorizer, text):

    product_data=pd.read_csv('./static/csv/clothes_data.csv')
    grouped_product_data = product_data.groupby('商品名')['レビュー本文'].apply(' '.join).reset_index()  
    product_descriptions = product_data['レビュー本文'].values
    product_names = product_data['商品名'].values

    tfidf_matrix = vectorizer.transform(product_descriptions)

    text_tfidf = vectorizer.transform([text])
    similarities = cosine_similarity(tfidf_matrix, text_tfidf)

    related_products_indices = similarities.flatten().argsort()[-3:][::-1]
    recommendations = [product_names[idx] for idx in related_products_indices]
    print(recommendations)
    return recommendations   