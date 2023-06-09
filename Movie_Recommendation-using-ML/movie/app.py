
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import csv
import warnings

warnings.filterwarnings('ignore')

users = {}
coln = ["user_id", "movie_id", "rating", "timestamp"]
df = pd.read_csv("ml-100k/u.data", sep='\t', names=coln)

movie_title = pd.read_csv("ml-100k/u.item", sep='|', header=None, encoding='latin1')
movie_title = movie_title[[0, 1]]
movie_title.columns = ['movie_id', 'title']

df = pd.merge(df, movie_title, on="movie_id")

moviemat = df.pivot_table(index="user_id", columns="title", values="rating")
ratings = pd.DataFrame(df.groupby('title').mean()['rating'])
ratings['no_of_ratings'] = pd.DataFrame(df.groupby('title').count()['rating'])
corr_starwars = moviemat.corrwith(moviemat['Star Wars (1977)'])
corr_starwars = pd.DataFrame(corr_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars = corr_starwars.join(ratings['no_of_ratings'])
corr_starwars = corr_starwars[corr_starwars['no_of_ratings'] > 100].sort_values('Correlation', ascending=False)

app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
    return render_template('login.html', movie_titles=movie_title['title'])

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    movie_name = request.form['movie_name']
    predictions = predict_movies(movie_name)
    return render_template('recommendations.html', movie_name=movie_name, predictions=predictions)

def predict_movies(movie_name):
    movie_user_ratings = moviemat[movie_name]  
    similar_to_movie = moviemat.corrwith(movie_user_ratings)
    corr_movie = pd.DataFrame(similar_to_movie, columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(ratings['no_of_ratings'])
    predictions = corr_movie[corr_movie['no_of_ratings'] > 100].sort_values('Correlation', ascending=False)
    return predictions.head()
@app.route('/login_input', methods=['POST'])
def login_input():
    user_name = request.form['login uid']
    password = request.form['login pass']
    if verify_user(user_name,password):
        return render_template('index.html', movie_titles=movie_title['title'])
    else:
        return render_template('login.html')

    
@app.route('/signup_input', methods=['POST'])
def signup_input():
    username = request.form['signup username']
    email = request.form['signup email']
    password = request.form['signup pass']
    with open('dataset.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([username, email, password])
    return  render_template('login.html', movie_titles=movie_title['title'])

def verify_user(email, password):
    with open('dataset.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == email and row[2] == password:
                return True
        return False
    
@app.route('/home', methods=['POST'])
def index():
    return  render_template('index.html', movie_titles=movie_title['title'])
   
@app.route('/feedback', methods=['POST'])
def feedback():
    return  render_template('feedback.html', movie_titles=movie_title['title'])


if __name__ == '__main__':
    app.run(debug=True)

