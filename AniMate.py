from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import h5py
import requests

app = Flask(__name__)

def extract_weights(file_path, layer_name):
    # open .h5 file which has weights of model. Then normalize the weights using L2 norm.
    # We want weights for user as well as item embeddings.
    with h5py.File(file_path, 'r') as h5_file:
        if layer_name in h5_file:
            weight_layer = h5_file[layer_name]
            if isinstance(weight_layer, h5py.Dataset):  
                weights = weight_layer[()]
                weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))
                return [weights]

    raise KeyError(f"Unable to find weights for layer '{layer_name}' in the HDF5 file.")

file_path = 'AniMate_Model\\model.h5'
anime_weights = extract_weights(file_path, 'anime_embedding/anime_embedding/embeddings:0')
user_weights = extract_weights(file_path, 'user_embedding/user_embedding/embeddings:0')

anime_weights=anime_weights[0]
user_weights=user_weights[0]

# Loads a pickled object (anime_encoder) that maps anime names or IDs into indexes which when given to model return numerical vector embeddings .
with open('AniMate_Model/anime.pkl', 'rb') as file:
    anime_encoder = pickle.load(file)

# Loads a pickled object (user_encoder) that maps user IDs into indexes which when given to model return numerical vector embeddings .
with open('AniMate_Model/user.pkl', 'rb') as file:
    user_encoder = pickle.load(file)
with open('AniMate_Model/anime_data.pkl', 'rb') as file:
    df_anime = pickle.load(file)

# cleaning the data    
df_anime = df_anime.replace("UNKNOWN", "")
df=pd.read_csv('AniMate_Model/rating_data.csv', low_memory=True)
# at home render index.html
@app.route('/')
def home():
    return render_template('index.html')

# User based function
def find_similar_users(item_input, n=10, return_dist=False, neg=False):
    try:
        index = item_input
        # transform the user id into index
        encoded_index = user_encoder.transform([index])[0]
        weights = user_weights
        # calculate the distance between the user embedding and all other user embeddings
        dists = np.dot(weights, weights[encoded_index])
        # sort the distances in ascending order
        sorted_dists = np.argsort(dists)
        n = n + 1
        if neg:
            closest = sorted_dists[:n]
        else:
            closest = sorted_dists[-n:]
        SimilarityArr = []
        for close in closest:
            similarity = dists[close]
            if isinstance(item_input, int):
                decoded_id = user_encoder.inverse_transform([close])[0]
                # inverse trasnform so we have user_ids instead of indexes
                SimilarityArr.append({"similar_users": decoded_id, "similarity": similarity})
        Frame = pd.DataFrame(SimilarityArr).sort_values(by="similarity", ascending=False)
        return Frame
    except:
        print('\033[1m{}\033[0m, Not Found in User list'.format(item_input))

# User based function
def get_user_preferences(user_id):
    # uses rating data inside df to get all anime's user has rated
    animes_watched_by_user = df[df['user_id'] == user_id]
    if animes_watched_by_user.empty:
        print("User #{} has not watched any animes.".format(user_id))
        return pd.DataFrame()
    user_rating_percentile = np.percentile(animes_watched_by_user.rating, 75)
    # This calculate 75th percentile of all ratings of user. It helps pick top rated animes by this user. Next line filters ratings based on percentile
    animes_watched_by_user = animes_watched_by_user[animes_watched_by_user.rating >= user_rating_percentile]
    # top animes sorted descending by rating
    top_animes_user = (
        animes_watched_by_user.sort_values(by="rating", ascending=False)
        .anime_id.values
    )

    anime_df_rows = df_anime[df_anime["anime_id"].isin(top_animes_user)]
    anime_df_rows = anime_df_rows[["Name", "Genres"]]
    # returns those top picked animes by user as user preferences
    return anime_df_rows

# User based function
def get_recommended_animes(similar_users, user_pref, n=10):
    # function is to recommend animes based on what similar users have liked but only those which user has not watched yet
    # user pref comes from above function and reresents all top animes user has watched
    # similar user function above returns all similar users to user using model weights
    recommended_animes = []
    anime_list = []
    # get top rated animes watched by users similar to our user. Exclude those animes already watched by user.
    for user_id in similar_users.similar_users.values:
        pref_list = get_user_preferences(int(user_id))
        if not pref_list.empty: 
            pref_list = pref_list[~pref_list["Name"].isin(user_pref["Name"].values)]
            anime_list.append(pref_list.Name.values)
    if len(anime_list) == 0:
        print("No anime recommendations")
        return pd.DataFrame()
    anime_list = pd.DataFrame(anime_list)
    # value counts tells how many similar users liked each anime and sorts descending based on count.
    sorted_list = pd.DataFrame(pd.Series(anime_list.values.ravel()).value_counts()).head(n)
    anime_count = df['anime_id'].value_counts()
    for i, anime_name in enumerate(sorted_list.index):
        if isinstance(anime_name, str):
            try:
                anime_image_url = df_anime[df_anime['Name'] == anime_name]['Image URL'].values[0]
                anime_id = df_anime[df_anime.Name == anime_name].anime_id.values[0]
                genre = df_anime[df_anime.Name == anime_name].Genres.values[0]                
                Synopsis = df_anime[df_anime.Name == anime_name].Synopsis.values[0]
                n_user_pref = anime_count.get(anime_id, 0)  
                english_name = df_anime[df_anime.Name == anime_name]['English name'].values[0]
                other_name = df_anime[df_anime.Name == anime_name]['Other name'].values[0]
                score = df_anime[df_anime.Name == anime_name].Score.values[0]
                Type = df_anime[df_anime.Name == anime_name].Type.values[0]
                status = df_anime[df_anime.Name == anime_name].Status.values[0]
                aired = df_anime[df_anime.Name == anime_name].Aired.values[0]
                episodes = df_anime[df_anime.Name == anime_name].Episodes.values[0]
                premiered = df_anime[df_anime.Name == anime_name].Premiered.values[0]
                studios = df_anime[df_anime.Name == anime_name].Studios.values[0]
                source = df_anime[df_anime.Name == anime_name].Source.values[0]
                rating = df_anime[df_anime.Name == anime_name].Rating.values[0]
                rank = df_anime[df_anime.Name == anime_name].Rank.values[0]
                favorites = df_anime[df_anime.Name == anime_name].Favorites.values[0]
                duration = df_anime[df_anime.Name == anime_name].Duration.values[0]
                if status == "Not yet aired" and aired == "Not available":
                    aired = "TBA"
                else:
                    aired = "" if aired == "Not available" else aired.replace(" to ", "-")
                if episodes != "":
                    episodes = int(float(episodes))
                    if status == "Currently Airing":
                        episodes = str(episodes)+"+ EPS"
                    else:
                        episodes = str(episodes)+" EPS"
                else:
                    if status == "Currently Airing":
                        aired_year = df_anime[df_anime.Name == anime_name].Aired.values[0]
                        if ',' in aired_year:
                            aired_year = aired_year.split(',')[1].strip()
                            aired_year = aired_year.split(' to ')[0].strip()
                        else:
                            aired_year = aired_year.split(' to ')[0].strip()
                        if aired_year != "Not available" and int(aired_year) <= 2020:
                            episodes = "∞"
                        else:
                            episodes = ""
                    else:
                        episodes = ""
                rating = rating if rating == "" else rating.split(' - ')[0]
                rank = rank if rank == "" else "#"+str(int(float(rank)))
                episode_duration = ""
                if episodes != "":
                    time = ""
                    if 'hr' in duration:
                        hours, minutes = 0, 0
                        time_parts = duration.split()
                        for i in range(len(time_parts)):
                            if time_parts[i] == "hr":
                                hours = int(time_parts[i-1])
                            elif time_parts[i] == "min":
                                minutes = int(time_parts[i-1])
                        time = str(hours * 60 + minutes) + " min"
                    else:
                        time= duration.replace(" per ep","")
                    episode_duration = "("+ episodes + "  x " + time +")"
                else:
                    episode_duration = "("+ duration +")"


                recommended_animes.append({"anime_image_url": anime_image_url, "n": n_user_pref,"anime_name": anime_name, "Genres": genre,
                                           "Synopsis": Synopsis,"English Name": english_name,"Native name": other_name,"Score": score,
                                           "Type": Type, "Aired": aired, "Premiered": premiered, "Episodes": episodes, "Status": status,
                                           "Studios": studios,"Source": source, "Rating": rating, "Rank": rank, "Favorites": favorites,
                                           "Duration": duration, "Episode Duration": episode_duration,"anime_id":anime_id})
            except:
                pass
    #recommended animes has info about all animes we calculated in sorted_list.

    return pd.DataFrame(recommended_animes)

# Item based function
# the function finds and returns a list of anime titles similar to a given anime
def find_similar_animes(name, n=10, return_dist=False, neg=False):
    try:
        # Get the row of the anime that matches the given name
        anime_row = df_anime[df_anime['Name'] == name].iloc[0]
        index = anime_row['anime_id']  # Extract the anime_id

        # Convert anime_id to encoded index (used in embedding matrix)
        encoded_index = anime_encoder.transform([index])[0]

        weights = anime_weights  # Precomputed anime embeddings from the model

        dists = np.dot(weights, weights[encoded_index])

         # Get sorted indices based on similarity
        sorted_dists = np.argsort(dists)
        n = n + 1            
        if neg:
            closest = sorted_dists[:n]
        else:
            closest = sorted_dists[-n:]
        print('Animes closest to {}'.format(name))
        if return_dist:
            return dists, closest
        SimilarityArr = []
        for close in closest:
            # get the anime_id from the encoded index
            decoded_id = anime_encoder.inverse_transform([close])[0]
            # get the anime row from decoded id
            anime_frame = df_anime[df_anime['anime_id'] == decoded_id]
            anime_id=anime_frame['anime_id'].values[0]
            anime_image_url = anime_frame['Image URL'].values[0]
            anime_name = anime_frame['Name'].values[0]
            genre = anime_frame['Genres'].values[0]
            Synopsis = anime_frame['Synopsis'].values[0]
            similarity = dists[close]
            similarity = "{:.2f}%".format(similarity * 100)
            
            english_name = anime_frame['English name'].values[0]
            other_name = anime_frame['Other name'].values[0]
            score = anime_frame['Score'].values[0]
            Type = anime_frame['Type'].values[0]
            other_name = anime_frame['Other name'].values[0]
            status = anime_frame['Status'].values[0]
            aired = anime_frame['Aired'].values[0]
            episodes = anime_frame['Episodes'].values[0]
            premiered = anime_frame['Premiered'].values[0]
            studios = anime_frame['Studios'].values[0]
            source = anime_frame['Source'].values[0]
            rating = anime_frame['Rating'].values[0]
            rank = anime_frame['Rank'].values[0]
            favorites = anime_frame['Favorites'].values[0]
            duration = anime_frame['Duration'].values[0]
            if status == "Not yet aired" and aired == "Not available":
                aired = "TBA"
            else:
                aired = "" if aired == "Not available" else aired.replace(" to ", "-")
            if episodes != "":
                episodes = int(float(episodes))
                if status == "Currently Airing":
                    episodes = str(episodes)+"+ EPS"
                else:
                    episodes = str(episodes)+" EPS"
            else:
                if status == "Currently Airing":
                    aired_year = anime_frame['Aired'].values[0]
                    if ',' in aired_year:
                        aired_year = aired_year.split(',')[1].strip()
                        aired_year = aired_year.split(' to ')[0].strip()
                    else:
                        aired_year = aired_year.split(' to ')[0].strip()
                    if aired_year != "Not available" and int(aired_year) <= 2020:
                        episodes = "∞"
                    else:
                        episodes = ""
                else:
                    episodes = ""
            rating = rating if rating == "" else rating.split(' - ')[0]
            rank = rank if rank == "" else "#"+str(int(float(rank)))
            episode_duration = ""
            if episodes != "":
                time = ""
                if 'hr' in duration:
                    hours, minutes = 0, 0
                    time_parts = duration.split()
                    for i in range(len(time_parts)):
                        if time_parts[i] == "hr":
                            hours = int(time_parts[i-1])
                        elif time_parts[i] == "min":
                            minutes = int(time_parts[i-1])
                    time = str(hours * 60 + minutes) + " min"
                else:
                    time= duration.replace(" per ep","")
                episode_duration = "("+ episodes + "  x " + time +")"
            else:
                episode_duration = "("+ duration +")"
            
            
            SimilarityArr.append({"anime_image_url": anime_image_url,"Name": anime_name, "Similarity": similarity, "Genres": genre,
                                  "Synopsis":Synopsis,"English Name": english_name,"Native name": other_name,"Score": score,"Type": Type,
                                  "Aired": aired, "Premiered": premiered, "Episodes": episodes, "Status": status, "Studios": studios,
                                  "Source": source, "Rating": rating, "Rank": rank, "Favorites": favorites,"Duration": duration,
                                  "Episode Duration": episode_duration,"anime_id":anime_id})
        Frame = pd.DataFrame(SimilarityArr).sort_values(by="Similarity", ascending=False)
        return Frame[Frame.Name != name]
    except:
        print('{} not found in Anime list'.format(name))

# main route to recommend animes        
@app.route('/recommend', methods=['POST'])
def recommend():
    recommendation_type = request.form['recommendation_type']
    num_recommendations = int(request.form['num_recommendations'])
    # checks recommendation is user based or item based
    if recommendation_type == "user_based":
        user_id = request.form['user_id']
        
        if not user_id:
            return render_template('index.html', error_message="Please enter a User ID.", recommendation_type=recommendation_type)
        try:
            user_id = int(user_id)
        except ValueError:
            return render_template('index.html', error_message="Please enter a valid User ID (must be an integer).", recommendation_type=recommendation_type)
        similar_user_ids = find_similar_users(user_id, n=15, neg=False)
        if similar_user_ids is None or similar_user_ids.empty:
            url = f'https://api.jikan.moe/v4/users/userbyid/{user_id}' 
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if 'data' not in data:
                    message1 = "Available"
                else:
                    message1 = "No anime recommendations"
            else:
                message1 = "User with user_id " + str(user_id) + " does not exist in the database."
            return render_template('recommendations.html', message=message1, animes=None, recommendation_type=recommendation_type)

        # filters similar users based on similarity
        similar_user_ids = similar_user_ids[similar_user_ids.similarity > 0.4]
        similar_user_ids = similar_user_ids[similar_user_ids.similar_users != user_id]
        # according to users similar to user_id, we get user preferences for all similar users
        user_pref = get_user_preferences(user_id)
        # using animes we get from similar users, we get recommended animes
        recommended_animes = get_recommended_animes(similar_user_ids, user_pref, n=num_recommendations)
        return render_template('recommendations.html', animes=recommended_animes, recommendation_type=recommendation_type)

    elif recommendation_type == "item_based":
        anime_name = request.form['anime_name']
        
        if not anime_name:
            return render_template('index.html', error_message="Please enter Anime name.", recommendation_type=recommendation_type)
        
        recommended_animes = find_similar_animes(anime_name, n=num_recommendations, return_dist=False, neg=False)
        if recommended_animes is None or recommended_animes.empty:
            message2 = "Anime " + str(anime_name) + " does not exist"
            return render_template('recommendations.html', message=message2, animes=None, recommendation_type=recommendation_type)
        
        return render_template('recommendations.html', animes=recommended_animes, recommendation_type=recommendation_type)

    else:
        return render_template('index.html', error_message="Please select a recommendation type.")
    
# autocomplete function for index.html
@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    search_term = request.args.get('term')
    if search_term:
        filtered_animes = df_anime[df_anime['Name'].str.contains(search_term, case=False)]
        anime_names = filtered_animes['Name'].tolist()
    return jsonify(anime_names)

if __name__ == '__main__':
    app.run(debug=True)