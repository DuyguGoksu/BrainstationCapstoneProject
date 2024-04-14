## **Project Overview**

This project is about predicting song popularity using numeric data as well as text data (i.e., lyrics). Music platforms such as Spotify have playlists made for everyone in addition to personalized lists. These playlists mostly consist of top songs or new releases in some genre. To get into these playlists, a track must have a high popularity score(around 50 out of 100). Popularity score of a track mostly depends on recent stream count and save and skip rate.

A model that predicts track popularity is a very helpful tool for artists who are trying to get into top category playlists. There are around 11 million artists on Spotify. Each get paid about $0.003 - $0.005 per stream, which they also share with their music label company. The outcomes of this project will help artists and record companies that represent them increase their revenue. Also, listeners' choice will be understood better, and this will improve their user experience.

## **Project Flowchart**

In the first notebook, I explore the numeric measures in the main dataset (30000 Spotify songs - link provided in the data dictionary). I also merge two other datasets from Kaggle matching on song name and artist or track id, to identify the number of lyrics I can get from these datasets. 

In notebook 2, I build a baseline linear regression model, optimizing hyperparameters via a grid search with 5 fold cross-validations. Fetures in the baseline linear regression model in the notebook 2 come from the main dataset (30000 Spotify songs), excluding lyrics. 

Merging with other lyrics datasets from Kaggle and also using lyricsgenius package and Genius API, I add lyrics to the data in notebook 3. 

In notebook 4, I clean lyrics identifying non-lyric text. 

In notebook 5, I merge the lyrics with the rest of the data, and create a dataframe for collecting ratings from gpt 3.5 turbo model using chat-completion by making API calss to Open AI.

In notebook 6, I make the API calls to Open AI, using an OpenAI key in a chat-completion, and clean the retrieved gpt ratings.

In notebook 7, I merge dataframe with the gpt ratings with the rest of the data used in baseline modeling. I also create visuals to explore the distributions of the ratings and check for the correlations between independent variables.

In notebook 8, I build neural network models and compare them to linear regression models. I try different dimensions (features) and structures. My best model is a neural network that I pickle at the end of notebook 8.



## **Data**

‘spotify_songs.csv’ from [this](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs) source is the main dataset used in this project. To this dataset, lyrics for ~2000 songs are taken from two other datasets: ‘lyrics_10k.csv’ from [here](https://www.kaggle.com/datasets/evabot/spotify-lyrics-dataset) and `labeled_lyrics_cleaned.csv` from [here](https://www.kaggle.com/datasets/edenbd/150k-lyrics-labeled-with-spotify-valence). 

Overall, the following columns are present in the dataframes used in the notebooks. Not every dataframe has all of them. Categorical columns are changed to one-hot encoded columns for modeling.

**`track_id:`** Unique id of the song.

**`track_name:`** Song name.

**`track_artist:`** Song artist.

**`track_popularity:`** Score for song popularity, from 0 to 100, where higher is better.

**`track_album_id:`** Unique id of the album that the song is in.

**`track_album_name:`** Name of the album that the song is in.

**`track_album_release_date:`** Date on which the album of the song is released.

**`playlist_name:`** Name of the playlist that track was in.

**`playlist_id:`** Unique id for the playlist that the track was in.

**`playlist_genre:`** Genre of the playlist that the track was in.

**`playlist_subgenre:`** Subgenre of the playlist that the track was in.

**`danceability:`** [as described in original source] Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.

**`energy:`** [as described in original source] Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. - dropped due to a huge class imbalance

**`key:`** [as described in original source] The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation . E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.

**`loudness:`** [as described in original source] The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.

**`mode:`**  [as described in original source] Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.

**`speechiness:`**  [as described in original source] Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.

**`acousticness:`** A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.

**`instrumentalness:`** [as described in original source] Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.

**`liveness:`** [as described in original source] Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.

**`valence:`** [as described in original source] A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).

**`tempo:`** [as described in original source] The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.

**`duration_ms:`** Track durations in milliseconds.

**`track_lyrics:`** lyrics added by merging with two separate datasets and Genius API as described above.

**`compound:`** Compound sentiment score, calculated using the nltk.sentiment.vader package. (range: -1 to 1) - experimented with only in notebook 1

**`unique_cwc:`** Count of unique content words in each lyric, calculated using the spaCy package. - experimented with only in notebook 1

After collecting lyrics with API calls to Lyrics Genius, cleaning and having gpt 3.5 to rate the lyrics with API calls to Open AI chat-completion, I have added the following columns:

**`language:`** Abbreviation for the dominant language in the lyrics of a given track. 

**`Clear:`** Ratings from gpt from 1 to 10 on how clear song lyrics are.

**`Relatable:`** Ratings from gpt from 1 to 10 on how relatable song lyrics are.

**`Memorable:`** Ratings from gpt from 1 to 10 on how memorable song lyrics are.

**`Engaging:`** Ratings from gpt from 1 to 10 on how engaging song lyrics are.

**`Symbolic:`** Ratings from gpt from 1 to 10 on how symbolic song lyrics are.

**`Lyrical:`** Ratings from gpt from 1 to 10 on how lyric-like (rather than another form of text) song lyrics are.

**`Happy:`** Ratings from gpt from 1 to 10 on how happy the emotions conveyed in the song lyrics are.

**`Surprised:`** Ratings from gpt from 1 to 10 on how surprised the emotions conveyed in the song lyrics are.

**`Fearful:`** Ratings from gpt from 1 to 10 on how clear fearful the emotions conveyed in the song lyrics are.

**`Angry:`** Ratings from gpt from 1 to 10 on how clear angry the emotions conveyed in the song lyrics are.

**`Disgusted:`** Ratings from gpt from 1 to 10 on how disgusted the emotions conveyed in the song lyrics are.

**`Sad:`** Ratings from gpt from 1 to 10 on how sad the emotions conveyed in the song lyrics are.


