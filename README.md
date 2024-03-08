## **Project Overview**

This project is about modeling song similarity using lyrics. Music platforms such as Spotify have playlists made for everyone in addition to personalized lists. These playlists mostly consist of top songs or new releases in some genre. In addition to these, there are playlists for doing a certain activity such as exercise, study, or shower (e.g., the famous ‘songs to sing in the shower’). There are also playlists for songs from a certain era. 

Modeling songs based on semantic and thematic similarity between lyrics will enable creating new and interesting playlists. These can be named, very simply, ‘Songs about xyz’ or the naming can be more creative. The dataset I am using has other features that can be incorporated in narrowing down the playlist category and the songs in it, but the main characteristic of each playlist will come from the semantic and thematic similarity between the lyrics of songs in the playlist. I plan to use topic modeling and word embedding models from the gensim library in Python. 

The outcomes of this project will improve customer experience of music platform users. There are over 230 million premium users of Spotify. Average revenue per user is 4.27 Euros. These new playlists will potentially attract more customers and increase revenue.

## **Data**

‘spotify_songs.csv’ from [this](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs) source is the main dataset used in this project. To this dataset, lyrics for ~2000 songs are taken from two other datasets: ‘lyrics_10k.csv’ from [here](https://www.kaggle.com/datasets/evabot/spotify-lyrics-dataset) and `labeled_lyrics_cleaned.csv’ from [here](https://www.kaggle.com/datasets/edenbd/150k-lyrics-labeled-with-spotify-valence). 

The final data frame df_fin has 28347 rows and 25 columns. These columns are:

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

**`energy:`** [as described in original source] Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.

**`key:`** [as described in original source] The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation . E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.

**`loudness:`** [as described in original source] The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.

**`mode:`**  [as described in original source] Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.

**`speechiness:`**  [as described in original source] Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.

**`acousticness:`** A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.

**`instrumentalness:`** [as described in original source] Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.

**`liveness:`** [as described in original source] Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.

**`valence:`** [as described in original source] A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).

**`tempo:`** [as described in original source] The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.

**`duration_ms:`** Track durations in milliseconds

**`track_lyrics:`** 1926 non-null values, lyrics added by merging with two separate datasets as described above.

**`artist_gender:`** 4 categories ‘male’, ‘female’, ‘unknown’, ‘andy’; created using the gender_guesser.detector package.

I created a subset of this dataframe by dropping all the rows with null values for the ‘track_lyrics’ column. There are 2 additional columns in this smaller dataframe with 1926 rows. These columns are described below.

**`compound:`** Compound sentiment score, calculated using the nltk.sentiment.vader package. (range: -1 to 1)

**`unique_cwc:`** Count of unique content words in each lyric, calculated using the spaCy package. 
