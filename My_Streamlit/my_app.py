import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the combined data
combined_data = pd.read_csv('combined_data.csv')  # Replace 'combined_data.csv' with the path to your combined data file

# Perform PCA on selected columns
selected_columns = [col for col in combined_data.columns if col not in ['track_popularity', 'Prediction', 'track_id']]
X = combined_data[selected_columns]
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Rename dimensions
combined_data['x'] = X_pca[:, 0]
combined_data['y'] = X_pca[:, 1]
combined_data['z'] = X_pca[:, 2]

# Create a Streamlit app
st.title('Explore the Model Predictions!')

# Display the combined data
st.write('Data:')
st.write(combined_data)

# Create a dropdown menu for subgenre features
subgenre_features = ['All Subgenres'] + [col for col in combined_data.columns if col.startswith('playlist_subgenre_')]
selected_subgenre = st.selectbox('Select a Subgenre', subgenre_features)

# Create a dropdown menu for album decades
album_decades = ['All Decades'] + [col for col in combined_data.columns if col.startswith('album_decade_')]
selected_decade = st.selectbox('Select a Decade', album_decades)

# Check if 'All Subgenres' is selected
if selected_subgenre != 'All Subgenres':
    # Filter the data based on the selected subgenre feature
    filtered_data = combined_data[combined_data[selected_subgenre] == 1]
else:
    filtered_data = combined_data

# Check if 'All Decades' is selected
if selected_decade != 'All Decades':
    # Further filter the data based on the selected album decade
    filtered_data = filtered_data[filtered_data[selected_decade] == 1]

# Get the overall minimum and maximum values of 'track_popularity' and 'Prediction' columns
min_value = min(filtered_data['track_popularity'].min(), filtered_data['Prediction'].min())
max_value = max(filtered_data['track_popularity'].max(), filtered_data['Prediction'].max())

# Create subplots for actual and predicted values
fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'projection': '3d'})

# Actual scores plot
sc1 = axes[0].scatter(filtered_data['x'], filtered_data['y'], filtered_data['z'], c=filtered_data['track_popularity'], cmap='viridis', vmin=min_value, vmax=max_value)
axes[0].set_title('Actual Scores', fontsize=16)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_zlabel('z')
axes[0].set_xlim(filtered_data['x'].min(), filtered_data['x'].max())
axes[0].set_ylim(filtered_data['y'].min(), filtered_data['y'].max())
axes[0].set_zlim(filtered_data['z'].min(), filtered_data['z'].max())

# Predicted scores plot
sc2 = axes[1].scatter(filtered_data['x'], filtered_data['y'], filtered_data['z'], c=filtered_data['Prediction'], cmap='viridis', vmin=min_value, vmax=max_value)
axes[1].set_title('Predicted Scores', fontsize=16)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_zlabel('z')
axes[1].set_xlim(filtered_data['x'].min(), filtered_data['x'].max())
axes[1].set_ylim(filtered_data['y'].min(), filtered_data['y'].max())
axes[1].set_zlim(filtered_data['z'].min(), filtered_data['z'].max())

# Create a color bar for each plot
cbar1 = fig.colorbar(sc1, ax=axes[0])
cbar1.set_label('Popularity Score')
cbar2 = fig.colorbar(sc2, ax=axes[1])
cbar2.set_label('Popularity Score')

# Adjust layout
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)




