import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv('data/atepc_result.csv')
data = data[['aspect', 'sentiment', 'confidence']]
data = data[data['aspect'] != '[]']

data.loc[:, 'aspect'] = data['aspect'].str.replace('[', '')
data.loc[:, 'aspect'] = data['aspect'].str.replace(']', '')
data.loc[:, 'aspect'] = data['aspect'].str.replace("'", '')

data.loc[:, 'sentiment'] = data['sentiment'].str.replace('[', '')
data.loc[:, 'sentiment'] = data['sentiment'].str.replace(']', '')
data.loc[:, 'sentiment'] = data['sentiment'].str.replace("'", '')

data.loc[:, 'confidence'] = data['confidence'].str.replace('[', '')
data.loc[:, 'confidence'] = data['confidence'].str.replace(']', '')
data.loc[:, 'confidence'] = data['confidence'].str.replace("'", '')


aspect_df = data['aspect'].str.split(
    ',', expand=True).stack().reset_index(level=1, drop=True)
sentiment_df = data['sentiment'].str.split(
    ',', expand=True).stack().reset_index(level=1, drop=True).str.strip()
confidence_df = data['confidence'].astype(str).str.split(
    ',', expand=True).stack().reset_index(level=1, drop=True)

final_data = pd.DataFrame({
    'aspect': aspect_df,
    'sentiment': sentiment_df,
    'confidence': confidence_df
})

# Convert confidence column back to float type
final_data['confidence'] = final_data['confidence'].astype(float)

data = final_data


# vislualsing results

aspect_counts = data['aspect'].value_counts()

# Select the top N aspects to display (optional)
N = 15
top_aspects = aspect_counts.head(N)


fig, ax = plt.subplots(figsize=(20, 6))
sns.barplot(x=top_aspects.index, y=top_aspects.values)
plt.xlabel('Aspect')
plt.ylabel('Frequency')
plt.title(f'Top {N} Aspects')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

st.write("# Top 15 Aspects")
st.pyplot(fig)


aspect_sentiment_counts = data.groupby(
    ['aspect', 'sentiment']).size().unstack()


filtered_data = data[data['aspect'].isin(top_aspects.index)]

# Plotting the grouped bar chart using seaborn
# color_dict = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}

fig, ax = plt.subplots(figsize=(20, 6))


# plot the data
sns.barplot(data=filtered_data, x='aspect', y='confidence',
            hue='sentiment', ax=ax)

# rotate x labels for better visibility if there are many aspects
plt.xticks(rotation=90)

st.write("# Aspect Sentiment Confidence")
st.pyplot(fig)

# plt.show()

pivot_df = filtered_data.pivot_table(
    values='confidence', index='aspect', columns='sentiment', aggfunc=np.mean)

fig, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(pivot_df, annot=True, center=0.0, ax=ax)

st.write("# Aspect Sentiment Confidence")
st.pyplot(fig)


# Plotting the grouped bar chart using seaborn
color_dict = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}

# map sentiments to colors in the DataFrame
filtered_data['color'] = filtered_data['sentiment'].map(color_dict)
st.write('# Aspect Sentiment Classification')
fig, ax = plt.subplots(figsize=(10, 6))

# plot the data
# you can adjust the multiplier on 'confidence' in s to get an appropriate size for your bubbles
scatter = ax.scatter(filtered_data['aspect'], filtered_data['sentiment'],
                     s=filtered_data['confidence']*1000, c=filtered_data['color'], alpha=0.6)

# rotate x labels for better visibility
plt.xticks(rotation=90)

st.pyplot(fig)
