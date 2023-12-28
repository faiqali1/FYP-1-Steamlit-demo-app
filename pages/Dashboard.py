import json
import ast
from collections import Counter
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

data = pd.read_csv('result/atepc_result.csv')
# data.head(10)

# we drop the ones with no aspects terms
data = data[data['aspect'] != '[]']
data.info()

# cleaning the data

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

# Join the separate dataframes back together to create the final dataframe
final_data = pd.DataFrame({
    'aspect': aspect_df,
    'sentiment': sentiment_df,
    'confidence': confidence_df
})

# assert all columns are of the same length
assert (len(aspect_df) == len(sentiment_df) == len(confidence_df))

# Convert confidence column back to float type
final_data['aspect'] = final_data['aspect'].str.strip()
final_data['sentiment'] = final_data['sentiment'].str.strip()
final_data['confidence'] = final_data['confidence'].astype(float)

data = final_data

data.head(10)


aspect_counts = data['aspect'].value_counts()

# Select the top N aspects to display (optional)
N = 20
top_aspects = aspect_counts.head(N)

fig = px.bar(x=top_aspects.index, y=top_aspects.values, labels={
             'x': 'Aspect', 'y': 'Frequency'}, title=f'Top {N} Aspects')
fig.update_layout(autosize=False, width=800, height=500)
# fig.update_xaxes(tickangle=90)
# fig.show()
st.plotly_chart(fig)


sentiment_freq = Counter(data['sentiment'].values)

# Create a pie chart for Sentiment Distribution
fig = px.pie(values=list(sentiment_freq.values()), names=list(
    sentiment_freq.keys()), title='Sentiment Distribution')
# fig.show()
st.plotly_chart(fig)

# Count the number of comments for each aspect


# Get the aspects that have 10 or more comments
aspects_with_10_or_more_comments = aspect_counts[aspect_counts >= 10].index

# Filter the data to include only the aspects with 10 or more comments
filtered_data = data[data['aspect'].isin(aspects_with_10_or_more_comments)]

# Calculate the average confidence level for each aspect
average_confidence = filtered_data.groupby(
    'aspect')['confidence'].mean().reset_index()

# Plotly visualization
fig = px.bar(average_confidence, x='aspect', y='confidence', title='Average Aspect Confidence Analysis', labels={
    'confidence': 'Average Confidence Level', 'aspect': 'Aspect'})
# fig.show()
st.plotly_chart(fig)

# yes


# Group by aspect and sentiment and count occurrences
aspect_sentiment_count = data.groupby(
    ['aspect', 'sentiment']).size().reset_index(name='Count')

COUNT_THRESHOLD = 10

# Filter to only include aspects with more than 20 comments
aspect_sentiment_count = aspect_sentiment_count[aspect_sentiment_count['Count'] > COUNT_THRESHOLD]


# Create a bar chart for the count of positive and negative comments for each aspect
fig = px.bar(aspect_sentiment_count, x='aspect', y='Count', color='sentiment', title='Count of Positive and Negative Comments per Aspect',
             labels={'Count': 'Number of Comments',
                     'Aspect': 'Aspect', 'Sentiment': 'Sentiment'},
             color_discrete_map={'positive': 'green', 'neutral': 'blue', 'negative': 'red'})
fig.update_layout(barmode='group')
# fig.show()
st.plotly_chart(fig)


fig_confidence_distribution = px.box(data, x='sentiment', y='confidence',
                                     title="Distribution of Confidence Scores by Sentiment",
                                     labels={'sentiment': 'Sentiment', 'confidence': 'Confidence Score'})
# fig_confidence_distribution.show()
st.plotly_chart(fig_confidence_distribution)
