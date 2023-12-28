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

# streamlit text on screen
st.title('Aspect Based Sentiment Analysis Training Data Visualisation')
st.write('The data below is the training data used to train the model. The data is visualised to show the distribution of the data and the confidence level of the model.')

st.caption('The graphs below are interactive. You can hover over the graphs to see more information. You can also zoom in and out of the graphs.')
st.divider()
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


aspect_counts = data['aspect'].value_counts()

# Select the top N aspects to display (optional)
N = 20
top_aspects = aspect_counts.head(N)

fig = px.bar(x=top_aspects.index, y=top_aspects.values, labels={
             'x': 'Aspect', 'y': 'Frequency'}, title=f'Top {N} Aspects')
fig.update_layout(autosize=False, width=800, height=500)
# fig.update_xaxes(tickangle=90)
# fig.show()

st.header('Top 20 Aspects')
st.write('The top 20 aspects are shown below. The aspect with the highest frequency is *delivery* with a frequency of 679. The aspect with the lowest frequency is *delivered* with a frequency of 56.')
st.plotly_chart(fig)


sentiment_freq = Counter(data['sentiment'].values)

# Create a pie chart for Sentiment Distribution
fig = px.pie(values=list(sentiment_freq.values()), names=list(
    sentiment_freq.keys()), title='Sentiment Distribution')
# fig.show()

st.header('Sentiment Distribution')
st.write('The sentiment distribution is shown below. The sentiment with the highest frequency is *positive* with a frequency of 4,709. The sentiment with the lowest frequency is *neutral* with a frequency of 299.')
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

# max_confidence = average_confidence['confidence'].max()
# max_confidence_aspect = average_confidence[average_confidence['confidence']
#                                            == max_confidence]['aspect'].values[0]
# min_confidence = average_confidence['confidence'].min()
# min_confidence_aspect = average_confidence[average_confidence['confidence']
#                                            == min_confidence]['aspect'].values[0]

# max_confidence_aspect
# max_confidence
# min_confidence_aspect
# min_confidence

st.header('Average Aspect Confidence Analysis')
st.write('The average aspect confidence analysis is shown below. Aspect terms with less than 10 comments have been ignored for this visualisation. The aspect with the highest average confidence is *packing* with an average confidence of 0.998. The aspect with the lowest average confidence is *height* with an average confidence of 0.875.')
st.plotly_chart(fig)

# yes


# Group by aspect and sentiment and count occurrences
aspect_sentiment_count = data.groupby(
    ['aspect', 'sentiment']).size().reset_index(name='Count')

COUNT_THRESHOLD = 10

# Filter to only include aspects with more than 10 comments
aspect_sentiment_count = aspect_sentiment_count[aspect_sentiment_count['Count'] > COUNT_THRESHOLD]

# count the max number of positve aspects
# max_positive = aspect_sentiment_count[aspect_sentiment_count['sentiment']
#                                       == 'Positive']['Count'].max()
# max_positive_aspect = aspect_sentiment_count[aspect_sentiment_count['Count']
#                                              == max_positive]['aspect'].values
# max_positive_aspect
# max_positive

# count the max number of negative aspects
# max_negative = aspect_sentiment_count[aspect_sentiment_count['sentiment']
#                                       == 'Negative']['Count'].max()
# max_negative_aspect = aspect_sentiment_count[aspect_sentiment_count['Count']
#                                              == max_negative]['aspect'].values[0]
# max_negative_aspect
# max_negative

# Create a bar chart for the count of positive and negative comments for each aspect
fig = px.bar(aspect_sentiment_count, x='aspect', y='Count', color='sentiment', title='Count of Positive and Negative Comments per Aspect',
             labels={'Count': 'Number of Comments',
                     'Aspect': 'Aspect', 'Sentiment': 'Sentiment'},
             color_discrete_map={'positive': 'green', 'neutral': 'blue', 'negative': 'red'})
fig.update_layout(barmode='group')
# fig.show()

st.header('Count of Positive and Negative Comments per Aspect')
st.write('The count of positive and negative comments per aspect is shown below. Aspect terms with less than 10 comments have been ignored for this visualisation. The aspect with the highest number of positive comments is *delivery* with 562 positive comments. The aspect with the highest number of negative comments is *app* with 157 negative comments.')
st.plotly_chart(fig)


fig_confidence_distribution = px.box(data, x='sentiment', y='confidence',
                                     title="Distribution of Confidence Scores by Sentiment",
                                     labels={'sentiment': 'Sentiment', 'confidence': 'Confidence Score'})
# fig_confidence_distribution.show()

st.header('Distribution of Confidence Scores by Sentiment')
# st.write('The distribution of confidence scores by sentiment is shown below. The sentiment with the highest median confidence score is *positive* with a confidence score of 0.9983. The sentiment with the lowest median confidence score is *neutral* with a confidence score of 0.9465.')
st.write('The distribution of confidence scores by sentiment is shown below. *Positive* sentiment has the highest median confidence score of 0.9983. *Negative* sentiment has a median confidence score of 0.9978. *Neutral* sentiment has the lowest median confidence score of 0.9465.')
st.plotly_chart(fig_confidence_distribution)
