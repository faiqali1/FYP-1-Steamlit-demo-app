from collections import Counter
import plotly.express as px
import pandas as pd
import streamlit as st

from pyabsa import ATEPCCheckpointManager

data = None

err_msg = None


def load_aspect_extractor_model():
    model_path = 'state_dict_model_FACT_LCF_ATEPC/fast_lcf_atepc_custom_dataset_cdw_apcacc_89.89_apcf1_75.38_atef1_80.49'
    return ATEPCCheckpointManager.get_aspect_extractor(checkpoint=model_path, auto_device='mps')

# Decorate this function with @st.cache to cache the model loading


@st.cache(allow_output_mutation=True)
def load_model_wrapper():
    return load_aspect_extractor_model()


# Using the wrapped model loading function
aspect_extractor = load_model_wrapper()


st.title('Aspect Based Sentiment Analysis Batch Review Visualisation')
st.write('This page allows you to upload a .csv file containing reviews. The reviews will be analysed and visualised to show the aspect terms present in the file.')

st.caption('The graphs below are interactive. You can hover over the graphs to see more information. You can also zoom in and out of the graphs.')
st.divider()
st.markdown(
    '> *Any CSV file should only contain one column of text. The text should be in the first column only, any subsequent columns will be ignored. A maximum of file size of 10 MB is allowed.*')
uploaded_file = st.file_uploader("Click to add files", type="csv")


if uploaded_file is not None:
    size = uploaded_file.size

    # Check if the file is empty
    if size <= 0:
        st.markdown(
            '> :red[The file is empty. Please check again and reupload the file.]')

    elif size > 1000000:
        st.markdown(
            '> :red[The file is too large. Please check again and reupload the file.]')
    else:
        try:
            dataframe = pd.read_csv(uploaded_file)

            # Validate if the first column contains strings
            if not all(isinstance(item, str) for item in dataframe.iloc[:, 0]):
                st.markdown(
                    '> :red[The file contains non-string characters. Please check and reupload the file.]')
            else:
                aspect_extractor = load_aspect_extractor_model()

                # Extracting text from the first column
                text_data = dataframe.iloc[:, 0].dropna().tolist()

                # Perform aspect-based sentiment analysis
                if text_data:
                    atepc_result = aspect_extractor.extract_aspect(
                        inference_source=text_data,
                        pred_sentiment=True
                    )
                    data = pd.DataFrame(atepc_result)
                else:
                    st.markdown(
                        '> :red[No valid text data found in the file.]')

        except Exception as e:
            st.markdown(f'> :red[Error in processing the file: {e}]')


# we drop the ones with no aspects terms
if data is not None and err_msg is None:

    data = data[data['aspect'].astype(str) != '[]']
    # streamlit text on screen

    # cleaning the data
    data['aspect'] = data['aspect'].astype(str)
    data.loc[:, 'aspect'] = data['aspect'].str.replace('[', '')
    data.loc[:, 'aspect'] = data['aspect'].str.replace(']', '')
    data.loc[:, 'aspect'] = data['aspect'].str.replace("'", '')

    data['sentiment'] = data['sentiment'].astype(str)
    data.loc[:, 'sentiment'] = data['sentiment'].str.replace('[', '')
    data.loc[:, 'sentiment'] = data['sentiment'].str.replace(']', '')
    data.loc[:, 'sentiment'] = data['sentiment'].str.replace("'", '')

    data['confidence'] = data['confidence'].astype(str)
    data.loc[:, 'confidence'] = data['confidence'].str.replace('[', '')
    data.loc[:, 'confidence'] = data['confidence'].str.replace(']', '')
    data.loc[:, 'confidence'] = data['confidence'].str.replace("'", '')

    aspect_df = data['aspect'].str.split(
        ',', expand=True).stack().reset_index(level=1, drop=True)
    sentiment_df = data['sentiment'].str.split(
        ',', expand=True).stack().reset_index(level=1, drop=True)
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
    st.write('The top 20 aspects are shown below.')
    st.plotly_chart(fig)

    sentiment_freq = Counter(data['sentiment'].values)

    # Create a pie chart for Sentiment Distribution
    fig = px.pie(values=list(sentiment_freq.values()), names=list(
        sentiment_freq.keys()), title='Sentiment Distribution')
    # fig.show()

    st.header('Sentiment Distribution')
    st.write('The sentiment distribution is shown below.')
    st.plotly_chart(fig)

    # Count the number of comments for each aspect

    # Get the aspects that have 10 or more comments
    # aspects_with_10_or_more_comments = aspect_counts[aspect_counts >= 1].index

    # # Filter the data to include only the aspects with 10 or more comments
    # filtered_data = data[data['aspect'].isin(aspects_with_10_or_more_comments)]

    # # Calculate the average confidence level for each aspect
    # average_confidence = filtered_data.groupby(
    #     'aspect')['confidence'].mean().reset_index()

    # # Plotly visualization
    # fig = px.bar(average_confidence, x='aspect', y='confidence', title='Average Aspect Confidence Analysis', labels={
    #     'confidence': 'Average Confidence Level', 'aspect': 'Aspect'})
    # # fig.show()

    # st.header('Average Aspect Confidence Analysis')
    # st.write('The average aspect confidence analysis is shown below. Aspect terms with less than 10 comments have been ignored for this visualisation. The aspect with the highest average confidence is *packing* with an average confidence of 0.998. The aspect with the lowest average confidence is *height* with an average confidence of 0.875.')
    # st.plotly_chart(fig)

    # yes

    # Group by aspect and sentiment and count occurrences
    aspect_sentiment_count = data.groupby(
        ['aspect', 'sentiment']).size().reset_index(name='Count')

    COUNT_THRESHOLD = 1

    # Filter to only include aspects with more than 10 comments
    aspect_sentiment_count = aspect_sentiment_count[aspect_sentiment_count['Count'] > 0]

    # Create a bar chart for the count of positive and negative comments for each aspect
    fig = px.bar(aspect_sentiment_count, x='aspect', y='Count', color='sentiment', title='Count of Positive and Negative Comments per Aspect',
                 labels={'Count': 'Number of Comments',
                         'Aspect': 'Aspect', 'Sentiment': 'Sentiment'},
                 color_discrete_map={'positive': 'green', 'neutral': 'blue', 'negative': 'red'})
    fig.update_layout(barmode='group')
    # fig.show()

    st.header('Count of Positive and Negative Comments per Aspect')
    st.write('The count of positive and negative comments per aspect is shown below.')
    st.plotly_chart(fig)

    # fig_confidence_distribution = px.box(data, x='sentiment', y='confidence',
    #                                      title="Distribution of Confidence Scores by Sentiment",
    #                                      labels={'sentiment': 'Sentiment', 'confidence': 'Confidence Score'})
    # # fig_confidence_distribution.show()

    # st.header('Distribution of Confidence Scores by Sentiment')
    # # st.write('The distribution of confidence scores by sentiment is shown below. The sentiment with the highest median confidence score is *positive* with a confidence score of 0.9983. The sentiment with the lowest median confidence score is *neutral* with a confidence score of 0.9465.')
    # st.write('The distribution of confidence scores by sentiment is shown below. *Positive* sentiment has the highest median confidence score of 0.9983. *Negative* sentiment has a median confidence score of 0.9978. *Neutral* sentiment has the lowest median confidence score of 0.9465.')
    # st.plotly_chart(fig_confidence_distribution)
