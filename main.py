import streamlit as st
from yaml.loader import SafeLoader
import yaml
import streamlit_authenticator as stauth
from streamlit_authenticator import Authenticate
import pandas as pd
import plotly.express as px


def process_and_label_aspects(api_data):
    def process_and_label_aspect(api_data):
        def transform_data(api_data):
            transformed_data = []
            sentences_processed = set()

            for entry_list in api_data:  # Expecting a list of lists
                for entry in entry_list:
                    sentence = entry["sentence"]
                    if sentence in sentences_processed:
                        continue

                    transformed_entry = {
                        "sentence": sentence,
                        "tokens": entry["tokens"],
                        "position": [],
                        "Aspect": [],
                        "Sentiment": []
                    }

                    for aspect_entry in entry_list:
                        if aspect_entry["sentence"] == sentence:
                            transformed_entry["position"].append(
                                aspect_entry["position"][0])
                            transformed_entry["Aspect"].append(
                                aspect_entry["Aspect"])
                            sentiment = "Positive" if aspect_entry["Sentiment"] == "Positive" else (
                                "Negative" if aspect_entry["Sentiment"] == "Negative" else "Neutral")
                            transformed_entry["Sentiment"].append(sentiment)

                    sentences_processed.add(sentence)
                    transformed_data.append(transformed_entry)

            return transformed_data

        def label_aspects_by_tokens(transformed_data):
            labeled_sentences = []

            for entry in transformed_data:
                sentence_tokens = entry["tokens"]
                aspects_data = zip(entry["position"],
                                   entry["Aspect"], entry["Sentiment"])

                for pos, aspect, sentiment in aspects_data:
                    if sentiment == "Positive":
                        color = ":green"
                    elif sentiment == "Negative":
                        color = ":red"
                    else:  # Neutral
                        color = ":blue"

                    if sentence_tokens[pos] == aspect:
                        sentence_tokens[pos] = f"{color}[{aspect}]"

                labeled_sentence = " ".join(sentence_tokens)
                labeled_sentences.append(labeled_sentence)

            return labeled_sentences

        transformed_data = transform_data(api_data)
        labeled_sentences = label_aspects_by_tokens(transformed_data)
        return labeled_sentences

    # Example usage with your API data

    labeled_sentences = process_and_label_aspect(api_data)

    for sentence in labeled_sentences:
        return (sentence)


#! Hashing passwords
# hashed_passwords = stauth.Hasher(['abc', 'def']).generate()
# print(hashed_passwords)

#! Preauthorised users
# User ID: jsmith
# Password: abc

# User ID: rbriggs
# Password: def

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)


authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')

# Add tailwind colors
TAILWIND_600_GREEN = '#16a34a'
TAILWIND_600_RED = '#dc2626'


def highlight_positive(val):
    if val == 'Positive':
        return f'color: {TAILWIND_600_GREEN}'
    elif val == 'Negative':
        return f'color: {TAILWIND_600_RED}'
    else:
        return ''


def prediction_page():
    st.title('Aspect Based Sentiment Analysis')
    st.write(f'Welcome *{name}* :smile:!')
    st.write('This is a web application for Aspect Based Sentiment Analysis. The application allows you to enter a sentence and the model will extract the aspect terms and predict the sentiment of each aspect term.')
    st.write('\n')
    st.caption('Please enter a sentence below and click submit to extract the aspect terms to predict the sentiment of each aspect term.')

    form = st.form(key='my_form')
    txt = form.text_input(label='Enter a sentence')
    submit_button = form.form_submit_button(label='Submit')

    from pyabsa import ATEPCCheckpointManager

    model_path = 'state_dict_model_FACT_LCF_ATEPC/fast_lcf_atepc_custom_dataset_cdw_apcacc_89.89_apcf1_75.38_atef1_80.49'

    @st.cache(allow_output_mutation=True)
    def load_model():
        return ATEPCCheckpointManager.get_aspect_extractor(checkpoint=model_path,
                                                           auto_device='mps')

    aspect_extractor = load_model()
    # TODO: add error checking to prevent empty input

    atepc_result = aspect_extractor.extract_aspect(inference_source=[txt],  # data needs to be a python list...
                                                   pred_sentiment=True,)
    # Predict the sentiment of extracted aspect terms

    obj = atepc_result
    # atepc_resul t

    data = []
    try:
        for o in obj:
            for i in range(len(o['aspect'])):
                data.append({
                    'sentence': o['sentence'],
                    'IOB': o['IOB'],
                    'tokens': o['tokens'],
                    'Aspect': o['aspect'][i],
                    'position': o['position'][i],
                    'Sentiment': o['sentiment'][i],
                    'probs': o['probs'][i],
                    'Confidence': o['confidence'][i],
                })
        # data
        df = pd.DataFrame(data)

        df = df[['Aspect', 'Sentiment', 'Confidence']]
        # data

        markdown_text = process_and_label_aspects([data])
        st.write('\n')
        st.divider()
        st.markdown(f"> **{markdown_text}**")
        st.caption('The aspect terms are highlighted in green for positive sentiment, red for negative sentiment and blue for neutral sentiment.')
        st.divider()
        st.header('Aspect Confidence Analysis')

        st.write('\n')
        # df
        st.write('The table below shows the confidence level of each aspect term.')
        st.dataframe(df.style.applymap(highlight_positive),
                     use_container_width=True, hide_index=True)

        # Create a bar chart
        fig = px.bar(df, x='Aspect', y='Confidence', color='Sentiment',
                     labels={'Aspect': 'Aspect', 'Confidence': 'Confidence',
                             'Sentiment': 'Sentiment'},
                     title='Aspect Confidence Analysis',
                     color_discrete_map={'Positive': TAILWIND_600_GREEN, 'Neutral': 'blue', 'Negative': TAILWIND_600_RED})

        # Display the bar chart in Streamlit
        st.write('\n')
        st.write(
            'The bar chart below shows the confidence level of each aspect term.')
        st.plotly_chart(fig)

    except:
        st.write('**No aspect terms found**')


if authentication_status:
    # after login succesful
    authenticator.logout('Logout', 'main')
    prediction_page()


elif authentication_status == False:
    st.error('Username/password is incorrect')


# if authentication_status:
#     try:
#         if authenticator.reset_password(username, 'Reset password'):
#             st.success('Password modified successfully')
#     except Exception as e:
#         st.error(e)

# try:
#     if authenticator.register_user('Register user', preauthorization=False):
#         st.success('User registered successfully')
# except Exception as e:
#     st.error(e)
