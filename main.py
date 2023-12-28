import streamlit as st
from yaml.loader import SafeLoader
import yaml
import streamlit_authenticator as stauth
from streamlit_authenticator import Authenticate
import pandas as pd
# from pages.prediction_page import prediction_page


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
    st.write(
        'Enter a review below to get a summary of aspect terms and sentence polarity ')

    form = st.form(key='my_form')
    txt = form.text_input(label='Enter some text')
    submit_button = form.form_submit_button(label='Submit')

    from pyabsa import ATEPCCheckpointManager

    model_path = 'state_dict_model_FACT_LCF_ATEPC/fast_lcf_atepc_custom_dataset_cdw_apcacc_89.89_apcf1_75.38_atef1_80.49'

    aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint=model_path,
                                                                   auto_device='mps')  # False means load model on CPU

    atepc_result = aspect_extractor.extract_aspect(inference_source=[txt],  # data needs to be a python list...
                                                   pred_sentiment=True,)  # Predict the sentiment of extracted aspect terms

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

        df = pd.DataFrame(data)

        df = df[['Aspect', 'Sentiment', 'Confidence']]

        # df
        st.dataframe(df.style.applymap(highlight_positive),
                     use_container_width=True, hide_index=True)
    except:
        st.write('**No aspect terms found**')


if authentication_status:
    # after login succesful
    authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{name}*')
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
