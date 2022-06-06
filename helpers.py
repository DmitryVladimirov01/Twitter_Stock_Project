import streamlit as st
from spacy_streamlit import load_model
import settings
from decorators import benchmark, long_running, error_wrap_auth
import tweepy
from spacy import displacy
from flair.data import Sentence


#%%
from get_tickers import get_tickers
@st.cache
def get_tickers_cached(): return get_tickers()


#%%
@st.cache(allow_output_mutation=True)
def load_full_model(spacy_model):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk import download
    download("vader_lexicon")
    vader_sia = SentimentIntensityAnalyzer()

    from flair.models import TextClassifier, SequenceTagger
    from flair.data import Sentence
    flair_classifier = TextClassifier.load("en-sentiment")
    flair_tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")


    nlp = load_model(spacy_model)

    from spacy.language import Language
    @Language.component("hashtag", retokenizes=True)
    def symbol_pipe(doc):
        i = []
        for token_index, token in enumerate(doc):
            if token.text == "#":
                i.append(token_index)
            elif token.text == "$" and not str(doc[token_index + 1]).isnumeric() and token_index != len(doc):
                i.append(token_index)
        for idx, token_pos in enumerate(i):
            with doc.retokenize() as retokenizer:
                token_pos = token_pos - idx
                retokenizer.merge(doc[token_pos: token_pos + 2])
        return doc
    try:
        # nlp.add_pipe("hashtag", last=True)
        nlp.add_pipe("spacytextblob")
        pass
    except:
        pass
    return nlp, vader_sia, flair_classifier, flair_tagger

#%%
# @error_wrap_auth
def authenticate_twitter():
    # st.cache работает медленнее, чем чистая функция
    auth = tweepy.OAuthHandler(settings.TWITTER_API_KEY, settings.TWITTER_API_KEY_SECRET)
    auth.set_access_token(settings.TWITTER_ACCESS_TOKEN, settings.TWITTER_ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)
    return api


#%%
def extract_metrics(doc, vader_sia, flair_classifier):
    from flair.data import Sentence
    metrics ={}
    metrics["textblob"] = doc._.blob.sentiment
    metrics["vader"] = vader_sia.polarity_scores(doc.text)
    sentence = Sentence(doc.text)
    flair_classifier.predict(sentence)
    for label in sentence.labels:
        metrics["flair"] = f'{label.value} {label.score}'
    return metrics

#%%
def extract_flair_ner(text, flair_tagger):
    sentence = Sentence(text)
    flair_tagger.predict(sentence)
    return sentence

def flair_to_displacy(sentence):
    dict_flair = {}
    dict_flair["text"] = sentence.to_original_text()
    temp = []
    for span in sentence.get_spans('ner'):
        temp.append({'end': span.end_position, 'start': span.start_position, 'label': span.tag})
    dict_flair['ents'] = temp
    st.markdown(displacy.render(dict_flair, style="ent", manual=True), unsafe_allow_html=True)


#%%
if __name__ == '__main__':
    print("Это вспомогательный модуль и он не должен исполняться самостоятельно")