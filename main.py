#%%
import streamlit as st
from streamlit import write
from stqdm import stqdm

import re
re_match = r"[#$]([A-Z]+)"

import spacy
from spacy import displacy
from spacy.tokens import Span
# import spacy_transformers

from spacytextblob.spacytextblob import SpacyTextBlob

from flair.data import Sentence

from helpers import load_full_model, get_tickers_cached, authenticate_twitter, extract_metrics
from helpers import extract_flair_ner, flair_to_displacy
DEBUG = st.sidebar.selectbox("Debug", [False, True])

#%%
spacy_model = st.sidebar.selectbox("Модель spacy", ["en_core_web_trf"])
flair_ner_model = st.sidebar.selectbox("Модель flair ner", ['flair/ner-english-ontonotes-large', 'ner'])
nlp, vader_sia, flair_classifier, flair_tagger = load_full_model(spacy_model)


if not DEBUG:
    #%%
    st.title("Автоматическое прогнозирование цены акций на основе анализа тональности финансовых новостей (на материале англоязычного твиттера)")

    #%%
    api = authenticate_twitter()
    #%%
    tickers = get_tickers_cached()

    #%%
    lang = st.sidebar.radio("Язык твитов", ['en'])
    count = st.sidebar.slider("Количество твитов", 0, 100)
    keyword = st.sidebar.selectbox("Тикер компании", tickers, index=tickers.index("TSLA"))
    #%%
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def extract_tweets_nlp(api_f, keyword, lang, count):
        search_results = api_f.search_tweets("#" + keyword, lang=lang, count=count, tweet_mode="extended")
        return [sr.full_text for sr in search_results]

    @st.cache(allow_output_mutation=True)
    def make_nlp(tweets):
        return list(nlp.pipe(tweets))

    tweets = extract_tweets_nlp(api, keyword, lang, count)
    docs = make_nlp(tweets)
    # write(docs)
    #%%
    metrics_all, flair_ner_all, spacy_sentences = [], [], []
    for doc in docs:
        metrics = extract_metrics(doc, vader_sia, flair_classifier)

        sentence = Sentence(doc.text)
        flair_tagger.predict(sentence)

        spacy_sentence = nlp(' '.join(sentence.to_original_text().split()))

        from spacy.tokens import Span
        for span in sentence.get_spans('ner'):
            idx = [token.idx for token in span.tokens]
            span = Span(spacy_sentence, idx[0] - 1, idx[-1])  # Create a span in Spacy
            spacy_sentence.ents = list(spacy_sentence.ents) + [span]

        metrics_all.append(metrics)
        flair_ner_all.append(sentence)
        spacy_sentences.append(spacy_sentence)
        

    print(flair_ner_all)
    #%%
    write(f"Всего найдено {len(docs)} твитов")
    displayed = st.slider("Сколько вы хотите увидеть твитов?", 0, count)
    for i in range(displayed):
        write(f"##########{i}############")
        st.markdown(displacy.render(docs[i], style="ent"), unsafe_allow_html=True)
        write(metrics_all[i])
        for entity in flair_ner_all[i].get_spans("ner"):
            write(entity)
        print(flair_ner_all[i].to_dict())
        # st.markdown(displacy.render(flair_ner_all[i], style='ent', manual=True))
else:
    texts = ["RT @ProblemSniper: 💭 $TSLA $NVDA $AAPL Price action before split. $AMZN wonderful cup and handle here. Just something to check out over the…",
            "I was hoping it will fall to 680-620 level for me to hoard more. Seriously. I got some at 730 some at 701 and lot of them holding since 580(long time). I am hoping stock split will be in.",
            "Jeremy Grantham , Michael Burry warn of another market plunge. Avoid this beginner mistake when the stock market corrects https://t.co/t0xouwtKF7 $SPY $QQQ $DIA $DJIA #stockmarket #investing #finance #stocks $AAPL $TSLA $AMZN $INTC $NFLX $AMC $NIO $GME",
            "@dissectmarkets I also was interested in $RIVN until I listened to a TED talk where Elon talked about the difficulties getting Tesla to where it is currently. He said the last car company to achieve volume production before $TSLA was Chrysler. $F and $ TSLA only U.S. autos that never bankrupted."]

    docs = nlp.pipe(texts)
    for i, doc in enumerate(docs):
        write(f"##########{i}############")
        write("**Spacy NER**")

        st.markdown(displacy.render(doc, style="ent", ), unsafe_allow_html=True)
        write("**Flair NER**")

        sentence = extract_flair_ner(doc.text, flair_tagger)
        flair_to_displacy(sentence)

        write("**Metrics**")
        metrics = extract_metrics(doc, vader_sia, flair_classifier)
        write(metrics)
        hashtag_tickers = re.findall(re_match, doc.text)
        write(hashtag_tickers)
    pass
