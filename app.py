import pandas as pd
import streamlit as st
import nltk
import uniq_token as uq
import webbrowser
import streamlit.components.v1 as components

from lexrank import LexRank
from lexrank.algorithms.power_method import stationary_distribution
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import MWETokenizer
from nltk.tokenize import word_tokenize
from collections import Counter

st.title('LexSumy_v1.1')
st.write('Made by @Hardus Tukan')
link = r'https://drive.google.com/file/d/1KQbGHhnQWj60uOmehyaaNQZFvdSVyNQN/view?usp=sharing'
if st.button('Download Korpus File'):
    webbrowser.open_new_tab(link)

### Upload File ###
file_names = []
#check row_text 
raw_texts = []
upload_file = st.file_uploader('Input Documents', type="txt", accept_multiple_files=True)
for upload_files in upload_file:
    byte_data = str(upload_files.read(),"utf-8")
    raw_texts.append(byte_data)
    title_file_name = upload_files.name.replace('.txt','')
    file_names.append(title_file_name)

factory = StemmerFactory()
stemmer = factory.create_stemmer()
listStopword =  set(stopwords.words('indonesian'))

docs = [i.strip("[]").split("\n") for i in raw_texts]


remove_list  = " ".join([str(t) for t in raw_texts])

expander = st.expander("Teks Awal")
expander.write(docs)

### Preprocessing ###
original_sentc = [sentence for sentence in nltk.sent_tokenize(remove_list)] 

def preprocessing_text(text):
  tokenizer = nltk.RegexpTokenizer(r"\w+")
  tokenized = [tokenizer.tokenize(s.lower()) for s in text]
  important_token = []
  for sent in tokenized:
    important_token.append(sent)
  sw_removed = [' '.join(t) for t in important_token]
  stemmed_sent = [stemmer.stem(sent) for sent in sw_removed]
  return stemmed_sent

mwe = MWETokenizer([k.split() for k in uq.unique_token], separator='_')
def phrase_preprocessing_text(text):
  tokenized_paragraph = [token for token in mwe.tokenize(word_tokenize(text))]
  tokenized_paragraph = ' '.join([str(t) for t in tokenized_paragraph])
  phrase_prepc = [sentence for sentence in nltk.sent_tokenize(tokenized_paragraph)]
  return phrase_prepc



main_sentences = st.sidebar.selectbox(
    'Pilih Sentences',
    ('Original Sentences',"Preprocessing Sentences",'Phrase Sentences')
)

set_threshold = st.sidebar.selectbox(
    'Atur Threshold',
    (0.1,0.2,0.3)
)

def get_main_sentences(name,remove_list,original_sentc):
    sentc = None
    if name == 'Original Sentences':
        sentc = original_sentc
    elif name == 'Preprocessing Sentences':
        sentc = preprocessing_text(original_sentc)
    else:
        sentc = phrase_preprocessing_text(remove_list)
    return sentc


def get_threshold(number):
    numb = None
    if number == 0.1:
        numb = .1
    elif number == 0.2:
        numb = .2
    else:
        numb = .3
    return numb


def visualize(sentence_list, best_sentences):
    text = ''
    for sentence in sentence_list:
        if sentence in best_sentences:
            text += ' ' + str(sentence).replace(sentence,f"<mark>{sentence}</mark>")
        else:
            text += ' ' + sentence
    return text

sentences = get_main_sentences(main_sentences,remove_list,original_sentc)
th = get_threshold(set_threshold)

column_table = pd.DataFrame({'Sentence':sentences})
st.dataframe(column_table)

st.subheader("LexRank Summary")
sm1 = sentences # teks query
sm2 = docs  # teks utama
sum_size = int(len(sm1) * 0.25)

    

### LEXRANK LIBRARY ###
def main():
    try:
        lxr = LexRank(sm2,stopwords=listStopword)
        summary = lxr.get_summary(sm1,summary_size=sum_size,threshold=th)
        scores_cont = lxr.rank_sentences(
            sm1,
            threshold=th,
            fast_power_method=True,
        )
        ordered_score = sorted(((scores_cont[i],score) for i,score in enumerate(summary)),reverse=True)
        # ordered_score = sorted((score) for score in enumerate(summary))
        st.caption("score")
        st.table(ordered_score)
        best_sentences = []
        number_of_sentences = sum_size
        
        for sentence in range(number_of_sentences):
            best_sentences.append(ordered_score[sentence][1])
        summarize = " ".join(best_sentences)
        
        with st.expander("Code"):
            
            with st.echo():
                
                t_f = lxr.tokenize_sentence
                tf_scores = [
                    Counter(t_f(sentence)) for sentence in sm1
                ]

                tf = tf_scores
                st.write(tf)
                idf = lxr._calculate_idf(tf)
                st.write(idf)
                
                
            with st.echo():
                idf_modified_csn = lxr._calculate_similarity_matrix(tf)
                st.write(idf_modified_csn)
                
            with st.echo():
                markov_m = lxr._markov_matrix(idf_modified_csn)
                markov_m_w_th = lxr._markov_matrix_discrete(idf_modified_csn,th)
                stat_distr_1 = stationary_distribution(markov_m)
                stat_distr_2 = stationary_distribution(markov_m_w_th)
                st.write(stat_distr_1)
                st.write(stat_distr_2)
        
        st.caption("Summary")
        st.write(summarize)
        
        with st.expander("HTML"):
            html_object  = visualize(sm1, best_sentences)
            components.html(html_object,width=680,height=600,scrolling=True)
        
        st.download_button(label='Download Teks' ,data=summarize,file_name=
                        'Summary.txt')
    except ValueError:
        st.write('no data') 

if __name__ == '__main__':
    main()