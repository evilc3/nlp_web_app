"""
 import file 
 display data
 show data info.


"""
import streamlit as st
import pandas as pd
import numpy as np
import base64
from nlp_base_webapp import *
import time


n = nlp()

path  = st.text_input("enter path",'file.csv')
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

encoding  = st.text_input("Enter encoding")


# if encoding == '':
#     encoding = 'UTF-8'

# if label == '':
#     print('target label not specified')

# genre = st.radio(
#      "What's your favorite movie genre",
#      ('Comedy', 'Drama', 'Documentary'))


st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


flag = 0
df = pd.read_csv(path)

st.dataframe(df.head())

# label = st.selectbox("Select target",df.columns)
data = {'Columns':[],'Field Length':[],'Nulls':[]}

st.markdown("<h3>Describe Data</h3>",unsafe_allow_html = True)


for col in df.columns:
    data['Columns'].append(col)
    data['Field Length'].append(len(df[col]))
    data['Nulls'].append(df[col].isnull().sum()) 

tmp = pd.DataFrame(data)

st.dataframe(tmp)


if tmp['Nulls'].sum() == 0:
    st.write("### ** No Nulls Found **")
else:
    st.write("Nulls Found")    
    flag = 1


if flag == 1:

    if st.button("Clean"):
        df.dropna(inpace = True)

st.write("###  **select feature column (not target column)**")
select = st.multiselect("",options = df.columns)

st.write(f'#### **you have selected *{select}* to preprocess **')

st.write('## **Filter Settings**')

remove_punct  = st.radio("Remove Punctuations ?",("yes","no"),index = 1)
remove_digits = st.radio("Remove Digits ?",("yes","no"),index = 1)
remove_chars = st.radio("Remove Invalid Characters ?",("yes","no"),index = 1)

# st.write("#### Enter value for N will ommite word if its length < N")
num = st.number_input("Enter value for N will ommite word if its length < N",value = 2)

# st.write("Value selected",num)


st.write("## **PreProcessing Settings**")

stemmer_option = st.radio("Stemmers Options:",('Porter','SnowBall','ISR'),index = 0)
stopword_option = st.radio("StopWord Option:",('nltk','extended'),index = 0)
lemmatizer_option = st.radio("Use Lemmatize: ?",("yes","np"),index = 1)
vectorizer_option = st.radio("Vectorizer",("Count","Tfidf"),index = 0)





if st.button("Start Preprocessing"):

    #Selecring columns from the dataframe
    t1 = time.time()

    
    
    #apply settings 

    #filter settings

    
    remove_chars =  True if remove_chars == 'yes' else  False
    remove_digits = True if remove_digits == 'yes' else False
    remove_punct = True if remove_punct == 'yes' else False


    st.write("settings")
    st.write(num,remove_chars,remove_punct,remove_digits)
    st.write(stemmer_option,stopword_option,vectorizer_option)

    n.set_parms(N = num , punct = remove_punct,digits = remove_digits,validate_chrs = remove_chars)

    #preprocessing settings

    n.apply_settings('word',stopwords=stopword_option,vectorizer=vectorizer_option,stemmer=stemmer_option)
 

    for col in select:
        df[col] = df[col].map(n.nlp_cleaner)


    st.write(int(time.time()-t1),' sec')
    st.dataframe(df)    

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'

    st.markdown(href,unsafe_allow_html = True)


    

