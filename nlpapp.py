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
import gc




st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


st.title('Auto NLP V0.01')

choise = st.checkbox('Show user mannual')



if choise:

    docs = """
        
            ### Follow these steps.
            1.Select the type of file format (currently only sopports .csv and .xlsx).\n
            2.Then import dataset form your pc.\n
            3.Then select the feature columns form the dataset.(can support multiple columns)\n
            ### note: selected columns needs to be text

            ## Filter setction 
            ### In this section you can select to 
            1. remove punctuations.
            2. remove digits.
            3. remove invlid characters eg. emojies
            4. parameter N which discards words of length less than N (default N = 2 )

            ## **Preoprcessing section**
            1. user can remove **stopwords**.\n
            ** note:- this section has 2 options**\n
            a. **nltk** this uses the stopwords found in nltk library\n
            b. **extended** more stopwords than nltk library


            2. **Stemming**

            ** Note : after preproceesing a download link will appear clicking on that will download a file . replace the extension 
            with .csv even if you had imported .xlsx file this app imports 
            only .csv files. Have to do this becz. curretly streamlit doesnt support 
            download wedgit. **


        """

    st.write(docs)



st.write("#### currently only supports .csv and .xlsx file formats")
file_type = st.selectbox("Please Choose a file type",['none','csv','xlsx'])


if file_type == 'csv' or file_type == 'xlsx':


    n = nlp()

    # path  = st.text_input("enter path",'file.csv')
    uploaded_file = st.file_uploader("Choose a  file",type = file_type)

    # encoding  = st.text_input("Enter encoding")


    # if encoding == '':
    #     encoding = 'UTF-8'

    # if label == '':
    #     print('target label not specified')

    # genre = st.radio(
    #      "What's your favorite movie genre",
    #      ('Comedy', 'Drama', 'Documentary'))



    if uploaded_file != None:


        flag = 0

        export = 0

        if file_type == 'csv':
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)    
        uploaded_file.close()

        st.dataframe(df[:10])

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
                st.write('dropping all the rows with nan values')
                df.dropna(inplace = True)
                st.write("## **after cleaning**")
                data = {'Columns':[],'Field Length':[],'Nulls':[]}
                
                for col in df.columns:
                    data['Columns'].append(col)
                    data['Field Length'].append(len(df[col]))
                    data['Nulls'].append(df[col].isnull().sum()) 

                tmp = pd.DataFrame(data)

                st.dataframe(tmp)



        st.write("###  **select feature column (not target column)**")
        select = st.multiselect("",options = df.columns)




        if select != []:



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
            # lemmatizer_option = st.radio("Use Lemmatize: ? (not implemented yet)",("yes","np"),index = 1)
            # vectorizer_option = st.radio("Vectorizer (not implemented yet)",("Count","Tfidf"),index = 0)

            

            preprocess_button = st.button("Start Preprocessing")


            if preprocess_button:

                #Selecring columns from the dataframe
                t1 = time.time()

                
                
                #apply settings 

                #filter settings

                
                remove_chars =  True if remove_chars == 'yes' else  False
                remove_digits = True if remove_digits == 'yes' else False
                remove_punct = True if remove_punct == 'yes' else False


                st.write("settings")
                # st.write(f" N = {num}, remove invalid characters = {remove_chars}, remove punctuations = {remove_punct}, remove digits = {remove_digits}")
                # st.write(f"stemmer = {stemmer_option},stopwords = {stopword_option},vectorizer = {vectorizer_option}")

                settings = pd.DataFrame(data = {'properties':['N','rm. invlaid chars.','remove punct.','remove digits','stemmer','stopwords'],
                                     'selected':[num,remove_chars,remove_punct,remove_digits,stemmer_option,stopword_option]
                                    })

                st.dataframe(settings)    


                n.set_parms(N = num , punct = remove_punct,digits = remove_digits,validate_chrs = remove_chars)

                #preprocessing settings

                n.apply_settings('word',stopwords=stopword_option,stemmer=stemmer_option)
            
                
                for col in select:
                        df[col] = df[col].map(n.nlp_cleaner)

                
                st.write('completed in ',int(time.time()-t1),' sec')
                st.dataframe(df.head(10))    
                st.write('total rows::',len(df))
               


              
              
                            


                
        else:
            st.write("## Please Select one or  ** *multiple Feature columns* ** for preprocessing")

  #show download link 
if st.button("Download file"):
        csv = df.to_csv(index=False)

            
        #delete df

        # st.write(df.memory_usage(deep = True).values.sum()/1024)
        del df
        with st.spinner("preparing"):
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        del csv
        gc.collect()
        href = f'<a href="data:file/csv;base64,{b64}" download>Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        st.markdown(href, unsafe_allow_html=True)
