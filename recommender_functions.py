import numpy as np
import pandas as pd

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def email_mapper(df):
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded


def tokenize(text):

    '''
    The function tokenizes input text.
    Input:
    text: text to be tokenized
    Output:
    tokens: tokens of input text, transfomations include:
        1) replacing urls with placeholder
        2) normalization
        3) removing punctuations
        4) tokenize
        5) removing stopwords
        6) lemmatization
    '''
    
    # replace urls with urlplaceholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")    
    
    # normalize, remove puntuations, tokenize, remove stopwords and lemmatize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stopwords.words("english")]
    
    return tokens


def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    user_item - user item matrix 
    
    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''
    # Fill in the function here
    
    user_item = df.pivot_table(index=['user_id'], columns=['article_id'], aggfunc=lambda x: 1, fill_value=0)
    user_item.columns = [x[1] for x in user_item.columns]
    
    return user_item # return the user_item matrix 


def get_article_names(article_ids, df_docs):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook
    
    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the title column)
    '''

    article_names = []
    for _id in article_ids:
        article_names.append(df_docs[df_docs['article_id']==_id]['title'].values[0])
    
    return article_names 


def get_top_articles(df, df_docs, num_recs=10):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles_ids - (list) A list of ids of the top 'n' article titles 
    top_articles - (list) A list of names of the top 'n' article titles 
    
    '''
    # Your code here
    top_article_ids = df.groupby('article_id').size().sort_values(ascending=False).head(num_recs).index
    top_articles = get_article_names(top_article_ids, df_docs)
    # top_articles = []
    # for _id in article_ids:
    #     top_articles.append(df_docs[df_docs['article_id']==_id]['title'].values[0])    
    
    return top_article_ids, top_articles


def get_user_article_ids(user_id, user_item):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the doc_full_name column in df_content)
    
    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''

    user_row = user_item.loc[user_id]
    article_ids = (user_row[user_row == 1]).index
        
    return article_ids


def get_top_sorted_users(user_id, user_item):
    '''
    INPUT:
    user_id - (int)
    df - (pandas dataframe) df as defined at the top of the notebook 
    user_item - (pandas dataframe) matrix of users by articles: 
            1's when a user has interacted with an article, 0 otherwise
    
            
    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user 
                    
    Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                    highest of each is higher in the dataframe
     
    '''
    
    neighbors_df = pd.DataFrame(data={'neighbor_id': user_item.index})
    neighbors_df['similarity'] = np.dot(user_item, user_item.loc[user_id])
    neighbors_df['num_interactions'] = user_item.sum(axis=1)
    neighbors_df.sort_values(by=['similarity', 'num_interactions'], ascending=False, inplace=True)
    neighbors_df = neighbors_df[neighbors_df['neighbor_id'] != user_id]
    
    return neighbors_df # Return the dataframe specified in the doc_string


def make_user_user_recs(user_id, user_item, df_docs, top_articles, num_recs=10):
    '''
    INPUT:
    user_id - (int) a user id
    num_recs - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until num_recs recommendations are found
    
    Notes:
    * Choose the users that have the most total article interactions 
    before choosing those with fewer article interactions.

    * Choose articles with the most total interactions 
    before choosing those with fewer total interactions. 
   
    '''
    
    user_articles = set(get_user_article_ids(user_id, user_item))
                       
    neighbor_df = get_top_sorted_users(user_id, user_item)
    article_order = top_articles[0]
    article_order = {art:idx for idx, art in enumerate(article_order)}
    
    rec_ids = []
    for i in range(neighbor_df.shape[0]):
        if neighbor_df.iloc[i]['similarity'] == 0:
            break

        neighbor_articles = set(get_user_article_ids(neighbor_df.iloc[i]['neighbor_id'], user_item))
        rec_articles = list(neighbor_articles - user_articles)
        rec_articles.sort(key=article_order.get)
        rec_ids.extend(rec_articles)
        if len(rec_ids) >= num_recs:
            break
    
    if len(rec_ids) < num_recs:
        rec_ids.extend(top_articles[0])

    rec_ids = rec_ids[:num_recs]
    rec_names = get_article_names(rec_ids, df_docs)
    return rec_ids, rec_names


def create_article_content_matrix(df_docs):

    
    vectorizer = TfidfVectorizer(tokenizer=tokenize)
    matrix = vectorizer.fit_transform(df_docs['title'])
    
    df_cont_transf = pd.DataFrame(data=matrix.toarray(), index=df_docs['article_id'], 
                                  columns=vectorizer.get_feature_names())
    
    return df_cont_transf

def make_content_recs(article_id, df_docs, num_recs=10):

    df_cont_transf = create_article_content_matrix(df_docs)

    df_docs_copy = df_docs
    
    df_docs_copy['similarity'] = np.dot(df_cont_transf, df_cont_transf.loc[article_id])
    df_docs_copy.sort_values(by=['similarity', 'num_interactions'], ascending=False, inplace=True)
    df_docs_copy = df_docs_copy[df_docs_copy['article_id'] != article_id]

    rec_ids = list(df_docs_copy['article_id'])
    if len(rec_ids) > num_recs:
        rec_ids = rec_ids[:num_recs]    

    rec_names = get_article_names(rec_ids, df_docs)
    return rec_ids, rec_names 
    
