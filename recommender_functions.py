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
    '''
    INPUT:
    df - dataframe with user-article interactions with email column

    OUTPUT:
    email_encoded - column where email is encoded by user ids
    '''
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
    INPUT:
    text - text to be tokenized

    OUTPUT:
    tokens - tokens of input text using the following transformations:
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
    Returns a matrix with user ids as rows and article ids on the columns having 1 values where a user interacted with 
    an article and a 0 otherwise
    '''
    
    user_item = df.pivot_table(index=['user_id'], columns=['article_id'], aggfunc=lambda x: 1, fill_value=0)
    user_item.columns = [x[1] for x in user_item.columns]
    
    return user_item 

def create_docs_matrix(df, df_content):
    '''
    INPUT:
    df - dataframe with user-article interactions
    df_content - dataframe with article id and contents

    OUTPUT:
    df_docs - dataframe that contains all articles in df and df_content, with the following columns:
                  article_id
                  title
                  num_interactions: number of times the article is read
              sorted by descending num_interactions then ascending article_id
    '''

    df1 = df[['article_id','title']]
    df2 = df_content[['article_id','doc_full_name']]
    df2.columns = ['article_id','title']

    df_docs = pd.concat([df1, df2], ignore_index=True)
    df_docs.drop_duplicates(subset='article_id', keep='first', inplace=True)
    df_docs.sort_values(by='article_id', inplace=True)
    df_docs.reset_index(drop=True, inplace=True)

    num_interactions = dict.fromkeys(df_docs['article_id'],0)
    for _id in df['article_id']:
        num_interactions[_id] += 1

    df_docs['num_interactions'] = df_docs['article_id'].apply(lambda x: num_interactions[x])  
    return df_docs  


def get_article_names(article_ids, df_docs):
    '''
    INPUT:
    article_ids - list of article ids
    df_docs - dataframe returned by "create_docs_matrix"
    
    OUTPUT:
    article_names - list of article names associated with the input list of article ids 
    '''

    article_names = []
    for _id in article_ids:
        article_names.append(df_docs[df_docs['article_id']==_id]['title'].values[0])
    
    return article_names 


def get_top_articles(df, df_docs, num_recs=10):
    '''
    INPUT:
    n -  the number of top articles to return
    df - dataframe with user-article interactions
    df_docs - dataframe returned by "create_docs_matrix" function
    
    OUTPUT:
    top_articles_ids - list of ids of the top n article titles 
    top_articles - list of titles of the top n article titles 
    '''
    top_article_ids = df.groupby('article_id').size().sort_values(ascending=False).head(num_recs).index
    top_articles = get_article_names(top_article_ids, df_docs) 
    
    return top_article_ids, top_articles


def get_user_article_ids(user_id, user_item):
    '''
    INPUT:
    user_id - user id
    user_item - user item matrix returned by "create_user_item_matrix" function
                has 1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    article_ids - list of the article ids seen by the user
    article_names - list of article names seen by the user
    
    Description:
    Returns a list of the article_ids and article titles that have been seen by a user
    '''

    user_row = user_item.loc[user_id]
    article_ids = (user_row[user_row == 1]).index
    article_ids = list(article_ids)
        
    return article_ids


def get_top_sorted_users(user_id, user_item):
    '''
    INPUT:
    user_id - user id
    df - dataframe with user-article interactions
    user_item - user item matrix returned by "create_user_item_matrix" function
                has 1's when a user has interacted with an article, 0 otherwise
            
    OUTPUT:
    neighbors_df - dataframe with:
                       neighbor_id - is a neighbor user_id
                       similarity - measure of the similarity of each user to the provided user_id
                       num_interactions - the number of articles viewed by the user 
                   Sorted by descending similarity and descending num_interactions
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
    user_id - user id
    df_docs - dataframe returned by "create_docs_matrix" function
    top_articles - list of all articles with decreasing number of times read
    num_recs - the number of recommendations for the user (default is 10)
    
    OUTPUT:
    rec_ids - list of recommendations for the user by article id
    rec_names - list of recommendations for the user by article title
    
    Description:
    Loops through the users with a positive semilarity to the input user in the order of decreasing similarity.
    For each user, finds articles the user hasn't seen before and provides them as recs.
    Does this until num_recs recommendations are found.
    
    Notes:
    * Users that have the most total article interactions are chosen before those with fewer article interactions.
    * Articles with the most total interactions are chosen before those with fewer total interactions. 
    * If less than num_recs articles are found, the rest is filled with most popular articles.
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
    '''
    INPUT:
    df_docs - dataframe returned by "create_docs_matrix" function    

    OUTPUT:
    df_cont_transf - dataframe with article_id as index, and titles transformed by sklearn's tfidfvectorizer.
    '''
    
    vectorizer = TfidfVectorizer(tokenizer=tokenize)
    matrix = vectorizer.fit_transform(df_docs['title'])
    
    df_cont_transf = pd.DataFrame(data=matrix.toarray(), index=df_docs['article_id'], 
                                  columns=vectorizer.get_feature_names())
    
    return df_cont_transf

def make_content_recs(article_id, df_docs, num_recs=10):
    '''
    INPUT:
    article_id: article id
    df_docs - dataframe returned by "create_docs_matrix" function    
    num_recs - the number of recommendations for the article (default is 10)

    OUTPUT:
    rec_ids - list of articles ids of most similar articles to input article
    rec_names - list of articles names of most similar articles to input article

    Description:
    Return articles that are most similar to the input article. 
    Results are sorted by decreasing similarity and decreasing num_interactions.
    '''

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
    
