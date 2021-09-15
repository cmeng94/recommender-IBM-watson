import numpy as np
import pandas as pd
import sys # can use sys to take command line arguments

import recommender_functions as rf

class Recommender():
    '''
    What is this class all about - write a really good doc string here
    '''
    def __init__(self, ):
        '''
        no required attributes
        '''



    def fit(self, interactions_pth, articles_pth):
        '''
        fit the recommender to your dataset and also have this save the results
        to pull from when you need to make predictions
        '''
        self.df = pd.read_csv(interactions_pth)
        email_encoded = rf.email_mapper(self.df)
        del self.df['email']
        del self.df['Unnamed: 0']
        self.df['user_id'] = email_encoded

        self.user_item = rf.create_user_item_matrix(self.df)
        self.n_users = self.user_item.shape[0]
        self.all_user_ids = set(self.user_item.index)

        df_content = pd.read_csv(articles_pth)
        del df_content['Unnamed: 0']
        df_content.drop_duplicates(subset='article_id', keep='first', inplace=True)

        df1 = self.df[['article_id','title']]
        df2 = df_content[['article_id','doc_full_name']]
        df2.columns = ['article_id','title']

        self.df_docs = pd.concat([df1, df2], ignore_index=True)
        self.df_docs.drop_duplicates(subset='article_id', keep='first', inplace=True)
        self.df_docs.sort_values(by='article_id', inplace=True)
        self.df_docs.reset_index(drop=True, inplace=True)

        num_interactions = dict.fromkeys(self.df_docs['article_id'],0)
        for _id in self.df['article_id']:
            num_interactions[_id] += 1

        self.df_docs['num_interactions'] = self.df_docs['article_id'].apply(lambda x: num_interactions[x])
        self.n_articles = self.df_docs.shape[0]
        self.all_article_ids = set(self.df_docs['article_id'])

        self.top_articles = rf.get_top_articles(self.df, self.df_docs, self.user_item.shape[1])
        self.top_5_articles = self.top_articles[:5]
        self.top_10_articles = self.top_articles[:10]
        self.top_20_articles = self.top_articles[:20]

    def make_recs(self, _id, id_type='user', num_recs=10):
        '''
        given a user id or a article id that an individual likes
        make recommendations
        '''

        rec_ids, rec_names = None, None

        if id_type == 'user':
            _id = int(float(_id))
            if _id in self.all_user_ids:
                rec_ids, rec_names = rf.make_user_user_recs(_id, self.user_item, self.df_docs, 
                                                            self.top_articles, num_recs)
            else:
                rec_ids, rec_names = self.top_articles
                rec_ids = rec_ids[:num_recs]
                rec_names = rec_names[:num_recs]
                print("\n Since this a new user, we recommend the most popular articles.")

        elif id_type == 'article':
            _id = float(_id)
            if _id in self.all_article_ids:
                rec_ids, rec_names = rf.make_content_recs(_id, self.df_docs, num_recs)
            else:
                print("\n Sorry, we are not able to make recommendations because this article is not in our database.")


        return rec_ids, rec_names


if __name__ == '__main__':

    import recommender as r
    rec = r.Recommender()
    rec.fit('data/user-item-interactions.csv', 'data/articles_community.csv')


    if len(sys.argv) == 4:
        num_recs = int(sys.argv[3])
        recs = rec.make_recs(sys.argv[1], sys.argv[2], num_recs)
    elif len(sys.argv) == 3:
        recs = rec.make_recs(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        recs = rec.make_recs(sys.argv[1])
    else:
        print("Wrong number of input arguments.")

    print("\n ===================== RECOMMENDATIONS ===================== \n")
    for i in range(len(recs[0])):
        print("ID: {}\t Title: {}".format(recs[0][i], str.capitalize(recs[1][i])))

    print('\n')
