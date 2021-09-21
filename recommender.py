import numpy as np
import pandas as pd
import sys 

import recommender_functions as rf

class Recommender():
    '''
    This Recommender class is built upon interactions that users have with articles on the IBM Watson Studio platform.
    This class has 2 functions and 8 attributes.

    FUNCTIONS:
        fit(interactions_pth, articles_pth):
            - fits recommender on input data sets and assigns values to class attribute.
        make_recs(_id, id_type='user', num_recs=10)
            - make recommendations for input user or article

    ATTRIBUTES:
        n_users - total number of users in the data base
        all_user_ids - list of length n_users containing all user ids 
        n_articles - total number of articles in the data base
        all_article_ids - list of length n_articles containing all article ids
        top_articles - top_articles[0]: list of all article ids with decreasing popularity
                     - top_articles[1]: list of all article names with decreasing popularity
        top_5_articles - 5 most popular article ids and article names
        top_10_articles - 10 most popular article ids and article names
        top_20_articles - 20 most popular article ids and article names
    '''
    def __init__(self, ):
        '''
        no required attributes
        '''

    def fit(self, interactions_pth, articles_pth, num_lat=2):
        '''
        INPUT:
        interactions_pth - path to data set containing entries of all user-article interaction
                           each row represents an article is read by a user
        articles_pth - path to data set containing titles and contents of articles

        OUTPUT:
        None

        Description:
        Clean input data sets.
        Save relative results as class attributes that are ready to use when making predictions.
        '''

        # clean df data set
        df = pd.read_csv(interactions_pth)
        email_encoded = rf.email_mapper(df)
        del df['email']
        del df['Unnamed: 0']
        df['user_id'] = email_encoded

        # clean df_content data set
        df_content = pd.read_csv(articles_pth)
        del df_content['Unnamed: 0']
        df_content.drop_duplicates(subset='article_id', keep='first', inplace=True)

        # create user_item matrix with user as row and article as column
        # matrix as 1's when a user has interacted with an article, 0 otherwise
        self.user_item = rf.create_user_item_matrix(df)

        # create docs dataframe with article ids, titles, and number of times the article is read
        # df is sorted by decreasing number of times read
        self.df_docs = rf.create_docs_matrix(df, df_content)

        # save attributes relative to data
        self.n_users = self.user_item.shape[0]
        self.all_user_ids = set(self.user_item.index)
        self.n_articles = self.df_docs.shape[0]
        self.all_article_ids = set(self.df_docs['article_id'])

        # save the most popular articles as class attributes
        self.top_articles = rf.get_top_articles(df, self.df_docs, self.user_item.shape[1])
        self.top_5_articles = self.top_articles[:5]
        self.top_10_articles = self.top_articles[:10]
        self.top_20_articles = self.top_articles[:20]

        # save latent features for use in make_recs
        u, s, vt = np.linalg.svd(self.user_item)
        self.s_lat, self.u_lat, self.vt_lat = np.diag(s[:num_lat]), u[:, :num_lat], vt[:num_lat, :]


    def make_recs(self, _id, id_type='user', num_recs=10):
        '''
        INPUT:
        _id - user id or article id
        id_type - type of id, 'user' or 'article' (default is 'user')
        num_recs - the number of recommendations (default is 10)

        OUTPUT:
        rec_ids - list of articles ids of recommendations for input user or article
        rec_names - list of articles names of recommendations for input user or article
        '''

        rec_ids, rec_names = None, None

        if id_type == 'user':
            _id = int(float(_id))
            if _id in self.all_user_ids:

                user_idx = np.where(self.user_item.index == _id)[0][0]
                art_idx = np.where(np.around(np.dot(np.dot(self.u_lat[user_idx,:], self.s_lat), self.vt_lat))==1)
                rec_ids = list(set(self.user_item.columns[art_idx]) - set(rf.get_user_article_ids(_id, self.user_item)))
                article_order = self.top_articles[0]
                article_order = {art:idx for idx, art in enumerate(article_order)}
                rec_ids.sort(key=article_order.get)

                rec_ids_2 = rf.make_user_user_recs(_id, self.user_item, self.df_docs, 
                                                            self.top_articles, num_recs)[0]
                
                for rec_id in rec_ids_2:
                	if len(rec_ids) >= num_recs:
                		break
                	if rec_id not in rec_ids:
                		rec_ids.append(rec_id)

                rec_ids = rec_ids[:num_recs]
                rec_names = rf.get_article_names(rec_ids, self.df_docs)

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
                return


        return rec_ids, rec_names


if __name__ == '__main__':

    # handles command line arguments

    import recommender as r
    rec = r.Recommender()
    rec.fit('data/user-item-interactions.csv', 'data/articles_community.csv')
    id_type = ""
    _id = 0

    if len(sys.argv) == 4:
        try:
            id_type = sys.argv[2]
            _id = sys.argv[1]
            num_recs = int(sys.argv[3])
            recs = rec.make_recs(_id, id_type, num_recs)
        except:
            raise ValueError("Command line arguments are not valid.")
    elif len(sys.argv) == 3:
        if sys.argv[2] == 'user' or sys.argv[2] == 'article':
            try:
                _id = sys.argv[1]
                id_type = sys.argv[2]
                recs = rec.make_recs(_id, sys.argv[2])
            except:
                raise ValueError("Command line arguments are not valid.")                
        else:
            try:
                id_type = "user"
                _id = sys.argv[1]
                num_recs = int(sys.argv[2])
                recs = rec.make_recs(_id, num_recs=num_recs)
            except:
                raise ValueError("Command line arguments are not valid.")

    elif len(sys.argv) == 2:
        try:
            _id = sys.argv[1]
            id_type = "user"
            recs = rec.make_recs(_id)
        except:
            raise ValueError("Command line arguments are not valid.")
           
    else:
        print("Wrong number of input arguments.")

    if recs:
        print("\n ================== RECOMMENDATIONS for {} {} ================== \n".format(str.capitalize(id_type), _id))
        for i in range(len(recs[0])):
            print("ID: {}\t Title: {}".format(recs[0][i], str.capitalize(recs[1][i])))

    print('\n')
