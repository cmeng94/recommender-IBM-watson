# Recommender Systems with IBM Watson

## Table of Contents
1. [Project Description](#intro)
2. [Class Descriptions](#class)
3. [Getting Started](#start)
4. [Contact](#contact)
5. [Acknowledgement and Licensing](#acknowledge)

<a id='intro'></a>
## 1. Project Description
For this project, we analyze the interactions that users have with articles on the IBM Watson Studio platform, and make recommendations to them about new articles they might like. The deliverable of this project is a Recommender Class that can be called within a Python script or function to make recommendations.

In this project, three types of recommender systems are implemented:
* Rank Based Recommender,
* Collaborative Filtering Recommender, and
* Content Based Recommender.

<a id='class'></a>
## 2. Class Description
The Recommender Class is built upon interactions that users have with articles on the IBM Watson Studio platform. The class has 2 functions and 8 attributes.

**Class Functions**:
1. `fit(interactions_pth, articles_pth)`: fits recommender on input data sets and assigns values to class attribute.
2. `make_recs(_id, id_type='user', num_recs=10)`: make recommendations for input user or article

**Class Attributes**:
1. `n_users`:         total number of users in the data base
2. `all_user_ids`:    list of length n_users containing all user ids 
3. `n_articles`:      total number of articles in the data base
4. `all_article_ids`: list of length n_articles containing all article ids
5. `top_articles`: 
    - top_articles[0]: list of all article ids with decreasing popularity
    - top_articles[1]: list of all article names with decreasing popularity
6. `top_5_articles`:  5 most popular article ids and article names
7. `top_10_articles`: 10 most popular article ids and article names
8. `top_20_articles`: 20 most popular article ids and article names

<a id='start'></a>
## 3. Getting Started

<a id='contact'></a>
## 4. Contact
**Chang Meng**
* Email: chang_meng@live.com
* Website: [https://sites.google.com/view/changmeng](https://sites.google.com/view/changmeng)

<a id='acknowledge'></a>
## 5. Acknowledgement and Licensing
This project is part of the [Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) program at [Udacity](https://www.udacity.com/). User and article data sets are provided by [IBM Watson Studio](https://www.ibm.com/cloud/watson-studio).