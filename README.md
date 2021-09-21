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

In this project, four types of recommender systems are implemented:
* Rank Based Recommender,
* Collaborative Filtering Recommender, 
* Content Based Recommender, and
* Matrix Factorization Recommender.

<a id='class'></a>
## 2. Class Description
The Recommender Class is built upon interactions that users have with articles on the IBM Watson Studio platform. The class has 2 functions and 13 attributes.

**Class Functions**:
1. `fit(interactions_pth, articles_pth, num_lat=2)`: fits recommender on input data sets and assigns values to class attribute.
2. `make_recs(_id, id_type='user', num_recs=10)`: make recommendations for input user or article.

The recommender function `Recommender.make_recs` works as follows.
* When a known user ID is input, recommendations are first made using **Matrix Factorization**. If `num_recs` is not reached, then **Collaborative Filtering** is used.
* When an unknown user ID is input, **Rank Based** recommender is used.
* When an article ID is input, articles that are most similar to the input article are returned using **Content Based** recommender.

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
9. `user_item`: matrix with user ids as rows and article ids on the columns having 1 values where a user interacted with 
    an article and a 0 otherwise
10. `df_docs`: dataframe that contains all article information from both input paths
11. `u_lat`: U part of the SVD decomposition of `user_item` truncated to rank num_lat
12. `s_lat`: S part of the SVD decomposition of `user_item` truncated to rank num_lat
13. `vt_lat`: Vt part of the SVD decomposition of `user_item` truncated to rank num_lat


<a id='start'></a>
## 3. Getting Started
### Dependencies
The code is developed with Python 3.9 and is dependent on a number of python packages listed in `requirements.txt`. To install required packages, run the following line in terminal:
```sh
pip3 install -r requirements.txt
```

### Installation
To run the code locally, create a copy of this GitHub repository by running the following code in terminal:
```sh
git clone https://github.com/cmeng94/recommender-IBM-watson
```

### Execution
The package can be accessed both in the terminal using command line arguments and inside python code as a module. You need to be located inside the `recommender-IBM-watson` directory to run the code below.

* **Command Line**
    * To make ***num*** recommendations for user with ***user_id***:
    ```sh
    python3 recommender.py user_id num
    ```
    * To make ***num*** recommendations for article with ***article_id***:
    ```sh
    python3 recommender.py article_id article num
    ```

* **Inside Python**, first import and define recommender by executing the following code
    ```sh
    from recommender import Recommender
    rec = Recommender()
    ```
    Then, fit the recommender to the provided data sets
    ```sh
    rec.fit("data/user-item-interactions.csv", "data/articles_community.csv")
    ```

    To 
    * make ***num*** recommendations for user with ***user_id***:
        ```sh
        user_recs = rec.make_recs(user_id, num_recs=num)
        ```
    * make ***num*** recommendations for article with ***article_id***:
        ```sh
        article_recs = rec.make_recs(article_id, 'article', num)
        ```


<a id='contact'></a>
## 4. Contact
**Chang Meng**
* Email: chang_meng@live.com
* Website: [https://sites.google.com/view/changmeng](https://sites.google.com/view/changmeng)

<a id='acknowledge'></a>
## 5. Acknowledgement and Licensing
This project is part of the [Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) program at [Udacity](https://www.udacity.com/). User and article data sets are provided by [IBM Watson Studio](https://www.ibm.com/cloud/watson-studio).