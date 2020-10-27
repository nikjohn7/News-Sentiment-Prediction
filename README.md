# News Sentiment Prediction Methods

This was a private invite-only challenge on Hackerearth. The task was to predict the sentiment of a news article based on its title and headline. A sentiment score between -1 to 1 was to be given to both the title and headline

## Dataset

This is a large data set of news items and their respective social feedback on multiple platforms: `Facebook`, `Google+` and `LinkedIn`. 

The collected data relates to a period of 8 months, between November 2015 and July 2016, accounting for about 100,000 news items on four different topics: `economy`, `microsoft`, `obama` and `palestine`. 

This data set is tailored for evaluative comparisons in predictive analytics tasks, although allowing for tasks in other research areas such as topic detection and tracking, sentiment analysis in short text, first story detection or news recommendation. 

The dataset contains news headlines and their respective information. They include
- Facebook news
- Google-Plus news
- Linkedln news

The attributes for each of the tables are : 
- **IDLink (numeric):** Unique identifier of news items
- **Title (string):** Title of the news item according to the official media sources
- **Headline (string):** Headline of the news item according to the official media sources
- **Source (string):** Original news outlet that published the news item
- **Topic (string):** Query topic used to obtain the items in the official media sources
- **Publish-Date (timestamp):** Date and time of the news items' publication
- **Sentiment-Title (numeric):** Sentiment score of the text in the news items' title
- **Sentiment-Headline (numeric):** Sentiment score of the text in the news items' headline 
- **Facebook (numeric):** Final value of the news items' popularity according to the social media source Facebook
- **Google-Plus (numeric):** Final value of the news items' popularity according to the social media source Google+
- **LinkedIn (numeric):** Final value of the news items' popularity according to the social media source LinkedIn
- **SentimentTitle:** Sentiment score of the title, Higher the score, better is the impact or +ve sentiment and vice-versa. _(Target Variable 1)_
- **SentimentHeadline:** Sentiment score of the text in the news items' headline. Higher the score, better is the impact or +ve sentiment. _(Target Variable 2)_



## Solution 1: Custom Transform pipelines with Multi-Output Regressor

A pipeline in sklearn is a set of chained algorithms to extract features, preprocess them and then train or use a machine learning algorithm. Each pipeline has a number of steps, which is defined as a list of tuples. The first element in the tuple is the name of the step in the pipeline. The second element of the tuple is the transformer. When predicting an outcome the pipeline preprocesses the data before running it through the estimator to predict the outcome.

A pipeline component is defined as a `TransformerMixin` derived class with two important methods:

`fit` - Uses the input data to train the transformer

`transform` - Takes the input features and transforms them

The `fit` method is used to train the transformer. This method is used by components such as the CountVectorizer to setup the internal mappings for words to vector elmeents. It gets both the features and the expected output.

The `transform` method only gets the features that need to be transformed. It returns the transformed features.

The final step in the pipeline is the estimator. The estimator can be a classifier, regression algorithm, a neural network or even some unsupervised algorithm.

Here I created a custom pipeline used Scikit-learn's `TransformerMixin` and used a regressor like `Random Forrest` or `XGBRegressor`

## Solution 2: GloVe Embeddings + BiLSTM Network

`GloVe` is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

### Why GloVe with BiLSTM?
- GloVe word embeddings are generated from a huge text corpus like Wikipedia and are able to find a meaningful vector representation for each word in the news data.
- This allows us to use Transfer learning and train further over our data.
- In this project I have used the 50-dimensional data.
- When used with a BiLSTM, the results seem to be better than Bag-of-Words and Tf-Idf vectorization methods.

In order to get a continuous valued output between -1 and 1, I used **Mean squared error** as the loss function for my network, and I used a **custom activation function** in the output layer of my network. 


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
