# Twitter Sentiment Analysis

## Table of Contents

- Sentimental Analysis
- Project Overview 
- Problem Statement 
- Data Understanding
- Methodology
- Evaluation
- Conclusion
- Recommendation
- Next Steps


# Business understanding
In the modern digital era, social media platforms like Twitter have emerged as influential channels for obtaining immediate customer feedback and viewpoints. It is crucial for businesses to comprehend the sentiment conveyed by customers regarding particular brands and products. This understanding allows companies to make well-informed choices, improve customer contentment, and uphold a favorable brand image.

The objective of this project is to create a sentiment analysis model designed specifically for evaluating Twitter data associated with Google, Apple, and various other products.


# Problem Statement

As a consulting firm, Twitter has entrusted us with the responsibility of constructing a model capable of assessing the sentiment expressed in a Tweet by analyzing its content. This model should accurately classify Twitter sentiments regarding Apple and Google products into categories of positivity, negativity, or neutrality. The primary objective is to extract valuable insights from public sentiment, enabling businesses to make well-informed decisions in shaping their strategic approaches and enhancing overall customer satisfaction.

# Data Understanding

In this project, I made use of a dataset sourced from CrowdFlower through Data.world, containing around 9,000 tweet sentiments regarding Apple and Google products. The dataset encompasses various columns, including tweet_text, emotion_in_tweet_is_directed_at, and is_there_an_emotion_directed_at_a_brand_or_product.
My primary goal is to create a sentiment analysis model capable of effectively categorizing tweets into positive, negative, or neutral sentiment categories.
#### Objectives for the project:

1. To develop a binary classifier that can classify tweets into positive or negative sentiment categories. This will serve as a proof of concept and provide a foundation for further analysis. The classifier will be a Logistic regression model and the benchmark accuracy will be 85%.
2. To expand to a multiclass classifier, thereby incorporating the neutral tweets to create a multiclass classifier that can accurately classify tweets as positive, negative, or neutral. This will provide a more comprehensive sentiment analysis of the tweets. The classifier will be a XGBoost model and MultinomialNB the benchmark accuracy will be 70%. 
3. To compare sentiment between Apple and Google products by analyzing the sentiment distribution of tweets mentioning Apple,  Google, and other products.

# Project Overview
The objective of this project is to develop a sentiment analysis model that can accurately classify customer reviews into positive, negative, or neutral sentiment categories. The model aims to automate the process of sentiment classification, enabling businesses to quickly understand customer sentiment at scale.
To achieve this, I implemented the following steps:
1. Data Preparation: I splitted the dataset into training and testing sets to evaluate the performance of the model. I used X% of the data for training and Y% for testing.
2. Feature Extraction: I applied various techniques such as bag-of-words, count vectorizer 
3. Model Selection: We experimented with different algorithms, such as logistic regression, linear SVC, multinnomial NB, xGboost. I evaluated the models using appropriate evaluation metrics such as accuracy, , F1 score, recall, precision and classification report.
4.Model Training and Evaluation: I trained the selected model on the training data and evaluated its performance on the testing data.
5. Model Deployment: Once we achieved satisfactory performance, I deployed the sentiment analysis model to make predictions on new, unseen customer reviews. using streamlit library.
By undertaking this project, my aim was to provide businesses with valuable insights into customer sentiment, enabling them to make data-driven decisions, improve customer satisfaction, and enhance their overall brand reputation.
Please note that the above example is just a brief illustration, and you should customize it according to your specific project requirements and scope.

Methodology
Data Preprocessing: I performed data cleaning and preprocessing steps such as removing special characters, stopwords, and performing tokenization. We also applied lemmatization using NLTK's lemmatizer to reduce words to their common root form.

Vectorization Techniques:
a. Bag-of-Words (CountVectorizer): I used sklearn's CountVectorizer to convert the text data into a numerical representation, capturing the frequency of words in each document. This approach creates a matrix of word counts.

b. Classification Models: I trained and evaluated several classification models using the vectorized data, including:

1.Logistic Regression Classifier for the binary classifier model

2.Multinomial Naive Bayes , Linear SVM and XGBoost model for the multiclass classification model 


c. Model Evaluation: For each model, we employed  accuracy for logistic regression model,multinommial NB anduy xg boost but for Linear SVC i used both classification report, accuracy score, f1 score, recall and precision to evaluate the model's ability to correctly predict sentiment.

d. Handling Class Imbalance: we tried to address this issue using synthetic Minority random Undersampling but didnt improve the accuracy of the XGBoost model.

# Evaluation
To evaluate the performance of the NLP sentiment analysis model, I conducted thorough testing and analysis using various evaluation metrics. The following evaluation results provide insights into the effectiveness of our approach:
* Accuracy:
1. The binary sentiment analysis model achieved an overall accuracy of 90% which surpused our trget of achieving 85%.
2. The linear SVM model achieved an accuracy score of 91%
3. The MultinomialNb multiclass model achieved an overall accuracy of 65%
4. our XGboost multiclass model achieved an overall accuracy of 67%

# Visualizations
 ![Sentiment Count Distribution](Sentiment%20count%20D.PNG)



 # Findings
Most of the tweets were directed to no specific brand.
Positive sentiments had the highest count compared to Negative sentiments, indicating that most people in general liked respective brands(Google and Apple)
Most of the positive tweets were directed to Apple brands
In the field of sentiment analysis, one of the significant challenges is dealing with language ambiguity and sarcasm detection. Natural language is complex and often subjective, making it difficult to accurately interpret sentiments from text.
On average most of the tweets were 10-15 words long - more words increase ambiguity.
NLP is a difficult task to model accurately.
Most tweets were directed to None brand category. This may indicate that customers were not engaging with the brand.

# Recommendations
From the analysis I recommend that there be more customer engagement.
Probably check on this areas;
Churn ratio - rate at which customers discontinue their relationship with a product company within a given time period
Social media influencers through brand or product endorsement
Customer feedback - The brands can introduce a rating system to accurately capture the sentiments of their customers

# Nextsteps
1. In  future work, I plan to explore advanced techniques such as incorporating attention mechanisms, using ensemble methods to further enhance the model's performance by incorporating domain-specific and fine-tuning the model on industry-specific datasets could improve its accuracy and adaptability.
2. By considering these evaluation metrics, addressing limitations, and planning for future improvements, we aim to develop a robust NLP sentiment analysis solution that effectively captures sentiment in text data.
3. Looking for a better dataset 

# Limitations
- Class Imbalance Issue: The dataset suffers from class imbalance, where one sentiment class is dominant while others are underrepresented. This can result in biased models that are more accurate for the majority class but perform poorly on the minority classes. Addressing this issue is important to ensure fair and balanced sentiment analysis.
- Limited Dataset Size: The dataset used for sentiment analysis is relatively small, which can limit the model's ability to capture the full complexity of sentiments expressed in text. A larger and more diverse dataset would provide a broader representation of sentiments and improve the model's performance and generalization.
- Language Ambiguity and Sarcasm Detection: Language can be inherently ambiguous, and detecting sarcasm in text adds an extra layer of complexity. Sarcasm detection is challenging due to the subtleties and nuances involved. Developing robust strategies to handle language ambiguity and detect sarcasm is crucial for accurate sentiment analysis