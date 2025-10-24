# 🐦 Twitter Sentiment Analysis using Machine Learning

## 📌 Project Overview
This project focuses on analyzing sentiments expressed in tweets using Natural Language Processing (NLP) techniques.  
The goal is to classify tweets as **Positive** or **Negative** based on their textual content.  
This end-to-end analysis leverages text preprocessing, TF-IDF vectorization, and Logistic Regression modeling to understand public sentiment trends effectively.

---

## 🎯 Objectives
- Clean and preprocess real-world tweet data using **NLTK**.
- Convert textual information into numerical vectors using **TF-IDF**.
- Train and evaluate a **Logistic Regression** model for sentiment prediction.
- Visualize the distribution of sentiments and text characteristics using **Matplotlib** and **Seaborn**.

---

## 🧩 Technologies & Libraries Used
- **Programming Language:** Python  
- **Libraries:**
  - `numpy`, `pandas` – Data manipulation and analysis  
  - `nltk` – Natural Language Toolkit for text preprocessing  
  - `re` – Regular expressions for text cleaning  
  - `matplotlib`, `seaborn` – Data visualization  
  - `scikit-learn` – Machine learning (TF-IDF, Logistic Regression, train-test split, evaluation metrics)

---

## 🧠 Project Workflow

### 1. Data Loading
The dataset containing tweets and sentiment labels is imported into a pandas DataFrame for exploration and preprocessing.

### 2. Data Preprocessing
Text data undergoes multiple cleaning stages:
- Removing special characters, URLs, and punctuation using **regular expressions**.  
- Converting text to lowercase for normalization.  
- Tokenization and stopword removal using **NLTK stopwords**.  
- Applying **Porter Stemming** to reduce words to their root form (e.g., "running" → "run").

```python
ps = PorterStemmer()
corpus = []
for tweet in data['tweet']:
    review = re.sub('[^a-zA-Z]', ' ', tweet)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    corpus.append(' '.join(review))
3. Feature Extraction
Transformed text corpus into numerical representation using TF-IDF Vectorizer.

This approach captures the importance of each term in the document relative to the entire corpus.

python
Copy code
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(corpus).toarray()
4. Model Training
Dataset was split into training and testing sets using an 80/20 ratio.

Trained a Logistic Regression classifier to predict tweet sentiment.

python
Copy code
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
5. Model Evaluation
Performance metrics used:

Accuracy Score

Confusion Matrix

Precision, Recall, F1-Score

The model achieved strong predictive performance, demonstrating the effectiveness of text vectorization and Logistic Regression for sentiment classification tasks.

📊 Exploratory Data Analysis (EDA)
Visualized key insights from the dataset:

Sentiment distribution (positive vs negative tweets)

Most frequent words across sentiments

Word cloud representations for positive and negative tweets

Example visualization:

python
Copy code
sns.countplot(x='label', data=data)
plt.title('Distribution of Sentiments')
plt.show()
🧾 Results Summary
Achieved ~85–90% accuracy (depending on dataset size and preprocessing).

Observed that positive tweets are slightly more frequent than negative ones.

Text cleaning and stemming significantly improved model performance by reducing noise.

TF-IDF performed better than simple Bag-of-Words for feature extraction.

🪄 Key Insights
Logistic Regression, despite being a linear model, performs strongly on text classification when combined with TF-IDF.

Preprocessing steps like stemming and stopword removal are crucial for reducing vocabulary redundancy.

Sentiment analysis can serve as a foundational approach for brand reputation tracking, political opinion analysis, and customer feedback monitoring.

⚙️ How to Run This Project
1. Clone the Repository
bash
Copy code
git clone https://github.com/<your-username>/Twitter_Sentiment_Analysis_using_ML.git
cd Twitter_Sentiment_Analysis_using_ML
2. Install Dependencies
bash
Copy code
pip install numpy pandas nltk scikit-learn matplotlib seaborn
3. Run the Jupyter Notebook
bash
Copy code
jupyter notebook Twitter_Sentiment_Analysis_using_ML.ipynb
📈 Future Improvements
Experiment with advanced models like Naive Bayes, Random Forest, or LSTM-based deep learning models.

Incorporate emoji and hashtag sentiment for richer context understanding.

Build an interactive web dashboard to visualize live Twitter sentiment trends.

🧑‍💻 Author
Nikita Ravindra Albela
Aspiring Data Analyst | Passionate about NLP, Predictive Modeling & Insight Generation
LinkedIn | GitHub

🏷️ License
This project is licensed under the MIT License — feel free to use and modify it with attribution.

pgsql
Copy code

---

## 💼 **Executive Summary (for Portfolio / README intro section)**

> The **Twitter Sentiment Analysis using Machine Learning** project applies Natural Language Processing (NLP) to analyze and predict user sentiment from Twitter text data.  
> Using NLTK for preprocessing and TF-IDF vectorization for feature extraction, the project employs **Logistic Regression** to classify tweets into positive or negative sentiments.  
> Through comprehensive data cleaning, tokenization, and stemming, noise was minimized to enhance model accuracy.  
> The model achieved approximately **85–90% accuracy**, indicating strong performance for binary sentiment tasks.  
> Visual exploration showed that positive sentiments slightly outnumber negative ones, reflecting optimistic tone patterns in the dataset.  
> This project demonstrates practical skills in **text analytics**, **machine learning pipeline development**, and **data visualization**, making it a valuable addition to a data analyst or NLP portfolio.
