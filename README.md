# 📩 Spam-Ham Classification Project

This project demonstrates a **Spam-Ham text classification** pipeline using Natural Language Processing (NLP) techniques such as **Bag of Words (BoW)** and **TF-IDF**, followed by **Naive Bayes** modeling. The dataset used is the classic **SMSSpamCollection**, which contains labeled SMS messages as spam or ham.

---

## 🧠 Features
- Data cleaning and text preprocessing
- Feature extraction using CountVectorizer and TfidfVectorizer
- Model training with Multinomial Naive Bayes
- Performance evaluation using accuracy, confusion matrix, and classification report

---

## 🗂️ Dataset

The dataset used is `SMSSpamCollection.txt`, which contains:
- `label`: either `spam` or `ham`
- `message`: the actual SMS content

Example:
```
ham     Go until jurong point, crazy.. Available only in bugis n great world la e buffet...
spam    Free entry in 2 a wkly comp to win FA Cup...
```

---

## 🧹 Text Preprocessing

Steps involved:
- Removal of punctuation and non-alphabetic characters
- Conversion to lowercase
- Removal of stopwords using NLTK
- Tokenization and optional stemming/lemmatization

```python
import re
from nltk.corpus import stopwords

# Example cleanup
text = re.sub('[^a-zA-Z]', ' ', text)
text = text.lower()
tokens = [word for word in text.split() if word not in stopwords.words('english')]
```

---

## 🧾 Feature Extraction

Two vectorization techniques are implemented:

1. **Bag of Words (BoW) using CountVectorizer**
```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus)
```

2. **TF-IDF (Term Frequency–Inverse Document Frequency)**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus)
```

---

## 🧪 Model Training

The model used is **Multinomial Naive Bayes**, suitable for text classification tasks:

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = MultinomialNB()
model.fit(X_train, y_train)
```

---

## 📊 Evaluation

Model performance is evaluated using:
- Accuracy Score
- Confusion Matrix
- Classification Report

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## 🛠️ Requirements

Install dependencies using:

```bash
pip install pandas scikit-learn nltk
```

Also run once in your notebook:
```python
import nltk
nltk.download('stopwords')
```

---

## 📁 Project Structure

```
span-ham-classification-project.ipynb
SMSSpamCollection.txt
```

---

## 🚀 Future Improvements

- Use deep learning models like LSTM or BERT
- Add UI for SMS classification
- Integrate live API for spam detection
