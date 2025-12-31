import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
# Note: The UCI dataset is often inside a zip; this URL points to a raw version 
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
df = pd.read_table(url, header=None, names=['label', 'message'])

# Preview data
print(df.head())
print(df.info())

# Convert label to numerical variable
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# Define features (X) and target (y)
X = df.message
y = df.label_num

# Split into Training and Testing sets (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Instantiate the vectorizer
vect = CountVectorizer()

# Learn the vocabulary and create the document-term matrix (DTM)
X_train_dtm = vect.fit_transform(X_train)

# Transform testing data (using fitted vocabulary)
X_test_dtm = vect.transform(X_test)

print(f"Number of unique words (features): {X_train_dtm.shape[1]}")

# Instantiate the model
nb = MultinomialNB()

# Train the model
nb.fit(X_train_dtm, y_train)

# Make predictions on the test set
y_pred_class = nb.predict(X_test_dtm)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Accuracy Score: {accuracy:.4f}")

# Print Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Detailed Classification Report
print(classification_report(y_test, y_pred_class))

def predict_spam(new_message):
    new_message_dtm = vect.transform([new_message])
    prediction = nb.predict(new_message_dtm)
    return "SPAM" if prediction[0] == 1 else "HAM"

# Test cases
print(predict_spam("Hey, are we still meeting for lunch at 12?"))
print(predict_spam("WINNER! You have won a $1000 Walmart gift card. Click here to claim now."))
print(predict_spam("Urgent: Your account has been locked. Call 08001234 now."))
