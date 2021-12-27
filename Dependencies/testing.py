import pandas as pd
import os
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import nltk
nltk.download('stopwords')
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pathlib


def vectorize(text, vectorizer):
  """vectorize text using precalculated vectorizer.
  Returns vectored text"""
  return [vectorizer.transform(text)]


def print_accuracy(y_test, predictions):
  print(f'Total accuracy is {accuracy_score(y_test, predictions):.2%}.')
  print('\n')


def show_confusion_matrix(y_test, predictions, model):
  """Display confusion matrix as well as other performance measures."""

  cm = confusion_matrix(y_test, predictions, labels=model.classes_)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
  disp.plot()
  plt.show()

  print('\n')

  tn, fp = cm[0]
  fn, tp = cm[1]

  sensitivity = (tp / (tp + fn)) * 100
  specificity = (tn / (tn + fp)) * 100

  print(f'Sensitivity: {sensitivity:.2f}%')
  print(f'Specificity: {specificity:.2f}%')

  precision = (tp / (tp + fp))
  recall = sensitivity / 100

  f1 = 2 *((precision * recall) / (precision + recall))

  print(f'Precision: {precision:.4f}')
  print(f'Recall: {recall:.4f}')
  print(f'F1 Score: {f1:.4f}')
  

def test_xml_sgm(folder_path, model, vectorizer, drop_na=True):
  """Reads in and parses XML files located within
  folder located at folder_path string.
  
  If drop_na is True (default), then any missing values will be removed before testing."""

  # Checks if folder_path is indeed a string
  if not type(folder_path) is str:
    raise TypeError('folder_path must be a string')

  file_ext = '.sgm'
  os.chdir(pathlib.Path(__file__).parent.resolve())
  import xmltodict
  import my_stop_words as mysw 

  # Looped through all .sgm files in the directory specified.
  xml_string = ''
  for file in os.listdir(folder_path):
    if file.endswith(file_ext):
      with open(f'{folder_path}/{file}', 'rb') as f:
        # Read xml data from current open file and append to xml_string.
        xml_append = f.read().decode('utf-8', 'ignore')
        xml_string = f'{xml_string}{xml_append}'

  # Removed control characters and other special characters.
  # Control characters were causing errors in parsing the xml text.
  xml_string = re.sub(r'&#\d{1,2};', '', xml_string)

  # Removed all 'DOCTYPE' declarations in concatenated string.
  xml_string = re.sub(r'<!DOCTYPE.*>', '', xml_string)

  # Add 'lewis' root tags, from dtd file, in order to parse xml as well as original DOCTYPE declaration.
  xml_string = f'<!DOCTYPE lewis SYSTEM "lewis.dtd"><lewis>{xml_string}</lewis>'

  # Parsed xml string and converted to dictionary.
  xml_dict = xmltodict.parse(xml_string)

  # Dictionary paths to relevant data:
  #
  #   xml_dict['lewis']['REUTERS']
  #     A list containing all the articles.
  #
  #   xml_dict['lewis']['REUTERS'][index]['TOPICS']['D']
  #     This is where the article's topics go. An article can have one, more than one, or no topics.
  #     If an article has not topics, there are no 'D' tags.
  #
  #   xml_dict['lewis']['REUTERS'][index]['TEXT']['BODY']
  #     Contains the text of the article's body.
  #

  # List of all articles
  articles = xml_dict['lewis']['REUTERS']

  # Below, a dataframe was created containing the values of the three attributes listed above.
  #
  # I used the link below to help me solve a bug I was having with the list comprehensions.
  #
  # https://stackoverflow.com/questions/26264359/why-is-this-list-comprehension-giving-me-a-syntax-error
  # 

  df = pd.DataFrame({

      # If there were multiple topics, they were joined together as one string using ',' as the delimiter.
      'topic': [None if article['TOPICS'] is None else ','.join(article['TOPICS']['D']) \
                  if type(article['TOPICS']['D']) is list else article['TOPICS']['D'] for article in articles],
                
      'article_text' : [article['TEXT']['BODY'] if 'BODY' in article['TEXT'] else None for article in articles]
  })

  # Remove all rows with any missing data from df before testing
  if drop_na:
    df = df.dropna()

  # Remove newline characters from the body text
  df['article_text'] = df['article_text'].str.replace(r'\n', ' ')

  # Target variable
  df['earn_topic'] = df['topic'].apply(lambda x: True if 'earn' in x else False)

  # https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f
  #
  # From my github account from a prior project
  # https://github.com/randr000/MyPythonJupyterNotebooks/blob/main/Twitter%20Tweepy%20Data%20Cleaning%20for%20Text%20Mining.ipynb
  #

  #
  # Further article text cleaning
  #

  # Created a copy of the article_text column and cleaned the data in the new copy
  df['cleaned_text'] = df['article_text']

  # Made every letter lowercase
  df['cleaned_text'] = df['cleaned_text'].str.lower()

  # Removed all hyperlinks
  df['cleaned_text'] = df['cleaned_text'].str.replace(r'(https?:/?/?\S+)', '', flags=re.IGNORECASE)

  # Removed any leading and trailing whitespace
  df['cleaned_text'] = df['cleaned_text'].str.strip()

  # Removed all punctuation
  df['cleaned_text'] = df['cleaned_text'].str.replace(r'[^\w\s]', '')

  # Removed any numbers
  df['cleaned_text'] = df['cleaned_text'].str.replace(r'\d', '')

  # Tokenized text
  tokenizer = RegexpTokenizer(r'\w+')
  df['cleaned_text'] = df['cleaned_text'].apply(lambda x: tokenizer.tokenize(x))

  # Removed stop words
  def remove_stop_words(text, stop_words):
    return [w for w in text if w not in stop_words]

  stop = stopwords.words('english')
  df['cleaned_text'] = df['cleaned_text'].apply(remove_stop_words, stop_words=stop)

  # Stemmed all words
  ps = PorterStemmer()

  def stem_words(text):
    return [ps.stem(w) for w in text]

  df['cleaned_text'] = df['cleaned_text'].apply(stem_words)

  # Removed stemmed stop words
  stemmed_stop_words = mysw.my_stop_words
  df['cleaned_text'] = df['cleaned_text'].apply(remove_stop_words, stop_words=stemmed_stop_words)

  # Created a new dataframe with just the data needed for X and y.
  df = df.loc[:, ['earn_topic', 'cleaned_text']]

  # Joined each list of words under cleaned_text as strings in order to use the X data
  # with the TfidfVectorizer package.
  #

  df['cleaned_text'] = [' '.join(words) for words in df['cleaned_text'].values]

  # Vectorized text
  X = vectorizer.transform(df['cleaned_text'])

  df = df.rename(columns={'earn_topic' : 'y'})

  # Predictions
  predictions = model.predict(X)

  # Accuracy
  print_accuracy(df['y'], predictions)

  # Created a confusion matrix to visualize specificity and sensitivity.
  show_confusion_matrix(df['y'], predictions, model)