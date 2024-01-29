# hw1-kovalevich-olga-204
print ('hello, world!')
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
file_path = 'input.txt'
with open(file_path, 'r') as file:
text_content = file.read()
import re
def clean_text(text):
text = text.lower()
text = re.sub(r'[^a-z\s]', '', text)
text = re.sub(r'\s+', ' ', text).strip()
return text
cleaned_text = clean_text(text_content)
cleaned_text[:100]
import nltk
from nltk.tokenize import word_tokeniz
nltk.download('punkt')
tokens = word_tokenize(cleaned_text)
tokens[:10]
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
text = "first citizen before we proceed any further hear me speak all speak speak first citizen you are all"
tokens = word_tokenize(text)
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("Наш текст:", text)
print("Токены без стоп-слов:", filtered_tokens)
from gensim.models import Word2Vec
model = Word2Vec(sentences=[tokens], vector_size=100, window=5, min_count=1, workers=4)
model.wv['citizen']
model.wv.most_similar(["citizen"])
model.wv.similarity('citizen', 'alien')
model.wv.similarity('citizen', 'alien')
model.wv.doesnt_match(['citizen', 'resident', 'alien'])
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
from sklearn.decomposition import PCA
def pca_scatterplot(model, words=None, sample=0):
word_vectors = [model.wv[w] for w in words]
vectors_2d = PCA().fit_transform(word_vectors)
plt.figure(figsize=(12,10))
plt.scatter(vectors_2d[:,0], vectors_2d[:,1], c='g')
for i, word in enumerate(words):
plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]))
pca_scatterplot(model, ['greasy', 'mutinous', 'poor', 'good', 'loving', 'several', 'kind', 'gravest', 'generous', 'gentle', 'ancient'])
