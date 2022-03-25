import streamlit as st
# Data wrangling and data visualistion 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np

# Processing text
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

# Machine learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm
import pickle
from sklearn.naive_bayes import GaussianNB


siteHeader = st.container()
# Será el contenedor del título
with siteHeader: 
    st.title("Clasificador de Tweets")
    st.markdown(""" El objetivo de este proyecto es predecir la postura de los 
                usuarios de Twitter en español sobre la vacunación contra el COVID-19. """)

# Empezamos con el modelo
newFeautures = st.container()
with newFeautures: 
    st.header(" Base Inicial")
    st.markdown(""" Demos un vistazo al dataset: """)

# Cargamos la base y damos un preview de la info
df = pd.read_csv('https://raw.githubusercontent.com/patriciacolemus/ClasificacionTweets/main/tweets_2cats.csv')
df.drop(['Unnamed: 0'], axis = 1, inplace= True)

st.write(df.sample(5))
st.markdown(""" Construiremos nuestro modelo con base en la columna **Tipo**. """)

with open("best_rfc.pickle", 'rb') as pfile:  
    model=pickle.load(pfile)

# Interacción con el usuario
finalFeatures = st.container()
with finalFeatures: 
    st.header("¡Prueba el modelo!")

prueba = st.text_area('Ingresa un mensaje para clasificar:','Ingreso el texto aqui')

# Limpieza del texto
prueba = prueba.replace("\r", " ")
prueba = prueba.replace("\n", " ")
prueba = prueba.replace("    ", " ")
prueba = prueba.lower() # Lowercasing the text
prueba = prueba.replace(r"http\S+", "") # Remove links
punctuation_signs = list("¿?:!¡@.,#_-;")
for punct_sign in punctuation_signs:
    prueba = prueba.replace(punct_sign, '')
stop = stopwords.words('spanish')
my_stop = ['vacuna', 'vacunar', 'vacunación', 'vacunas', 'vacunados', 'covid-19', 'coronavirus', 'covid', 'corona', 'buenas', 'https', 'tco', 'https tco', 'covid19', 'COVID19', 'covid 19', 'hoy']
#prueba = prueba.apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))
#prueba = prueba.apply(lambda x: ' '.join([item for item in x.split() if item not in my_stop]))
print(prueba)
               
with open("tfidf.pickle", 'rb') as pfile:  
    vector=pickle.load(pfile)

features_train = vector.transform(pd.Series(prueba))

# Resultados
predictions = model.predict(features_train)


# Output para el usuario    
st.text('Resultado ...')
st.write(predictions)

# Estilo de lo que vamos a ejecutar lo que teniamos allá arriba:
st.markdown( """ <style>
 .main {
 background-color: #AF9EC;
}
</style>""", unsafe_allow_html=True )