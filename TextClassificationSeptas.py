import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import streamlit as st
import plotly.express as px
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
print(pd)

st.set_page_config(page_title='TextClassificationSeptas')
st.title("!! TEXT CLASSIFICATION FOR FREE !!")
st.write("Disclaimer : This Website never save your data! only put your google drive data key !!")

apiOneDrive="https://onedrive.live.com/download?cid="
apiGoogleDrive="https://drive.google.com/uc?export=download&id="

with st.expander("DATA PREPARATION : ", expanded=True):
    with st.form("UPLOAD YOUR DATA TRAINING :"):
        st.write("Ensure your data ONLY HAVE 2 COLUMNS WITH HEADER NAMES IS : Description & Category ")
        sources=st.selectbox("Select Source : ",["Google Drive","One Drive"])
        if sources=="Google Drive":
            apiKey=apiGoogleDrive
        else:
            apiKey=apiOneDrive            
        keyDriveData = st.text_input("Type your " + sources +" data key :")
        submittedData = st.form_submit_button("Submit")
        
        if keyDriveData == "" :
            st.stop()
            if submittedData:
                st.write("Start Processing..")
            
@st.cache(suppress_st_warning=True)
def read_df():
    masDat=pd.read_csv(apiKey + keyDriveData,sep=";",engine="python",encoding = "ISO-8859-1")
    masDat=masDat.apply(lambda x: x.astype(str).str.upper())
    return masDat[~masDat['Category'].isin(['LAND',
                                            'BUILDING IMPROVEMENT',
                                            'MACHINERY',
                                            'FACTORY EQUIPMENT'])]
masDat=read_df()

st.success('Read Data Success')

# PLOT DESCRIPTION
@st.cache(suppress_st_warning=True)
def gb_Data():
    return masDat.groupby(['Category'])['Description'].count().reset_index()
gbDat=gb_Data()

pltDat=px.bar(gbDat.sort_values(by="Description",
                               ascending=False)
              ,x="Category",
               y="Description",
               title="PROFILE DATA",
               height=500)

st.success('Plotting Description Success')

# FACTORIZE
@st.cache(suppress_st_warning=True)
def fact_Data():
    factDat = masDat[pd.notnull(masDat['Description'])]
    factDat.columns = ['Description', 'Category']
    factDat['category_id'] = factDat['Category'].factorize()[0]
    return factDat
factDat=fact_Data()

#SET CATEGORY ID
category_id_df = factDat[['Category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Category']].values)
    
#FEATURES EXTRACTION
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(factDat['Description']).toarray()
labels = factDat.category_id

# FEATURES SELECTION
from sklearn.feature_selection import chi2
N = 2
for Product, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    #print("# '{}':".format(Product))
    #print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
    #print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

st.success('Preparation Models Success')
    
#SPLIT DATA TRAINING & TESTING
X_train, X_test, y_train, y_test = train_test_split(factDat['Description'], factDat['Category'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

st.success('Data Training Testing Split Success')

# FIT MODELS NAIVE BAYES DKK
clf_NaiveBayes = MultinomialNB().fit(X_train_tfidf, y_train)
clf_LinearSVC = LinearSVC().fit(X_train_tfidf, y_train)
clf_LogisticRegression = LogisticRegression().fit(X_train_tfidf, y_train)
clf_RandomForest = RandomForestClassifier().fit(X_train_tfidf, y_train)
st.success('Fit Models Success')

# GENERATE EVALUATION MODEL
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

st.success('Data Modelling Success')

# PLOTING EVALUATION MODEL
bxPlotEvMod = px.box(cv_df, x='model_name', y='accuracy', points="all", color="model_name")

#HEATMAPS
model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, factDat.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df['Category'], yticklabels=category_id_df['Category'])
plt.ylabel('Actual')
plt.xlabel('Predicted')


# MODELS ACCURATION
modelAccuracy = cv_df.groupby('model_name').accuracy.mean().reset_index().sort_values(by="accuracy",ascending=False)
bestModels=modelAccuracy["model_name"][0]

pltModAccuracy=px.bar(modelAccuracy,
                   x="model_name",
                   y="accuracy",
                   title="ACCURATION MODELS",
                   barmode="group",
                   text_auto=True)

st.success('Setup Accuration Success')

# VISUALIZAITON STREAMLIT
with st.expander("DATA PROFILE : "):
    col1,col2=st.columns(2)
    col1.write("Table Data")
    col1.dataframe(masDat.head(10))
    col2.write("Data Summary")
    col2.dataframe(gbDat)
    st.plotly_chart(pltDat)
    
with st.expander("DATA MODELING :"):
    st.write("Factorize at Category_ID")
    st.dataframe(factDat)
    st.write("Boxplot Models")
    st.plotly_chart(bxPlotEvMod)
    st.write("Heatmaps")
    st.pyplot(fig)
    st.plotly_chart(pltModAccuracy)
    
# TEXT PREDICTION
with st.expander("PREDICTION FROM TEXT :"):
    with st.form("PREDICT TEXT"):
        textToBePred = st.text_input("Type your text to predicted....")
        # Every form must have a submit button.
        submittedText = st.form_submit_button("Predict Text")
        if submittedText:
            predictedTextNB=clf_NaiveBayes.predict(count_vect.transform([textToBePred.upper()]))
            predictedTextLSVC=clf_LinearSVC.predict(count_vect.transform([textToBePred.upper()]))
            predictedTextLReg=clf_LogisticRegression.predict(count_vect.transform([textToBePred.upper()]))
            predictedTextRandFor=clf_RandomForest.predict(count_vect.transform([textToBePred.upper()]))
            st.subheader("Prediction Text Result")
            st.subheader("Best Models : " + bestModels)
            
            st.write("Naive Bayes Prediction Result         : " + predictedTextNB[0])
            st.write("Linear SVC Prediction Result          : " + predictedTextLSVC[0])
            st.write("Logistic Regression Prediction Result : " + predictedTextLReg[0])
            st.write("Random Forest Prediction Result       : " + predictedTextRandFor[0])

# DATA PREDICTION
with st.expander("PREDICTION FROM DATA :"):
    with st.form("PREDICT DATA"):
        # LOAD DATA TO PREDICTION
        sourcesPred=st.selectbox("Select Source : ",["Google Drive","One Drive"])
        if sourcesPred=="Google Drive":
            apiKeyPred=apiGoogleDrive
        else:
            apiKeyPred=apiOneDrive
        
        textToBePred = st.text_input("Type your google/onedrive data key to predicted....")
        
        submittedData = st.form_submit_button("Predict Data")
        if submittedData:
            @st.cache(suppress_st_warning=True)
            def read_pred():
                return pd.read_csv(apiKeyPred+textToBePred,sep=";",engine="python",encoding = "ISO-8859-1")
            dataToPred=read_pred()
            dataToPred=dataToPred.apply(lambda x: x.astype(str).str.upper())
            
            # DATA PREDICTION
            listDescPred=dataToPred['Description'].tolist()
            for f in listDescPred:
                dataToPred.loc[(dataToPred['Description'] == f ),'MultinomialNB'] = str(clf_NaiveBayes.predict(count_vect.transform([f])))
                dataToPred.loc[(dataToPred['Description'] == f ),'LinearSVC'] = str(clf_LinearSVC.predict(count_vect.transform([f])))
                dataToPred.loc[(dataToPred['Description'] == f ),'LogisticRegression'] = str(clf_LogisticRegression.predict(count_vect.transform([f])))
                dataToPred.loc[(dataToPred['Description'] == f ),'RandomForestClassifier'] = str(clf_RandomForest.predict(count_vect.transform([f])))
            st.subheader("Prediction Data Result")
            st.subheader("Best Models : " + bestModels)
            st.dataframe(dataToPred)
