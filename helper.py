import pandas as pd
import numpy as np
import json
import datetime
from bs4 import BeautifulSoup as soup

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, 
f1_score, accuracy_score, precision_score, recall_score)

import matplotlib.pyplot as plt

def predict(model, X, threshold=0.5):
    '''Return prediction of the fitted binary-classifier model model on X using
    the specifed `threshold`. NB: class 0 is the positive class'''
    return np.where(model.predict_proba(X)[:, 1] > threshold,
                    model.classes_[1],
                    model.classes_[0])

def confusion_matrix1(model, X, y, threshold=0.5):
    cf = pd.crosstab(y, predict(model, X, threshold))
    cf = cf.add(pd.DataFrame([[0,0],[0,0]]))
    cf.index.name = 'actual'
    cf.columns.name = 'predicted'
    return cf

def calculate_threshold_values(prob, y):
    '''
    Build dataframe of the various confusion-matrix ratios by threshold
    from a list of predicted probabilities and actual y values
    '''
    df = pd.DataFrame({'prob': prob, 'y': y})
    df.sort_values('prob', inplace=True)
    
    actual_p = df.y.sum()
    actual_n = df.shape[0] - df.y.sum()

    df['tn'] = (df.y == 0).cumsum()
    df['fn'] = df.y.cumsum()
    df['fp'] = actual_n - df.tn
    df['tp'] = actual_p - df.fn

    df['fpr'] = df.fp/(df.fp + df.tn)
    df['tpr'] = df.tp/(df.tp + df.fn)
    df['precision'] = df.tp/(df.tp + df.fp)
    df = df.reset_index(drop=True)
    return df

def score_table(model, y_true,y_pred):
    print(type(model).__name_)
    print(f'Accuracy Score: {}'.format(accuracy_score(y_true,y_pred)))
    print(f'Precision Score: {}'.format(precision_score(y_true,y_pred)))
    print(f'F1 Score: {}'.format(f1_score(y_true,y_pred)))
    print(f'Recall Score: {}'.format(recall_score(y_true,y_pred)))

def plot_precision_recall(ax, df,random=False):
    ax.plot(df.tpr,df.precision, label='precision/recall')
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('Precision/Recall Curve')
    if random:
        ax.plot([0,1],[df.precision[0],df.precision[0]], 'k', label='random')
    ax.set_xlim(xmin=0,xmax=1)
    ax.set_ylim(ymin=0,ymax=1)
    ax.legend()

def plot_roc(ax, df, model, random=False):
    auc = round(roc_auc_score(df['y'].to_numpy(), df['prob'].to_numpy()),4)
    label = type(model).__name__ + ' (AUC: '+ str(auc) + ')'
    ax.plot([1]+list(df.fpr), [1]+list(df.tpr), label=label)
    if random:
        ax.plot([0,1],[0,1], 'k', label="random")
    ax.set_xlabel('fpr')
    ax.set_ylabel('tpr')
    ax.set_title('ROC Curve')
    ax.legend()


# NLP FUNCTION
import string
import unicodedata
import nltk

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams
from nltk import pos_tag
from nltk import RegexpParser



def extract_bow_from_raw_text(text_as_string):
    """Extracts bag-of-words from a raw text string.

    Parameters
    ----------
    text (str): a text document given as a string

    Returns
    -------
    list : the list of the tokens extracted and filtered from the text
    """
    if (text_as_string == None):
        return []

    if (len(text_as_string) < 1):
        return []

    nfkd_form = unicodedata.normalize('NFKD', text_as_string)
    text_input = str(nfkd_form.encode('ASCII', 'ignore'))

    sent_tokens = sent_tokenize(text_input)

    tokens = list(map(word_tokenize, sent_tokens))

    sent_tags = list(map(pos_tag, tokens))

    grammar = r"""
        SENT: {<(J|N).*>}                # chunk sequences of proper nouns
    """

    cp = RegexpParser(grammar)
    ret_tokens = list()
    stemmer_snowball = SnowballStemmer('english')

    for sent in sent_tags:
        tree = cp.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == 'SENT':
                t_tokenlist = [tpos[0].lower() for tpos in subtree.leaves()]
                t_tokens_stemsnowball = list(map(stemmer_snowball.stem, t_tokenlist))
                #t_token = "-".join(t_tokens_stemsnowball)
                #ret_tokens.append(t_token)
                ret_tokens.extend(t_tokens_stemsnowball)
            #if subtree.label() == 'V2V': print(subtree)
    #tokens_lower = [map(string.lower, sent) for sent in tokens]

    return(ret_tokens)
    