#Imports and Settings
import pandas as pd
import numpy as np
import swifter
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime
from operator import mul
import seaborn as sn
import random
import pickle

#Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, 
f1_score, accuracy_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler

#Helper Functions
from helper import predict, confusion_matrix1, calculate_threshold_values, plot_roc, seeks_run, format_assign
#Pandas Settings to Display Rows and Cols
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', 10) 

#Matplotlib Style Settings
plt.style.use('fivethirtyeight')
params = {'legend.fontsize': 'large',
          'figure.figsize': (15, 10),
         'axes.labelsize': 'large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
plt.rcParams.update(params)

import warnings
warnings.filterwarnings("ignore")

from tqdm.auto import tqdm
tqdm.pandas(desc="my bar!")

class MyModel():
    def __init__(self):
        self.model = GradientBoostingClassifier(learning_rate=0.05, 
                                      max_depth=6, 
                                      min_samples_leaf=6,
                                      n_estimators=200,
                                      random_state=1,
                                      verbose=True)

    def fit(self, X, y):
        self.model.fit(X,y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

if __name__ == '__main__':

    '''
    Looping through data folder to concat CSV's into one big dataframe
    1002400 rows in dataset
    63 columns
    Reinitialize HERE
    '''
    for i,name in tqdm(enumerate(glob.glob('data/CSV/*'))):
        if i == 0:
            df = pd.read_csv(name)
        df2 = pd.read_csv(name)
        concat = pd.concat([df,df2],ignore_index=True)
        df = concat

    #Trimming off extraneous columns
    df = df.iloc[:,1:57]
    df = df.drop(['Classics_Request_ID','End_Of_Run','Play_Dates','Start_Date','Local','Rental_Measure','Boxoffice_Net',
                'Film_Rental','To_Receive','FRM_Perc','Min_Max','MG_FF','Price','Customer_Reference','Office',
                'Remark','Payer_Remark','DType','Aspect_Ratio','Sound_System','Seat_Type','Amenity','Frame_Rate',
                'Resolution','Projection_Type','Licensor','L','Hire','Rem_','Rem_Date','Media_Type','Dim_','LV','Circuit'],axis=1)

    #Drop any duplicate ID's
    df.drop_duplicates('LA_No',inplace=True)

    #Dropping any non SP payments
    df = df[df['FRM']=='sp'] 

    # Only looking at records with grosses returned
    df = df[df['Status']=='Returns In']

    df = df[df.Division != 'CLASSICS']

    #Dropping 500 NAN media formats
    df = df.dropna(axis=0)

    #Not in YT Data
    df = df[df.Release !='THEY SHALL NOT GROW OLD']
    df = df[df.Release !='FANTASTIC BEASTS AND WHERE TO FIND THEM']

    # Setting field to DT
    df['First_Date'] = pd.to_datetime(df['First_Date'])
    df['Last_Date'] = pd.to_datetime(df['Last_Date'])

    # Setting fields to numeric
    df['Boxoffice'] = df['Boxoffice'].astype(str)
    df['Boxoffice'] = df['Boxoffice'].apply(lambda x: x.replace(',', ''))
    df['Boxoffice'] = df['Boxoffice'].astype(float)
    df['T_No'] = df['T_No'].astype(float)

    # Removed Admissions because of data leakage
    df['Admissions'] = df['Admissions'].astype(str)
    df['Admissions'] = df['Admissions'].apply(lambda x: x.replace(',', ''))
    df['Admissions'] = df['Admissions'].astype(float)

    df['Media_Formats'] = df['Media_Formats'].astype(str)
    df['Media_Formats'] = df['Media_Formats'].swifter.apply(format_assign)
    df['Seeks_Run'] = df['Seeks_Run'].swifter.apply(seeks_run)

    # Anonymizing Theatre Information
    from faker import Faker
    '''
    Number of values to annoymize
    6022 Theatres
    4661 Film Buyer/Buying Circuit/Payer
    '''
    # Creating list of unique values to anonymize
    confid_users = list(df.Payer.unique())
    confid_users.extend(list(df.Buying_Circuit.unique()))
    confid_users.extend(list(df.Film_Buyer.unique()))
    distinct_confid = set(confid_users)

    # Creating Fake name list 
    Faker.seed(2)
    fake = Faker()
    fake_names = []
    for _ in tqdm(range(5300)):
        fake_names.append(fake.company())
    fake_names = set(fake_names)
    fake_names = list(fake_names)

    # Applying fake name list to dictionary of unique data
    confid_dict = dict.fromkeys(distinct_confid)
    for key,value in tqdm(confid_dict.items()):
        confid_dict[key] = random.sample(fake_names,1)[0]

    #List of Cinema Names to generate random values from
    cinema_names = pd.read_csv('data/extra_data/Cinema_names.csv',header=None)
    cinema_list = []
    for n in tqdm(range(0,674)):
        if len(cinema_names[0].apply(lambda x: x.split(' '))[n])>1:
            cinema_list.append(cinema_names[0].apply(lambda x: x.split(' '))[n][1])
        else:
            continue

    #Dictionary of theatres from data source
    suffixes = ['Theaters','Cinemas','Movies','Flicks','Screens','Forum','Marquee']
    maccs_theatres = df['Theatre'].unique().tolist()
    maccs_theatres_dict = dict.fromkeys(maccs_theatres)

    #Applying Fakes Names to Dictionary
    for key,value in maccs_theatres_dict.items():
        maccs_theatres_dict[key] = random.choice(cinema_list) +' '+ random.choice(cinema_list) + ' '+ random.choice(suffixes) 

    #Replacing Values in Data Frame
    df['Theatre'].replace(maccs_theatres_dict,inplace=True)
    df['Buying_Circuit'].replace(confid_dict,inplace=True)
    df['Film_Buyer'].replace(confid_dict,inplace=True)
    df['Payer'].replace(confid_dict,inplace=True)

    ## Creating Opening Weekend Data Frame to build model from

    #Only opening week
    ow_df = df[df['Seeks_Run']=='FIRST RUN'].sort_values(['Theatre','Release','T_No'])
    ow_df = ow_df[ow_df['T_No']==1]
    ow_df.reset_index(inplace=True)
    ow_df.drop(['index','FRM','LA_No','Status','Last_Date','Terms_Perc','Branch','Area'],axis=1,inplace=True)

    #Total Opening Week Numbers
    ow_bo = ow_df.groupby('Release').sum()['Boxoffice']
    ow_bo = ow_bo.astype(int)
    ow_bo = ow_bo.reset_index()
    ow_bo['Boxoffice_Total'] = ow_bo['Boxoffice']
    ow_bo.drop('Boxoffice',axis=1,inplace=True)

    #Opening Weekend Data Frame
    ow_data = ow_df.merge(ow_bo,how='left',left_on='Release',right_on='Release')

    #Bringing in Movie Meta Data
    wb_list = df['Release'].unique().tolist()
    clean = []
    for title in wb_list:
        save = title.split("(")
        clean.append(save[0])

    import difflib
    import itertools
    def get_close_matches_icase(word, possibilities, *args, **kwargs):
        """ Case-insensitive version of difflib.get_close_matches to help match Title fields"""
        lword = word.lower()
        lpos = {}
        for p in possibilities:
            if p.lower() not in lpos:
                lpos[p.lower()] = [p]
            else:
                lpos[p.lower()].append(p)
        lmatches = difflib.get_close_matches(lword, lpos.keys(), *args, **kwargs)
        ret = [lpos[m] for m in lmatches]
        ret = itertools.chain.from_iterable(ret)
        return set(ret)

    #Genre Dataframe
    genre_df = pd.read_csv('data/extra_data/wb_df.csv')

    #Match Titles
    genre_df['title'] = genre_df['title'].apply(lambda x: str(get_close_matches_icase(x,wb_list,n=1,cutoff=0.4)))
    genre_df['title'] = genre_df['title'].apply(lambda x: x.replace("{'",''))
    genre_df['title'] = genre_df['title'].apply(lambda x: x.replace("'}",''))

    #Matching between TMDb and WB
    genre_df.at[9,'title'] = "HOUSE, THE"
    genre_df.at[29,'title'] = "MEG, THE"
    genre_df.at[31,'title'] = "NUN, THE"
    genre_df.at[35,'title'] = "MULE, THE"
    genre_df.at[44,'title'] = "SHAFT (2074539)"
    genre_df = genre_df[genre_df.title !='They Shall Not Grow Old']
    genre_df['title'] = genre_df['title'].apply(lambda x: str(get_close_matches_icase(x,wb_list,n=1,cutoff=0.4)))
    genre_df['title'] = genre_df['title'].apply(lambda x: x.replace("{'",''))
    genre_df['title'] = genre_df['title'].apply(lambda x: x.replace("'}",''))
    genre_df.at[12,'title'] = "IT (2017)"
    genre_df.at[38,'title'] = "ISN'T IT ROMANTIC"
    genre_df.at[27,'title'] = "OCEAN'S 8"

    genre_df = genre_df.drop(['genre_ids','Unnamed: 0','adult','id','original_language','popularity','video','vote_average','vote_count'],axis=1)

    # Adding OW field
    genre_df = genre_df.merge(ow_bo,how='left',left_on='title',right_on='Release')

    # Adding Comp Library Fields
    comp_library = pd.read_csv('data/extra_data/Comparison Library - Test.xlsx - Database.csv')
    comp_library = comp_library[['TITLE','Dist','# of Runs','Genre','Rating','Season','WIDE                --------              Open Date']]
    comp_library = comp_library[comp_library['Dist']=='WB']

    #Filtering out Data
    comp_library['release_date'] = pd.to_datetime(comp_library['WIDE                --------              Open Date'])
    comp_library.drop('WIDE                --------              Open Date',axis=1,inplace=True)
    comp_library['year'] = comp_library['release_date'].dt.year
    comp_library = comp_library[comp_library['year']>=2017]
    comp_library = comp_library[comp_library['year']<2020]

    #Matching Titles
    comp_library['title'] = comp_library['TITLE'].apply(lambda x: str(get_close_matches_icase(x,wb_list,n=1,cutoff=0.3)))
    comp_library['title'] = comp_library['title'].apply(lambda x: x.replace("{'",''))
    comp_library['title'] = comp_library['title'].apply(lambda x: x.replace("'}",''))

    comp_library.drop(index=[90,124,132,133,231,254,259,269,279,291,300,406,421,450,638],axis=0,inplace=True)
    comp_library.at[188,'TITLE'] = 'SHAFT (2074539)'
    comp_library.at[188,'title'] = 'SHAFT (2074539)'
    comp_library.at[248,'title'] = "ISN'T IT ROMANTIC"
    comp_library.at[396,'title'] = "OCEAN'S 8"

    #Combine both Data Sources
    genre_df = comp_library.merge(genre_df,how='right',left_on='title',right_on='title',suffixes=('_x','_y'))

    #Dropping TSNGO from Movie Description
    genre_df.drop(36,axis=0,inplace=True)
    genre_df.reset_index(inplace=True)
    genre_df.drop('index',axis=1,inplace=True)
    genre_df['# of Runs'] = genre_df['# of Runs'].apply(lambda x: x.replace(',',''))
    genre_df['# of Runs'] = genre_df['# of Runs'].astype(int)

    #Save Overview for NLP
    movie_text = genre_df['overview']

    #Manual Corrections
    genre_df.drop(['overview','release_date_x','release_date_y','Release','revenue'],axis=1,inplace=True)
    genre_df.at[32,'budget'] = 80000000 #Smallfoot
    genre_df.at[41,'budget'] = 9000000 #Sun is also a Star
    genre_df.at[46,'budget'] = 15000000 #Blinded by the Light

    #Creating Theatre by Genre DF, built from OW Data and Genre DF
    theatre_genre = ow_data.copy()
    theatre_genre = theatre_genre.merge(genre_df[['title','Genre']], how='left',left_on='Release',right_on='title')
    theatre_genre = round(theatre_genre.groupby(['Theatre','Genre'],as_index=False)['Boxoffice'].mean())

    #Combining with OW data
    ow_locavg = ow_data.merge(genre_df[['title','Genre']],how='left',left_on='Release',right_on='title')
    ow_locavg = ow_locavg.merge(theatre_genre,how='left',on=['Theatre','Genre'],suffixes=['_loc','_avg'])
    ow_locavg['over_index'] = ow_locavg['Boxoffice_loc']-ow_locavg['Boxoffice_avg']
    ow_locavg['over_index'] = ow_locavg['over_index'].apply(lambda x: 1 if x > 0 else 0)

    #Adding YT Data Source
    yt_views = pd.read_csv('data/extra_data/OW to YT - BoxOfficeReport - WB 2017-2019.csv')
    yt_views['Release Date'] = pd.to_datetime(yt_views['Release Date'])
    yt_views = yt_views[yt_views['Release Date']<'2020']

    #Dropping Live by Night rom 2016
    yt_views.drop(63,axis=0,inplace=True)

    yt_views['Film (Distributor)'] =yt_views['Film (Distributor)'].apply(lambda x: x.split('(')[0])
    yt_views['Film (Distributor)'] = yt_views['Film (Distributor)'].apply(lambda x: str(get_close_matches_icase(x,wb_list,n=1,cutoff=0.5)))
    yt_views['Film (Distributor)'] = yt_views['Film (Distributor)'].apply(lambda x: x.replace("{'",''))
    yt_views['Film (Distributor)'] = yt_views['Film (Distributor)'].apply(lambda x: x.replace("'}",''))

    #Manual Corrections
    yt_views.at[25,'Film (Distributor)'] = "ISN'T IT ROMANTIC"
    yt_views.at[36,'Film (Distributor)'] = "OCEAN'S 8"
    yt_views.at[53,'Film (Distributor)'] = "HOUSE, THE (2017)"

    #Full Movie Meta Data Source
    full_movie = genre_df.merge(yt_views,how='left',left_on='title',right_on='Film (Distributor)')
    full_movie.drop(['Dist','TITLE','Release Date','Trailer Link','title','YT Trailer Views','Opening Weekend','year'],axis=1,inplace=True)

    #Combination of Movie MetaData and OW
    iter4 = ow_locavg[['Media_Formats','Release','Division','Genre','over_index']]
    iter4 = iter4.merge(full_movie[['Film (Distributor)','# of Runs','Rating','Season','budget','runtime','OW to YT']],how='left',left_on='Release',right_on='Film (Distributor)')
    iter4_1hot = pd.get_dummies(data=iter4,columns=['Media_Formats','Division','Genre','Rating','Season'],drop_first=True)
    iter4_1hot.drop(['Release','Film (Distributor)'],axis=1,inplace=True)

    y = iter4_1hot['over_index']
    X = iter4_1hot.drop(columns=['over_index'])

    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) #No split because it will be the entire training set?
    model = MyModel()
    model.fit(X, y)

    with open('model.pkl', 'wb') as f:
        # Write the model to a file.
        pickle.dump(model, f)