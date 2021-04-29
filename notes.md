Day 1:
Extracted booking information from 2017-2019 films of all US theatres
Concatenate Data Sources
Found YT views before Opening Weekend

Day 2:
Kmeans clustering to omit payers/reduce features
one-hot encode all theatres, buying circuits, payer
present table of means/std/frequency of all clusters
box plot of patterns

from there, then you can add the category 

* Remove Title for clustering/dimension reduction once meta data is added in
PCA works here too
Each PCA captures an amount of variance
For each feature, how much is it correlated (eigen value)
Can Cluster from here
Each observation will have a value for PCA

* Limit to OW, and add YT data
* Create % of BO
* standardize data
* ?data leakage if including data in theatre average
* ?splitting data, do i need to pull out the latest movie for each genre/cluster
* can perform a correlation matrix with new feature (total average) to see if its relevant
* dummy classifier
* find an average percent of bo to establish baseline metric
* exclude current movie from genre specific average

DAY 4:
PERFORM PERSONAL GROUPING AND REDUCE DIMENSIONS, DO NOT RUN WITH ONE HOT ENCODED ON LOCAL MACHINE
RUN ONE HOT ENCODED DATA ON EC2
DECIDE ON METRIC (PRECISION OR RECALL) THINK ABOUT THIS
UPDATE OVER INDEX METRIC FOR GENRE/CLUSTER SPECIFIC
RUN ON IMBALANCED DATA
EDA: PCA ON TFIDF ON PLOT SUMMARY
MERGE TFIDF MATRIX ONTO FULL DATASET
SORT PREDICT PROBA ON VALIDATION TEST SET (LATEST LIST OF THEATRE DP)
CLUSTER THEATRE DATA WITH GENRE SPECIFIC AVERAGES/PERC OF BO AS COLUMNS 

DAY 5:
* Iterations:
    * Baseline - Feature Selection: Media Format, Division, Movie Features, OWtoYT, OVER INDEXING ON AVERAGE % OF BO OF ENTIRE WEEK 1 DATASET
        * EDA: Show correlation matrix of over-index field to these features
        * Scoring:  F1, Precision, Recall for all models (Logistic Regression, DT, RF, GB)
    * (IF TIME) Try Running 1Hot encoding of all features
    * Cluster Movie Data - Clustering on Genre Information
        * Score models with Baseline Features + Movie Genre Clusters
        * (!) Performed clustering, still not agreeing with Clusters based on Movie Genre Data, after condensing Genres, adding Budget/Runtime/OWtoYT Ratios/OW Ratios
    * Perform NLP on Movie Description - Cluster along Plot Overview
        * PCA plot of 2 latent features and visualize clustering
        * Score Models with Baseline Feauters + Movie Genre Clusters + Overview Clusters
        * (!) PCA is not able to capture 2 latent features
    * Cluster along Theatre Geographic Information + Genre Specific Averages
        * Score Models with Baseline Feauters + Movie Genre Clusters + Overview Clusters + Geographic Theatre Clustering
    * Adjust Over Index Average to Genre Average at Location
        * Score Models

* Look into Flask webpage (ask TOMAS)
    * Pickle Model
    * Deploy Pickle'd model
    * Take user input of: Movie Cluster + Budget + Runtime = Appended to latest 4,000 theatres in DP
    * Model and predict with new test set
    * Return theatres and sort by predict proba

DAY 6 (SUNDAY):
* Tried NLP Clustering, Genre Clustering + Rating, Budget, runtime, still not super satisfied with Clusterings
* Need to create a DF of Theatres, with Genre Types, and Avg Sum of films for that Genre, and Establish Over_Index threshhold
    - Calcuate baseline average a theatre makes for ANY GENRE
    - Calcualte average a theatre makes PER GENRE
    - Identify if a theatre Over_Indexes on a particular Genre
* Establish AVG of a Genre across entire data set, and use that as baseline

Speaking with Steven Donoho
Assigned a Percentile for how it performs, each movie, each theater
What's the average percentile that this theater performs at 
How big of a slice of a pie did this theater net
Calculate Average of this theater and it's slice of the pie
When this theater, shows this genre, what percentile does this get, and what slice of the pie
Some theaters over perform on certain genres
Theater Genre
Theater Rating
Theater Genre Rating

New movie coming out (horror R), what is the predicted percentile for all theaters

Fault, not enough data (needs at least 3 Horror R to calculate, fall back to just genre, fall back to just R, fall just back to average)

Web App
Select Film, Shows Map of Top 10 Performers, and Bottom 10 Performers
Factors in, Theaters within 10 mi radius


### Timeline
Tuesday 12:30 - Code Freeze, anything in Github is fair game of grading
Tuesday 5pm - Capstone 3 One-Pager


silhouette score: average inner distance over the maximum outer distance

Threshold:
Establish a goal for a Theatre + Movie to 'beat'

Scores: 
Recall (Want to minimize False Negatives, don't want to misidentify a theatre that will over perform)
- Ok with False Positives, sometimes a theatre will be selected to play even if it's not the most profitable, because it is the only theatre in a certain mile radius

Baseline: Undersampled 150k 0, 55k 1
Using the OW boxoffice mean of the entire data set as a threshold for Over-Indexing, + Full Movie meta data, not scaled, no sampling
LR: Recall Score - 0.4436
DT: Recall Score - 0.4021
RF: Recall Score - 0.4021
GB: Recall Score - 0.3700

#2: Genre Avg used as Threshold, No Resampling, No Hyperparameter Tuning 144k 0  61k 1 
LR: Recall Score - 0.4553
DT: Recall Score - 0.3712
RF: Recall Score - 0.3715
GB: Recall Score - 0.3165

#3: Genre Avg used as Threshold, Undersampled, No Hyperparameter Tuning (SAME AS NO RESAMPLING) 120k each
LR: Recall Score - 0.4559
DT: Recall Score - 0.7708
RF: Recall Score - 0.7657
GB: Recall Score - 0.7708

#4: Genre Avg used as Threshold, Budget Standardized, Undersampled, No Hyperparameter Tuning (STANDARDIZTION ONLY HELPED LOGISTIC REGRESSION) 144k 0 61k 1
LR: Recall Score - 0.6561
DT: Recall Score - 0.7708
RF: Recall Score - 0.7657
GB: Recall Score - 0.7708

#5: TV Makret + Genre Avg used as Threshold, No Resampling, No Hyperparameter Tuning 144k 0 61k 1
LR: Recall Score - 0.6582
DT: Recall Score - 0.7577
RF: Recall Score - 0.7565
GB: Recall Score - 0.7576

#5: TV Makret + Genre Avg used as Threshold, No Resampling, No Hyperparameter Tuning 144k 0 61k 1, Undersampled
LR: Recall Score - 0.6245
DT: Recall Score - 0.7055
RF: Recall Score - 0.7000
GB: Recall Score - 0.7039

Profit Curve
Threshold should be at a theatre level, average of genre
Will inform our Precision and Recall threshold

#6: Location's Genre Avg used as Threshold, 125k 0 80k 1
LR: Recall Score - 0.4155
DT: Recall Score - 0.7894
RF: Recall Score - 0.7761
GB: Recall Score - 0.7858

#7: Location's Genre Avg used as Threshold, Standardized
LR: Recall Score - 0.6516
DT: Recall Score - 0.7870
RF: Recall Score - 0.7718
GB: Recall Score - 0.7839

#8: Location's Genre Avg used as Threshold, Standardized, GridSearch'd 125k 0 80k 1
LR: Recall Score - 0.6663
DT: Recall Score - 0.7769
RF: Recall Score - 0.7814
GB: Recall Score - 0.7940

#9: Location's Genre Avg used as Threshold, Standardized, Undersampled, GridSearch'd
LR: Recall Score - 0.6484
DT: Recall Score - 0.8247
RF: Recall Score - 0.8428
GB: Recall Score - 0.8187

Train Data - with GS
LR: Recall Score - 0.6782
DT: Recall Score - 0.6917
RF: Recall Score - 0.7274
GB: Recall Score - 0.7321

Unseen GB: Recall Score - 0.7284
GB: Recall Score - 0.7305

BASELINE NO OW
LR: Recall Score - 0.3936
DT: Recall Score - 0.7411
RF: Recall Score - 0.7412
GB: Recall Score - 0.7375

Unseen GS - No OW
LR: Recall Score - 0.4138
DT: Recall Score - 0.6725
RF: Recall Score - 0.7242
GB: Recall Score - 0.7283

Feedback
Cut down EDA TOP 10
Should be able to vocalize the slide
Feature importance does not have localized *
Juststick to ROC Curve in presentation, move precision recall to appendix * 

ADDING QUANTILE FOR 'Film_Buyer','Buying_Circuit','Payer','TV_Market','City'
37 Features now
Reevaluating Baseline
And Final Model


LR: Recall Score - 0.6873
DT: Recall Score - 0.7001
RF: Recall Score - 0.7231
GB: Recall Score - 0.7310

Unseen GB: Recall Score - 0.7281
GB: Recall Score - 0.7310



Day 7: 
* Try to Pickle Model
* Try to connect to Flask Webpage
* Set up test data set of Theatre + Genre, and then run model through, assign values
* Store predict prob and sort by DESC
* Plot Top 10 values of Theatres with Coordinates

EDA: Top 10 Cities with the Highest Over-Index Rate (with at least 54 Movies)
|                           City | over_indexcount | over_indextotal | over_index_% |
|-------------------------------:|----------------:|----------------:|-------------:|
|                   NEW BERN, NC |              35 |              57 |     0.614035 |
|                  PARAMOUNT, CA |              42 |              81 |     0.518519 |
|                     YAKIMA, WA |              31 |              60 |     0.516667 |
|                 WHITEHORSE, YU |              32 |              62 |     0.516129 |
|               ORANGE BEACH, AL |              29 |              59 |     0.491525 |
|                 ZANESVILLE, OH |              28 |              57 |     0.491228 |
| IDAHO FALLS, ID #1 IDAHO FALLS |              58 |             120 |     0.483333 |
|             BONITA SPRINGS, FL |              27 |              56 |     0.482143 |
|                  WENATCHEE, WA |              34 |              71 |     0.478873 |
|               NEW ROCHELLE, NY |              30 |              63 |     0.476190 |

* Season rated by the number of OI, Winter is interesting, 3rd highest occurences, with 3rd lowest total, highest OI%, Winter is categorized as Jan-Feb, and is considered a 'dumping ground' for films/not as profitable
|       Season | over_indexcount | over_indextotal | over_index_% |
|-------------:|----------------:|----------------:|-------------:|
|       Summer |           24799 |           66253 |     0.374308 |
|         Fall |           18849 |           43951 |     0.428864 |
|       Winter |           13392 |           29885 |     0.448118 |
|       Spring |           12987 |           38265 |     0.339396 |
|    Christmas |            6232 |           14763 |     0.422136 |
| Thanksgiving |            3880 |           12307 |     0.315268 |

* Top OI% from Division+Genre Combination
| Division |           Genre | over_indexcount | over_indextotal | over_index_% |
|---------:|----------------:|----------------:|----------------:|-------------:|
| SOUTHERN | Romantic Comedy |            1100 |            2197 |     0.500683 |
| SOUTHERN |  Sci-Fi/Fantasy |            1146 |            2311 |     0.495889 |
| SOUTHERN |  Romantic Drama |             727 |            1482 |     0.490553 |
|  WESTERN |  Sci-Fi/Fantasy |            1057 |            2194 |     0.481768 |
|  EASTERN |  Sci-Fi/Fantasy |            1522 |            3171 |     0.479975 |
|  WESTERN | Romantic Comedy |            1050 |            2188 |     0.479890 |
|  EASTERN | Romantic Comedy |            1416 |            2976 |     0.475806 |
|  EASTERN |  Romantic Drama |             872 |            1848 |     0.471861 |
| SOUTHERN | Children/Family |             496 |            1053 |     0.471035 |
|  WESTERN |  Romantic Drama |             618 |            1317 |     0.469248 |