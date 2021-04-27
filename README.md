# Identifiying-Over-and-Under-Performing-US-Movie-Theaters
Creating a predictive model that will identify and segment similar performing movie theaters, to better identify and target for maximize profit

## Background
With the ever-changing theatrical landscape, it is now more important than ever, to maximize a Film's Box Office success at a theatre. As the theatrical window (the time period between when a movie is played in theatres to when it ends up on screen at home) is growing more and more smaller, so does it's chances for generating revenue. A film that could have spent weeks and weeks steadily holding over it's gross week-to-week, now may not get that same chance. 

To help maximize a Film's per location gross, I worked to develop a predictive model that would identify Theatre's that strongly performed above their own average gross, given the type of Film that was playing. 

## The Data
The data set for this project was built upon many similar data sources, all providing different feautre's for a Film.
1. Booking Information: Provided from confidential work source. 1,000,000 rows spanning the all the playweeks of a Studio's slate, from 2017-2019.
    * From this initial data set, data clean-up was performed, and the scope of the project narrowed down to primarily targeting the First Play Week of First Run theatres, the initial Wide wave of theatre's playing a Film on release.
2. The Movie Database (TMDb): Provided Budget, Runtime, and Plot Overview fields to help add more feature's into the dataset. 
3. Box Office Reporting: Opening Weekend to YT Views: Metrics tracked by Box Office Reporting, tracks views from the official YouTube sources, as well as Fandango. Counting is stopped on the Saturday prior to release
4. Internal Metadata report: Film features include Season, Genre, and Number of Locations when a Film was released.

## Data Clean-Up and Preparation
The bulk of the work for this dataset came from finding a way to neatly tie all of these sources together. They all provided their own unique method of labelling, where one film could be titled in four different ways. I employed a function that utilized the *difflib* Python library, that looked to match a target word from a selection of available choices. Other methods considered were FuzzyWuzzy lookup and Levenshtein measurement for similarity. Even after employing a matching function, there were still a few datapoints that needed manual intervention and attention. 

* Overall, there were 54 titles within the movie dataset.
* The Opening Week dataset consisted of 200k+ rows of Theatre + Film combinations. 

To create my 'target feature' for this classification project, I estblished a threshold at each theatre+film level that the Box Office gross needed to surpass to be considered **over-index**. This was a binary classification, where 1 was the case when a theatre+film combination over performed, and 0 was when it did not pass the threshold. 

## EDA
![Distribution](./images/Distribution_Num_Locs_OI.png)
Looking at the distribution of Theatres that do Over-Index, the large majority of data falls around the 20-film mark, meaning if a location played all 54 Film's in the dataset, almost half of them over performed.

So what kinds of Film's are we working with here? To begin my exploratory data analysis, I first segmented the Over-Indexed values down by Rating. PG13 films account for more that 266% of the Over-Indexed data points than their PG counterpart, due to the large part that there aren't quite as many PG films rated within the dataset. PG13 films tend to be the most popular at the box office, as they have the widest appeal to audiences. 

![Over Index By Rating](./images/OI_byRating.png)

![Over Index By Genre](./images/OI_byGenre.png)
Further breaking down this distribution, I examined the Over-Indexed counts by Genre. Here, we see a large lead for the Action/Adventure genre. This could most likely be due to the fact that many Summer and Holiday blockbusters fall into this Genre, and like we saw above, usually garner a PG13 rating. 

It's interesting to see Comedy in second place, as the genre has not been as popular as it was compared to last decade. A sign that the genre still has the ability to perform well at the Box Office, better than some of us think. 

Lastly, I took a look at the total counts of Over-Indexed values by Season. Again, to further agree with my previous two graphs, Summer holds a substantial lead from the next season. Christmas and Thanksgiving rank at the bottom of this list, but Film's in this bucket may perform strongly (high box office gross) during this period, but may have established a high average that is hard to over-perform.

![Over Index by Season](./images/OI_bySeason.png)

|       Season | over_indexcount | over_indextotal | over_index_% |
|-------------:|----------------:|----------------:|-------------:|
|       Summer |           24799 |           66253 |     0.374308 |
|         Fall |           18849 |           43951 |     0.428864 |
|       Winter |           13392 |           29885 |     0.448118 |
|       Spring |           12987 |           38265 |     0.339396 |
|    Christmas |            6232 |           14763 |     0.422136 |
| Thanksgiving |            3880 |           12307 |     0.315268 |

Looking more into detail of the Season breakdown, it is interesting to examine the Winter values 
(Winter is defined as January to February): 
* Winter has the 3rd highest Over Index Count, while having the 3rd lowest Over Index total
* Winter also has the highest Over Index percent, at 44%
* January, and by extension February, has long been thought of as a 'dumping ground' for Film's, where a release during this time typically means a quick and unmemorably exit from the box office in a few weeks time. 
* It appears as if some Film's are exceeding the low bar of the Season, and are having the chance to out-perform their lowly average, and over-index. 
    * Though, values could also be accounted for from February, which has been also improving in performance as a month

## Modelling

## Results

## Next Steps