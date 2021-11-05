# Project 2 - Ames Housing Data and Kaggle Challenge

                


## Problem

Before this new age of information, getting an appraisal on a house was a long, tedious process. It required an appraiser to go back through all of the records of a given county, find all of the homes that had similar features to the one being sold, and come up with a "comparable" price to those previous sold homes. Then came the information revolution and along came websites like Redfin, Realator.com, and Zillow.  Of these 3, it is Zillow who has most taken on the role of psudo-appraiser with their "Zestimate" of a given house's potential value. As a home owner myself, I know how highly the Zestimate is held when buying or selling a house. Many people today waive the official appraisals when buying a home, choosing to simply use the estimates given by these listing websites. How accurate are these "estimates" though and what goes into making them? That is what I set out to discover in this project.

This is the question that is plaguing my client, who is filing a lawsuit against Zillow because they have continously under valued her home, which has caused her to lose a substantial amount of money. 

## Description

Through this project I will setting out to find if the algorithms that give psuedo appraisals on sights like Redfin and Zillow are giving fair evaluations of the houses, or can it be shown that their models are consistently undervaluing homes.

In order to do that I created 3 separate models, one with multiple-linear regression and a moderate amount of fetaure engineering, one with a much greater amount of feature engineering, and the third one that uses the "lasso" method to choose the best features to use.

What I will show in this research is that the models consistently predict prices that lower than the actual prices that those homes were sold for. And this is not by a small margin, On average the models I used under priced the value of the homes in the dataset by $5,000 to $8,500 , or by roughly 5% of the true value.

The first model I created under valued the homes by $6,000

![](visualizations/MLR%20with%20Feature%20Engineering.png)

The second model I created under valued the homes by $5,500
![Polynomial](visualizations/Polynomial%20Predictions%20vs%20Real%20Value.png)

The third model I created under valued the homes by $8,500
![Lasso](visualizations/Lasso%20vs%20Real.png)

## Data Used

A description of the data used in this project can be found here: [data description](http://jse.amstat.org/v19n3/decock/DataDocumentation.txt)


## Process

I started with a really simple process of finding the features with the highest initial correlation with the sales price column. This can be seen in the following image:
![Sales Heatmap](visualiations/Sales%20Price%20Heatmap.png)

After finding those features with the highest correlation I started with a straightforward linear regression model including all of the features with correlation above positive .5.  I wanted to use this as a bench mark for future models that would have cleaned data, feature engineering, polynomial columns, and dummy columns. This first model had a fairly poor cross val score, but that was to be expected. 

Next it was time to start the data cleaning. It didn't take long to spot some the issues within the data. To begin with there were many columns with large amounts of zeros for values. The 3 most egregious were the "pool area", "mass vrn area", and the "wood deck square feet" columns. As seen in this graph: ![Columns with Zeros](visualiations/columns%20with%20zeros.png)
My assumption about these columns was that the rows with valus of 0 meant that the homes did not have these features. Since there was a high level of zeros, I ruled out using these features in any models.

My next task was to find how many NaN values were in the dataset. There were plenty of NaN values in the columns with ints and floats however, there were several object columns as well where the NaN values were represented with "N/A". Of the 80+ columns in the dataframe there seemed to be too many of them with NaN values to go through and replace them individually. So I created a function that would take in the whole dataframe, determine the datatype of the values in each column, then replace each NaN with "missing" if it was an object column or with the median value of the column if the datatype was an int or a float. 

After creating the above function I was simply able to run each of the training and test datasets through the function and transform all of the NaNs in every column.

The next big discovery in data cleaning was that there were many ordinal columns, which made sense in that we were looking at housing data, and ordinal valuations are key for qualitative valuations. I aslo decided that I did not want to go through and change each ordinal column from their string values into numeric equivalent scores. So I wrote a function for that also.


It honestly took me a while to figure out the logic of each function (which seems weird now looking at how simple they ended up) but the experience of creating them will, I'm sure, help with future projects. Being able to automate much of the cleaning task was a great relief.


Now that I had cleaned each of the training and test datasets, I felt confident in moving into feature engineering. After completeling the cleaning I took another look at the correlations of the columns: 

![Full Heat Map](visualiations/Full%20Heat%20Map.png) 


From this map I noticed that highest correlations were the "quality" columns, which were ordinal values. I decided that I would feature engineer columns that combined each "quality" column with the corresponding "condition" column. There were combinations made for the "overall", "kitchen", "garage" and "exterior" pairs. I discovered that pairwise combinations had a negative effect on the correlation with sale price, and so decided to scrap those columns. I did still want to engineer columns that could combine the "quality" columns, and after trying some pairs out I discovered that it was best to engineer one column that was all of the "quality" columns together, "total quality". This new feature now had the highest correlation score of any feature.

I decided it was time to test out these new features in the original linear regression model to see if there was much improvement. This was clearly going to be the case, and the subsequent model performed with an r2 score of .84 and a cross val score of .83. I decided that would be a good enough value to make my first submission with. This is the histogram showing the range of values for both the prediction model and the recorded prices of the houses:

![MLR](visualiations/MLR%20with%20Feature%20Engineering.png)

After making my first submission, I wanted to explore if there were other ways I could improve the model. I decided that would try to use the polynomial function along with creating dummy columns of the most highly correlated object columns. This took the number of columns from roughly 80 to well over 200. It was now time to try a new model with all of the new features!

Unfortunately this models performance was worse than the previous MLR model. I wasn't sure quite why this model performed poorly, but I took this as an opportunity to try out the process called "lasso". By using this process, it is possible to find out the potentially best features to use in a model. First, all of the data needs to be scaled so that it all can be judged equally by lasso. In order to accomplish this, it is best to use the SimpleScaler, which turns all values in the dataframe into their z-scores. This equalizes the data. Then, lasso sifts through all of the coefficients and can show which have the highest correlation, as seen in this graph:

![Highest Coefficients](visualiations/Largest%20Coefficients.png)

I used those top 10 features in the next linear regression model and ended up with an even worse score. Unfortunately I ran out of time on this project and was not able to go back and do an investigation as to why this happened. That is definitely something for a future project to discover.


## Conclusions

Websites and apps like Redfin, Realtor.com and Zillow use algorithms to predict values of homes across this country. What each of my three models demonstrate is that it is highly probable that the models used for prediction are under valuing the homes. For future research, I would need to spend more time with the data, do a more thourough cleaning and exploration to find ways to optimize my current models. Then I would like to see if the models can generalize on home data from any other counties to test whether the model under predicts the values of those homes as well.




