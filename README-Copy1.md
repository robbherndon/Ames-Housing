# Project 2 - Ames Housing Data and Kaggle Challenge

                                         ![](visualizations/houses.jpeg)

## Problem

Before this new age of information, getting an appraisal on a house was a long, tedious process. It required an appraiser to go back through all of the records of a given county, find all of the homes that had similar features to the one being sold, and come up with a "comparable" price to those previous sold homes. Then came the information revolution and along came websites like Redfin, Realator.com, and Zillow.  Of these 3, it is Zillow who has most taken on the role of psudo-appraiser with their "Zestimate" of a given house's potential value. As a home owner myself, I know how highly the Zestimate is held when buying or selling a house. Many people today waive the official appraisals when buying a home, choosing to simply use the estimates given by these listing websites. How accurate are these "estimates" though and what goes into making them? That is what I set out to discover in this project.

This is the question that is plaguing my client, who is filing a lawsuit against Zillow because they have continously under valued her home, which has caused her to lose a substantial amount of money. 

## Description

Through this project I will setting out to find if the algorithms that give psuedo appraisals on sights like Redfin and Zillow are giving fair evaluations of the houses, or can it be shown that their models are consistently undervaluing homes.

In order to do that I created 3 separate models, one with multiple-linear regression and a moderate amount of fetaure engineering, one with a much greater amount of feature engineering, and the third one that uses the "lasso" method to choose the best features to use.

What I will show in this research is that the models consistently predict prices that lower than the actual prices that those homes were sold for. And this is not by a small margin, On average the models I used under priced the value of the homes in the dataset by $5,000 to $8,500 , or by roughly 5% of the true value.

The first model I created under valued the homes by $6,000

![](visualizations/MLR%with%Feature%Engineering.png)

The second model I created under valued the homes by $5,500
![](visualizations/Polynomial%Predictions%vs%Real%Value.png)

The third model I created under valued the homes by $8,500
![](visualizations/Lasso%vs%Real.png)



# Process

I started with a really simple process of finding the features with the highest initial correlation with the sales price column. This can be seen in the following image:
![](visualiations/Sales%Price%Heatmap.png)

After finding those features with the highest correlation I started with a straightforward linear regression model including all of the features with correlation above positive .5.  I wanted to use this as a bench mark for future models that would have cleaned data, feature engineering, polynomial columns, and dummy columns. This first model had a fairly poor cross val score, but that was to be expected. 

Next it was time to start the data cleaning. It didn't take long to spot some the issues within the data. To begin with there were many columns with large amounts of zeros for values. The 3 most egregious were the "pool area", "mass vrn area", and the "wood deck square feet" columns. As seen in this graph: ![](visualiations/columns%with%zeros.png)
My assumption about these columns was that the rows with valus of 0 meant that the homes did not have these features. Since there was a high level of zeros, I ruled out using these features in any models.

My next task was to find how many NaN values were in the dataset. There were plenty of NaN values in the columns with ints and floats however, there were several object columns as well where the NaN values were represented with "N/A". Of the 80+ columns in the dataframe there seemed to be too many of them with NaN values to go through and replace them individually. So I created a function that would take in the whole dataframe, determine the datatype of the values in each column, then replace each NaN with "missing" if it was an object column or with the median value of the column if the datatype was an int or a float. 
The function is : 

def cat_convert_nan(df):
    for col in df.loc[:, df.dtypes == 'O']:
        df[col] = df[col].fillna(“Missing”)
    for cols in df.loc[:, df.dtypes == 'int64']:
        df[cols] = df[cols].fillna(df[cols].median())
    for columm in df.loc[:, df.dtypes == 'float64']:
        df[columm] = df[columm].fillna(df[columm].median())
    return df

After creating the above function I was simply able to run each of the training and test datasets through the function and transform all of the NaNs in every column.

The next big discovery in data cleaning was that there were many ordinal columns, which made sense in that we were looking at housing data, and ordinal valuations are key for qualitative valuations. I aslo decided that I did not want to go through and change each ordinal column from their string values into numeric equivalent scores. So I wrote a function for that also:

ordinal_change = {"NA":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5}
def convert_cond_qual(df):
    for col in df.loc[:, df.dtypes == 'O']:
        df[col] = df[col].replace(ordinal_change)
    return df

It honestly took me a while to figure out the logic of each function (which seems weird now looking at how simple they ended up) but the experience of creating them will, I'm sure, help with future projects. Being able to automate much of the cleaning task was a great relief.


Now that I had cleaned each of the training and test datasets, I felt confident in moving into feature engineering. After completeling the cleaning I took another look at the correlations of the columns: 

![](visualiations/Full%Heat%Map.png) 


From this map I noticed that highest correlations were the "quality" columns, which were ordinal values. I decided that I would feature engineer columns that combined each "quality" column with the corresponding "condition" column. There were combinations made for the "overall", "kitchen", "garage" and "exterior" pairs. I discovered that pairwise combinations had a negative effect on the correlation with sale price, and so decided to scrap those columns. I did still want to engineer columns that could combine the "quality" columns, and after trying some pairs out I discovered that it was best to engineer one column that was all of the "quality" columns together, "total quality". This new feature now had the highest correlation score of any feature.

I decided it was time to test out these new features in the original linear regression model to see if there was much improvement. This was clearly going to be the case, and the subsequent model performed with an r2 score of .84 and a cross val score of .83. I decided that would be a good enough value to make my first submission with. This is the histogram showing the range of values for both the prediction model and the recorded prices of the houses:

![](visualiations/MLR%with%Feature%Engineering.png)

After making my first submission, I wanted to explore if there were other ways I could improve the model. I decided that would try to use the polynomial function along with creating dummy columns of the most highly correlated object columns. This took the number of columns from roughly 80 to well over 200. It was now time to try a new model with all of the new features!

Unfortunately this models performance was worse than the previous MLR model. I wasn't sure quite why this model performed poorly, but I took this as an opportunity to try out the process called "lasso". By using this process, it is possible to find out the potentially best features to use in a model. First, all of the data needs to be scaled so that it all can be judged equally by lasso. In order to accomplish this, it is best to use the SimpleScaler, which turns all values in the dataframe into their z-scores. This equalizes the data. Then, lasso sifts through all of the coefficients and can show which have the highest correlation, as seen in this graph:

![](visualiations/Largest%Coefficients.png)

I used those top 10 features in the next linear regression model and ended up with an even worse score. Unfortunately I ran out of time on this project and was not able to go back and do an investigation as to why this happened. That is definitely something for a future project to discover.


## Conclusions

Websites and apps like Redfin, Realtor.com and Zillow use algorithms to predict values of homes across this country. What each of my three models demonstrate is that it is highly probable that the models used for prediction are under valuing the homes. For future research, I would need to spend more time with the data, do a more thourough cleaning and exploration to find ways to optimize my current models. Then I would like to see if the models can generalize on home data from any other counties to test whether the model under predicts the values of those homes as well.



**Primary Learning Objectives:**
1. Creating and iteratively refining a regression model
2. Using [Kaggle](https://www.kaggle.com/) to practice the modeling process
3. Providing business insights through reporting and presentation.

You are tasked with creating a regression model based on the Ames Housing Dataset. This model will predict the price of a house at sale.

The Ames Housing Dataset is an exceptionally detailed and robust dataset with over 70 columns of different features relating to houses.

Secondly, we are hosting a competition on Kaggle to give you the opportunity to practice the following skills:

- Refining models over time
- Use of train-test split, cross-validation, and data with unknown values for the target to simulate the modeling process
- The use of Kaggle as a place to practice data science

As always, you will be submitting a technical report and a presentation. **You may find that the best model for Kaggle is not the best model to address your data science problem.**

## Set-up

Before you begin working on this project, please do the following:

1. Sign up for an account on [Kaggle](https://www.kaggle.com/)
2. **IMPORTANT**: Click this link ([Regression Challenge Sign Up](https://www.kaggle.com/t/d365a5de71e84ec798ceb95995ebbee8)) to **join** the competition (otherwise you will not be able to make submissions!)
3. Review the material on the [DSIR-1011 Regression Challenge](https://www.kaggle.com/c/dsir-1011-project-2-regression-challenge)
4. Review the [data description](http://jse.amstat.org/v19n3/decock/DataDocumentation.txt).

## The Modeling Process

1. The train dataset has all of the columns that you will need to generate and refine your models. The test dataset has all of those columns except for the target that you are trying to predict in your Regression model.
2. Generate your regression model using the training data. We expect that within this process, you'll be making use of:
    - train-test split
    - cross-validation / grid searching for hyperparameters
    - strong exploratory data analysis to question correlation and relationship across predictive variables
    - code that reproducibly and consistently applies feature transformation (such as the preprocessing library)
3. Predict the values for your target column in the test dataset and submit your predictions to Kaggle to see how your model does against unknown data.
    - **Note**: Kaggle expects to see your submissions in a specific format. Check the challenge's page to make sure you are formatting your CSVs correctly!
    - **You are limited to models you've learned in class**. In other words, you cannot use XGBoost, Neural Networks or any other advanced model for this project.
4. Evaluate your models!
    - consider your evaluation metrics
    - consider your baseline score
    - how can your model be used for inference?
    - why do you believe your model will generalize to new data?

## Submission

Materials must be submitted by the beginning of class on **Friday, Nov. 5**.

Your technical report will be hosted on Github Enterprise. Make sure it includes:

- A README.md (that isn't this file)
- Jupyter notebook(s) with your analysis and models (renamed to describe your project)
- At least one successful prediction submission on [DSIR-1011 Regression Challenge](https://www.kaggle.com/c/dsir-1011-project-2-regression-challenge) --  you should see your name in the "[Leaderboard](https://www.kaggle.com/c/dsir-1011-project-2-regression-challenge/leaderboard)" tab.
- Data files
- Presentation slides
- Any other necessary files (images, etc.)

**Submit your materials through our [google classroom](https://classroom.google.com/u/1/c/NDEwMTU3NDY4OTQ4).**

---

## Presentation Structure

- **Must be within 5 minutes.**
- Use Google Slides or some other visual aid (Keynote, Powerpoint, etc).
- Consider the audience. **Semi-technical**.
- Start with the **data science problem**.
- Use visuals that are appropriately scaled and interpretable.
- Talk about your procedure/methodology (high level).
- Talk about your primary findings.
- Make sure you provide **clear recommendations** that follow logically from your analyses and narrative and answer your data science problem.

Be sure to rehearse and time your presentation before class.

---

## Rubric
We will evaluate your project (for the most part) using the following criteria.  You should make sure that you consider and/or follow most if not all of the considerations/recommendations outlined below **while** working through your project.

**Scores will be out of 27 points based on the 9 items in the rubric.** <br>
*3 points per section*<br>

| Score | Interpretation |
| --- | --- |
| **0** | *Project fails to meet the minimum requirements for this item.* |
| **1** | *Project meets the minimum requirements for this item, but falls significantly short of portfolio-ready expectations.* |
| **2** | *Project exceeds the minimum requirements for this item, but falls short of portfolio-ready expectations.* |
| **3** | *Project meets or exceeds portfolio-ready expectations; demonstrates a thorough understanding of every outlined consideration.* |

### The Data Science Process

**Problem Statement**
- Is it clear what the student plans to do?
- What type of model will be developed?
- How will success be evaluated?
- Is the scope of the project appropriate?
- Is it clear who cares about this or why this is important to investigate?
- Does the student consider the audience and the primary and secondary stakeholders?

**Data Cleaning and EDA**
- Are missing values imputed appropriately?
- Are distributions examined and described?
- Are outliers identified and addressed?
- Are appropriate summary statistics provided?
- Are steps taken during data cleaning and EDA framed appropriately?
- Does the student address whether or not they are likely to be able to answer their problem statement with the provided data given what they've discovered during EDA?

**Preprocessing and Modeling**
- Are categorical variables one-hot encoded?
- Does the student investigate or manufacture features with linear relationships to the target?
- Have the data been scaled appropriately?
- Does the student properly split and/or sample the data for validation/training purposes?
- Does the student utilize feature selection to remove noisy or multi-collinear features?
- Does the student test and evaluate a variety of models to identify a production algorithm (**AT MINIMUM:** linear regression, lasso, and ridge)?
- Does the student defend their choice of production model relevant to the data at hand and the problem?
- Does the student explain how the model works and evaluate its performance successes/downfalls?

**Evaluation and Conceptual Understanding**
- Does the student accurately identify and explain the baseline score?
- Does the student select and use metrics relevant to the problem objective?
- Is more than one metric utilized in order to better assess performance?
- Does the student interpret the results of their model for purposes of inference?
- Is domain knowledge demonstrated when interpreting results?
- Does the student provide appropriate interpretation with regards to descriptive and inferential statistics?

**Conclusion and Recommendations**
- Does the student provide appropriate context to connect individual steps back to the overall project?
- Is it clear how the final recommendations were reached?
- Are the conclusions/recommendations clearly stated?
- Does the conclusion answer the original problem statement?
- Does the student address how findings of this research can be applied for the benefit of stakeholders?
- Are future steps to move the project forward identified?

### Organization and Professionalism

**Project Organization**
- Are modules imported correctly (using appropriate aliases)?
- Are data imported/saved using relative paths?
- Does the README provide a good executive summary of the project?
- Is markdown formatting used appropriately to structure notebooks?
- Are there an appropriate amount of comments to support the code?
- Are files & directories organized correctly?
- Are there unnecessary files included?
- Do files and directories have well-structured, appropriate, consistent names?

**Visualizations**
- Are sufficient visualizations provided?
- Do plots accurately demonstrate valid relationships?
- Are plots labeled properly?
- Are plots interpreted appropriately?
- Are plots formatted and scaled appropriately for inclusion in a notebook-based technical report?

**Python Syntax and Control Flow**
- Is care taken to write human readable code?
- Is the code syntactically correct (no runtime errors)?
- Does the code generate desired results (logically correct)?
- Does the code follows general best practices and style guidelines?
- Are Pandas functions used appropriately?
- Are `sklearn` methods used appropriately?

**Presentation**
- Is the problem statement clearly presented?
- Does a strong narrative run through the presentation building toward a final conclusion?
- Are the conclusions/recommendations clearly stated?
- Is the level of technicality appropriate for the intended audience?
- Is the student substantially over or under time?
- Does the student appropriately pace their presentation?
- Does the student deliver their message with clarity and volume?
- Are appropriate visualizations generated for the intended audience?
- Are visualizations necessary and useful for supporting conclusions/explaining findings?

In order to pass the project, students must earn a minimum score of 1 for each category.
- Earning below a 1 in one or more of the above categories would result in a failing project.
- While a minimum of 1 in each category is the required threshold for graduation, students should aim to earn at least an average of 1.5 across each category. An average score below 1.5, while it may be passing, means students may want to solicit specific feedback in order to significantly improve the project before showcasing it as part of a portfolio or the job search.

### REMEMBER:

This is a learning environment and you are encouraged to try new things, even if they don't work out as well as you planned! While this rubric outlines what we look for in a _good_ project, it is up to you to go above and beyond to create a _great_ project. **Learn from your failures and you'll be prepared to succeed in the workforce**.
