# anomalous_stocks_search

This repository contains my [heroku dash app](https://anomalous-stocks-761f7ea35d00.herokuapp.com/) and any scripts which feed data into it. The market_source folder contains all of the current content on the site, and I hope to get the guru_query script working again too (currently being blocked from making requests).

I'm pulling market data daily from https://stockanalysis.com/api/screener api and related endpoints found in scripts/constants.py > LINKS. I clean and transform the data (market_data_transformed.pkl) resulting in a search of ~5100 stocks and then use an xgboost model to predict stock price across the dataset. I chose an xgboost model to learn as many linear relationships as possible in the source data. The model feature columns were chosen because of their relatively low missing rate and trial/error evidence of importance. 

The goal in fitting these stocks is to identify anomalous stocks which weren't accurately predicted from this model, the idea being that something different is affecting the price of these stocks that can't be explained from the input data. I used an IsolationForest model, Z-Score, and Standard Error from the best fit line as my anomaly metrics. Each have benefits, mainly filtering different regions of anomalous points, but the IsolationForest model or Isolation Score is the best general purpose metric in my opinion.

## Future Work

1) Getting stock value data from gurufocus.com to compare to the xg boost model prediction and actual price in a toggleable 3d scatterplot on the page.
2) Dynamically scale all the sites plots and selection components with screen size.
