# Using Gradient Boosting for predicting NYC Cab Trip duration
### Overview 
Many on-demand logistics companies like Uber and Lyft are displaying their predicted ETA for an enhanced customer experience. Predicting accurate estimated time arrival is imperative to logistics providers. Predicting arrival time is not always accurate as few situations are inherently unpredictable such as the weather and sudden spike in demand in certain neighborhoods. I am going to focus on defining meaningful,new features and using a Gradient Boosted Decision Tree Regressor to create the prediction model. 


### Goal
To predict the expected time of arrival of a cab in New York city
Dataset: http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml . Subset at https://www.kaggle.com/c/nyc-taxi-trip-duration/data

![heatmap of popular routes for cabs](https://github.com/sroy4/Predicting-ride-ETA-using-GBDT-/blob/main/pick%20up%20-drop%20off%20heat%20map.png)

### Methdology

The Kaggle dataset has pickup and drop-off coordinates. The first step is to map these coordinate into neighborhoods. Using Carto and Zillow's NYC shape files, we map each coordinate to a neighboorhood. 

We're missing weather information from the dataset. We import weather data from noaa.gov and integrate it with the taxi dataset based on date. 

Now, we define some aggregate features based on the pickup and dropoff location information. For example, average drop-off times from specific counties, e.g. from Manhattan to Queens are added as features. 

Finally, I use a GBDT regressor to predict ride duration. 


