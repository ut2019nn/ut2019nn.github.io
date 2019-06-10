## Two Sigma: Using News to Predict Stock Movements

The project was implemented as a part of Neural Networks (LTAT.02.001) course in University of Tartu. The original Kaggle competition can be found [here](https://www.kaggle.com/c/two-sigma-financial-news/data).

Collaborators:
- Erki  Aun
- Hindrek Teder
- Karen Danielyan
- Rudayasekaran PS

### Goal

In this project, we predict future stock price returns based on two sources of data:
- Market data (2007 to present) provided by Intrinio.
- News data (2007 to present) provided by Thomson Reuters.

### Abstract

### Datasets

#### Market Data

The data includes a subset of US-listed instruments. The set of included instruments changes daily and is determined based on the amount traded and the availability of information.

Within the marketdata, you will find the following columns:
- `time(datetime64[ns, UTC])` - the current time (in marketdata, all rows are taken at 22:00 UTC)
- `assetCode(object)` - a unique id of an asset
- `assetName(category)` - the name that corresponds to a group of assetCodes. These may be "Unknown" if the corresponding assetCode does not have any rows in the news data.
- `universe(float64)` - a boolean indicating whether or not the instrument on that day will be included in scoring. This value is not provided outside of the training data time period. The trading universe on a given date is the set of instruments that are avilable for trading (the scoring function will not consider instruments that are not in the trading universe). The trading universe changes daily.
- `volume(float64)` - trading volume in shares for the day
- `close(float64)` - the close price for the day (not adjusted for splits or dividends)
- `open(float64) -` the open price for the day (not adjusted for splits or dividends)
- `returnsClosePrevRaw1(float64)` - see returns explanation above
- `returnsOpenPrevRaw1(float64)` - see returns explanation above
- `returnsClosePrevMktres1(float64)` - see returns explanation above
- `returnsOpenPrevMktres1(float64)` - see returns explanation above
- `returnsClosePrevRaw10(float64)` - see returns explanation above
- `returnsOpenPrevRaw10(float64)` - see returns explanation above
- `returnsClosePrevMktres10(float64)` - see returns explanation above
- `returnsOpenPrevMktres10(float64)` - see returns explanation above
- `returnsOpenNextMktres10(float64)` - 10 day, market-residualized return. This is the target variable used in competition scoring.

#### Target variable 'returnsOpenNextMktres10' resampled to dayly, weekly and monthly frequency
![Time series plots for the target variable plots](https://drive.google.com/uc?export=view&id=1Eh4C3lEa9i9q3UfgnUdXOFv3TBVkHC0J)

#### Histogram for the target variable 'returnsOpenNextMktres10'
![Histogram for the target variable](https://drive.google.com/uc?export=view&id=1-ELVVM0_JDU2YkGOeo1EED5-ANIrjLX6)

#### Target variable 'returnsOpenNextMktres10' with independent variable 'returnsOpenPrevMktres10'
![Target variable 'returnsOpenNextMktres10' with 'returnsOpenPrevMktres10'](https://drive.google.com/uc?export=view&id=1-KbbygAwegofZwzdKQXI1FSr9D8GbNMJ)

#### News Data

The news data contains information at both the news article level and asset level.

- `time(datetime64[ns, UTC])` - UTC timestamp showing when the data was available on the feed (second precision)
- `sourceTimestamp(datetime64[ns, UTC])` - UTC timestamp of this news item when it was created
- `firstCreated(datetime64[ns, UTC])` - UTC timestamp for the first version of the item
- `sourceId(object)` - an Id for each news item
- `headline(object)` - the item's headline
- `urgency(int8)` - differentiates story types (1: alert, 3: article)
- `takeSequence(int16)` - the take sequence number of the news item, starting at 1. For a given story, alerts and articles have separate sequences.
- `provider(category)` - identifier for the organization which provided the news item (e.g. RTRS for Reuters News, BSW for Business Wire)
- `subjects(category)` - topic codes and company identifiers that relate to this news item. Topic codes describe the news item's subject matter. These can cover asset classes, geographies, events, industries/sectors, and other types.
- `audiences(category)` - identifies which desktop news product(s) the news item belongs to. They are typically tailored to specific audiences. (e.g. "M" for Money International News Service and "FB" for French General News Service)
- `bodySize(int32)` - the size of the current version of the story body in characters
- `companyCount(int8)` - the number of companies explicitly listed in the news item in the subjects field
- `headlineTag(object)` - the Thomson Reuters headline tag for the news item
- `marketCommentary(bool)` - boolean indicator that the item is discussing general market conditions, such as "After the Bell" summaries
- `sentenceCount(int16)` - the total number of sentences in the news item. Can be used in conjunction with firstMentionSentence to determine the relative position of the first mention in the item.
- `wordCount(int32)` - the total number of lexical tokens (words and punctuation) in the news item
- `assetCodes(category)` - list of assets mentioned in the item
- `assetName(category)` - name of the asset
- `firstMentionSentence(int16)` - the first sentence, starting with the headline, in which the scored asset is mentioned.
  - 1: headline
  - 2: first sentence of the story body
  - 3: second sentence of the body, etc
  - 0: the asset being scored was not found in the news item's headline or body text. As a result, the entire news item's text (headline + body) will be used to determine the sentiment score.
- `relevance(float32)` - a decimal number indicating the relevance of the news item to the asset. It ranges from 0 to 1. If the asset is mentioned in the headline, the relevance is set to 1. When the item is an alert (urgency == 1), relevance should be gauged by firstMentionSentence instead.
- `sentimentClass(int8)` - indicates the predominant sentiment class for this news item with respect to the asset. The indicated class is the one with the highest probability.
- `sentimentNegative(float32)` - probability that the sentiment of the news item was negative for the asset
- `sentimentNeutral(float32)` - probability that the sentiment of the news item was neutral for the asset
- `sentimentPositive(float32)` - probability that the sentiment of the news item was positive for the asset
- `sentimentWordCount(int32)` - the number of lexical tokens in the sections of the item text that are deemed relevant to the asset. This can be used in conjunction with wordCount to determine the proportion of the news item discussing the asset.
- `noveltyCount12H(int16)` - The 12 hour novelty of the content within a news item on a particular asset. It is calculated by comparing it with the asset-specific text over a cache of previous news items that contain the asset.
- `noveltyCount24H(int16)` - same as above, but for 24 hours
- `noveltyCount3D(int16)` - same as above, but for 3 days
- `noveltyCount5D(int16)` - same as above, but for 5 days
- `noveltyCount7D(int16)` - same as above, but for 7 days
- `volumeCounts12H(int16)` - the 12 hour volume of news for each asset. A cache of previous news items is maintained and the number of news items that mention the asset within each of five historical periods is calculated.
- `volumeCounts24H(int16)` - same as above, but for 24 hours
- `volumeCounts3D(int16)` - same as above, but for 3 days
- `volumeCounts5D(int16)` - same as above, but for 5 days
- `volumeCounts7D(int16)` - same as above, but for 7 days

#### Time series plots for noveltyCount3D, 5D and 7D
![Time series plots for noveltyCount3D, 5D and 7D](https://drive.google.com/uc?export=view&id=1-Lt2EZraprei4LXROvPo76sQ7T6MBVbq)

#### Kernel density estimate (KDE) plot for noveltyCount3D, 5D and 7D
![KDE plot for noveltyCount3D, 5D and 7D](https://drive.google.com/uc?export=view&id=1-m43R4Q5spjPxWejrUiMWDNYSgn2z-EX)

#### Kernel density estimate (KDE) plot for volumeCounts3D, 5D and 7D
![KDE plot for volumeCounts3D, 5D and 7D](https://drive.google.com/uc?export=view&id=1-oJ5lqEFvWh0PUWgn3HuJH_PKPdeOdvv)

#### Scatter plot for sentimentPositive, sentimentNegative and sentimentClass
![Scatter plot for sentimentPositive, Negative and Class](https://drive.google.com/uc?export=view&id=1-XJOO6RZkZ-wRX8D8H8Td4Tu__ryaaYO)

### Transformation

* Features used:
urgency, relevance, sentimentClass, sentimentNegative, sentimentNeutral, sentimentPositive, noveltyCount3D, noveltyCount5D, noveltyCount7D, volumeCounts3D, volumeCounts5D, volumeCounts7D, returnsOpenPrevMktres1, returnsClosePrevRaw1, returnsOpenPrevRaw10, returnsOpenPrevMktres10

* New features added:
close_open_diff - Simply the difference between open and close values.

* Timeseries data conversion:
Keras Timeseriesgenerator was being used to generate some sequence of data, in a span of continues 10 days and produce batches for training/validation.

![RNN input](https://cdn-images-1.medium.com/max/1600/1*v5_QpzkQfufVogeCY9eaOw.png)

https://cdn-images-1.medium.com/max/1600/1*v5_QpzkQfufVogeCY9eaOw.png

Data preprocessing notebook can be found [here](https://colab.research.google.com/drive/10TcWOObFY1SIeUjEnVPw-256NJZhYDGo).

This is a shared notebook that contains all the transformations we have done so far. We have taken apple stock prices and market news data, merged them together and took the most important features we think will benefit the learning process.

Those features are 'urgency', 'relevance','sentimentClass', 'sentimentNegative', 'sentimentNeutral', 'sentimentPositive','noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts3D', 'volumeCounts5D', 'volumeCounts7D', 'returnsOpenPrevMktres1', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw10', 'returnsOpenPrevMktres10' . And the target is taken to be 'returnsOpenNextMktres10'.

We chose open prices rather than close prices because we think that open prices react to news that have aggregated during the last evening more.

#### The open price for the day
![Times series plot for the open price](https://drive.google.com/uc?export=view&id=1--FjP3doQBKPB5MdWQUNkPcTgZIB8bMX)

We also created a new feature **'close_open_diff'**, which is the ratio of close price to open price (close_open_diff = close/open). 

#### Bootstrap plot on mean, median and mid-range statistics of the close_open_diff feature (samples - 1000, size - 50)
![Boostrap plot of close_open_diff feature](https://drive.google.com/uc?export=view&id=1EXOd5DlXTiKblEgu6T6HjzUp146Kk6Pg)

#### Correlation plot of the selected features
![Correlation plot of the selected features](https://drive.google.com/uc?export=view&id=1gI8aRls5pI23yi4Mo-zutGqtS_5Ypyfc)

#### Time series plots for all the selected features
![Ts plots for the selected features](https://drive.google.com/uc?export=view&id=1-s9pGIQ-tPleU5-yJyzlnIDyVjaoD74d)

### The Model


* LSTM figure
* models we tried
* the final model

Even though our task is regression and we should predict to which direction the stock price will go and by how much, in the evaluation of the models we use the accuracy measure for classification and confusion matrix. This cannot encompass the complete measurment of the model, but will give us some result. Additionally, we will do some particlar analysis to find the best model.

#### Random Forrest Regressor

One good model to start with would be the random forrest regressor. This has proven to give quite good results in almost any task and is not hard to implement and check.

Here estimated 2 depth model where we the data points separately and train the model on each data point. Loss function is taken to be "MSE".
The implementation of this task and the parameters can be found in [this notebook](https://colab.research.google.com/drive/1fOZa_7uhhSEGQo9AQHkYwdZk7OAaBMGX#scrollTo=Se4ZC7rmAyXN)

Here we have a classification accuracy of 54.4%. This is not bad, if we take into account the fact that we are dealing with stock markets and price predictions. At least it behaves better than random. 
But let's look at the predicted and actual returns.
The minimum validation loss is 0.0006

![Random Forrest. predicted vs actual](https://drive.google.com/uc?export=view&id=1dvCyjiBVVJWUDg2WhbFRgcvDouBlKlFh)

Here we see that the model tries to minimize MSE just simply flattening the predictions. In a complete random environment this would be the optimal choice, here we see that the model did not get much of an insight of the data, thus predicts some average values, even though the accuracy is more than average, but it does not estimate the size of the stock price movement properly.


#### Fully Connected Neural Network

Another stage of model estimation is already in the Deep learning environment. We decided to start with Fully connected net.
After carefully tunning the parameters, we took 4 dense layers one afer another and added dropout layers in between. In the end the activation layer with "tanh" as activation function, the loss is "MSE" and the optimizer is "Adam".
The notebook can be observed [here](https://colab.research.google.com/drive/1dgt8Av_ThIOCrIMfq6QnpvkrAOV2-gHr#scrollTo=mTZrr4bmlSMk)

Here we got around 56% accuracy in the classification sense. Better than random forrest in that sense.
The minimum validation loss is 0.0037.

Let's look at the predicted and actual returns.

![Fully Connected Neural Network. predicted vs actual](https://drive.google.com/uc?export=view&id=1TahbhAv9Bm-l-nkwSzKEgTKTVp0CU0vw)

Now, the network seems to catch some data insight, and the predictions are not as flat as they would be in random forrest case.
But still, the amplitude of the predictions do not exceed a certain limit, whereas the actual returns are quite volative.




#### LSTM network

Based on the nature of time series and the task, we need a network that will capture not only a current input features, but also previous values. This is based on the fact, that in finance stock prices react to not only yesterday's news, but also the news of previous days can have some overlasting impact. Moreover, the price of yesterday can affect on the price of today, even in some cases the prices can play a big role for a longer period of time. There are even several theories of financial time series that try to explain this(e.g. mean reverting time series, persistent, non-persistent series). 

Now, we made an LSTM layer in the front of the network and followed with 3 dense layers. Again, we added dropout layers in between all connections and activation in the end with "tanh" function. "MSE" is the loss function and "Adam" is the optimizer.

The network's implementation can be found in [this notebook](https://colab.research.google.com/drive/1WqVP-4bfY7v0rrE1Ag8d7zMSstIYY5eN#scrollTo=XkpaDOwykXlo)

The accuracy of the classification part is around 53%. This is not the best result, but in out opinion this network is capable of predicting the size and direction together better than the others.
We came to this point by looking at the graph of predictions.

![LSTM Neural Network. predicted vs actual](https://drive.google.com/uc?export=view&id=1NFUr2C8dPwxqF_4hwkbsjnIDP06ZY9ca)

Here we see that the oscilations are more or less from the same distribution and the behaviour of the predicted and actual time series are quite similar. This means that the network got some insight about the data.

The minimum validation loss is 0.00015.






The implementation of the neural networks we tried is also done in a shared notebook which can be accessed [here](https://colab.research.google.com/drive/1dgt8Av_ThIOCrIMfq6QnpvkrAOV2-gHr).

We tried different LSTM network with some feed forward part added in front of it, we also tried several combinations of LSTM networks.

Also parameter tunning is done.

### Results


* predicted vs actual
* compare to other ML models

As described on the models section, we are choosing the LSTM network because even though it had lower accuracy in the classification sense, it had the lowest validation loss, it was at least 4 times lower than random forrest and even more than the fully connected network.

This means that the size predictions will be better, and even though we get right predictions in 53 cases out of 100, taking into account the size prediction precision, in a long run the network can theoretically be used in an algorithmic trading environment.


![CM](https://drive.google.com/uc?export=view&id=1Ch8ndaROtddE3qvozycQGDvvz47ULua5)

![ROC-curve](https://drive.google.com/uc?export=view&id=1BYtbdcELF9zay0lBHButelvHO6ZN4ylM)

#### Kaggle Leaderboard

![Kaggle Leaderboard](https://drive.google.com/uc?export=view&id=1BIIkIffVFrDAdJu3iF_GqqgJBu8k-Ce3)

Mean score: 0.40454 (2019-06-10 14:51)

### Conclusion

Neural networks should not be the first solution for predictive models. They show good results but compared to the other methods, neural networks are hard to interprete. Further, deciding a neural network type also plays a pivotal role when designing a network. In our scenario RNN performed way better than FFN. Despite RNNs have shown great promise in many NLP tasks it can also be used on time series data as predictions are depends on predictions in the previous time stamp.

Ultimately, we have chosen Keras over PyTorch for heaps of reasonings but more chiefly we speculate that Keras is way matured compared to PyTorch in various number of approaches.


### Contributions

- Erki Aun - 
- Hindrek Teder - 
- Karen Danielyan - 
- Rudayasekaran PS - 
