## Two Sigma: Using News to Predict Stock Movements

The project was implemented as a part of Neural Networks (LTAT.02.001) course in University of Tartu. The original Kaggle competition can be found [here](https://www.kaggle.com/c/two-sigma-financial-news/data).

Collaborators:
- Erki  Aun
- Hindrek Teder
- Karen Danielyan
- Rudesh Sekaran

### Goal

In this project, we predict future stock price returns based on two sources of data:
- Market data (2007 to present) provided by Intrinio.
- News data (2007 to present) provided by Thomson Reuters.

### Abstract

### Datasets

TODO: add figures for features

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

### Transformation

TODO:
* what features we used and dropped
* new features
* timeseries data conversion

Data preprocessing notebook can be found [here](https://colab.research.google.com/drive/10TcWOObFY1SIeUjEnVPw-256NJZhYDGo).

This is a shared notebook that contains all the transformations we have done so far. We have taken apple stock prices and market news data, merged them together and took the most important features we think will benefit the learning process.

Those features are 'close_open_diff', 'urgency', 'relevance','sentimentClass', 'sentimentNegative', 'sentimentNeutral', 'sentimentPositive','noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts3D', 'volumeCounts5D', 'volumeCounts7D', 'returnsOpenPrevMktres1', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw10', 'returnsOpenPrevMktres10' . And the target is taken to be 'returnsOpenNextMktres10' .

We chose open prices rather than close prices because we think that open prices react to news that have aggregated during the last evening more.

The visualization of the data and some interesting plots can also be found on that notebook.

### The Model

TODO:
* LSTM figure
* models we tried
* the final model

The implementation of the neural networks we tried is also done in a shared notebook which can be accessed [here](https://colab.research.google.com/drive/1dgt8Av_ThIOCrIMfq6QnpvkrAOV2-gHr).

We tried different LSTM network with some feed forward part added in front of it, we also tried several combinations of LSTM networks.

Also parameter tunning is done.

### Results

TODO:
* predicted vs actual
* confusion matrix
* AUC plot
* compare to the Kaggle results
* compare to other ML models

The preliminary results can be found on the same notebook as the neural network is described. there are some visualizations showing the predicted and actual returns of Apple Inc. The notebook contains sufficient comments to read through. 

We also did some accuracy measurment just for simplicity assuming is we have done classification with the network output, then what percentage of the predictions would have the same direction. We got something closer to 60%. We need to try combining two tranches of data with two network and have dense layer in the end. We hope it will give better results.

### Conclusion

Neural networks should not be the first solution for predictive models. They show good results but compared to the other methods, neural networks are hard to interprete.
