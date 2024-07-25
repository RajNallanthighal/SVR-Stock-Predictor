# SVR-Stock-Predictor

**See the model for yourself! -> https://rajnallanthighal.pythonanywhere.com/**

Support Vector Regressor (SVR) built from scratch (only with numpy) to predict the tomorrow (next valid trading day) closing price of a specific stock. Currently fitted and tested to the S&P500, implemented with epsilon-insensitive loss function and linear kernel.

After testing and hyperparameter tuning, model runs with an average mean absolute error of 0.01959663% and 56.667% success rate in prediction price direction (ie: up or down). 

The goal of the project was to create a machine learning model from scratch to predict the S&P500 closing price each following trading day, thus, I see these results as a success. In the future, I plan to increase the model's accuracy to be on-par with industry model standards (ie: 70-80% real-world accuracy), and I've listed my planned changes below.

Future Improvements:
  - Change Decision Function: Polynomial and gaussian kernels were either a) far too resource intensive to proceed with testing or b) had to be ran at a lower "depth," which did not accurately converge. Thus, a linear kernel was implemented for this model, although I definitely want to change this going forward so that the model can better recognize non-linear trends in the stock data.
  - More Rigorous Parameter Testing: The number of parameters I could test, as well as the range of values of them I could test them with, was severely restricted by my current hardware. A few "assumptions" were made throughout the tuning process, which I hope to learn about and test more thoroughly with more powerful hardware in the future.
  - Better Feature Engineering: The data set is provided by the yfinance API, which includes basic details about the stock. Access to higher-frequency data about a specific stock or other market trends across the globe could help guide the model's predictions.
  - Inclusion of Sentiment Analysis: As what dictates stock market is more than just statistics and profit, sentiment analysis on national/global news and media could help the model's predictions by teaching it to capture nuanced trends beyond the numbers.

*Thanks so much!*
