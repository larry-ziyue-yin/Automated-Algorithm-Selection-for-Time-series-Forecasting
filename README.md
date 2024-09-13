# Automated Algorithm Selection for Time-series Forecasting
## Motivation
The accurate prediction of the stock price has always been a challenging task for data scientists. However, the deep learning models provide us with new potential to enhance the performance of stock price prediction.

Recently, there have been advancements in the field of time series forecasting, where models like Generative Pretrained Hierarchical Transformer (GPHT) are introduced, which are pre-trained on mixed datasets to improve the generalization of it in diverse scenarios (Liu et al., 2024, p. 2004). However, algorithm selection also significantly impacts the performance of such predictive models.

In this context, Misir & Sebag (2017) on algorithm selection introduces us to a novel approach named Alors. It uses collaborative filtering to recommend the most suitable algorithms for different problems. This method is particularly relevant to our project, as it addresses the challenge of selecting the optimal algorithm for stock price prediction. And the choice of algorithm can drastically affect the outcomes.

By integrating Alors with GPHT, this project aims to bridge the gap between time series forecasting techniques and their application to stock price prediction, with a particular focus on the role of feature extraction and algorithm selection.

## Objectives
There are three main objectives:

1.	**Measure the Effectiveness of Using GPHT in SP Forecasting:** According to hierarchical architecture and pretraining on mixed datasets which Liu has suggested. The value of the architectures presented in (2024) is that they can model highly complex temporal dependencies and non-linearities existing inside financial time series data, respectively.
2.	**Enhance Predictive Performance by Using Integrate Algorithm Selection Techniques:** Recognizing the critical role of algorithm selection in improving model outcomes, this project will incorporate the Alors system. Alors system was developed by Misir & Sebag (2017), this system could systematically identify and apply the most suitable algorithms for different subsets of stock price data. Such integration could optimize the model selection process. Moreover, this integration could also ensure that the chosen models are well-suited to the specific characteristics of the financial time series under investigation.
3.	**Develop a Comprehensive Framework for Stock Price Prediction:** Our goal for the project is to develop a robust framework that combines advanced time series forecasting techniques with algorithm selection methods. This framework will be designed to improve the accuracy of stock price predictions as well as provide a flexible and scalable approach that can be adapted to various financial datasets and market conditions.

In conclusion, this project aims to contribute to financial time series forecasting by advancing the application of different machine learning models. We would also enhance the decision-making process through an informed algorithm selection method.

## Methods
To address the outlined objectives, this project will follow the structured methods below:

1.	**Data Collection and Preprocessing:** We will start the project by compiling a diverse set of financial time series datasets, focusing on various stock indices, individual stock prices, and other relevant financial indicators. We will also split the datasets into training, validation, and testing sets to ensure robust model evaluation in this stage.
2.	**Implementation of the Generative Pretrained Hierarchical Transformer (GPHT):** As stated by Liu et al. (2024), the core of the forecasting methodology contains implementing the GPHT model. In this project, the models will be pre-trained on a mixed dataset which consists of multiple time series from different financial contexts. Apart from that, we will also conduct a fine-tuning process, during which we use the specific stock price datasets to tailor the model’s predictive capabilities to the financial domain. The GPHT model’s hierarchical structure will be utilized to construct both short-term and long-term dependencies within the given time series data.
3.	**Algorithm Selection Using the Alors System:** We would leverage the Alors system developed by Misir & Sebag (2017) to enhance forecasting performance. The system will be employed to select the most appropriate algorithms for different subsets of the stock price data. The collaborative filtering approach will be utilized to recommend algorithms based on past performance on similar data instances. In this step, we will apply multiple algorithms to a subset of the data, evaluate their performance metrics, and use Alors to identify the optimal algorithmic configuration for the entire dataset.
4.	**Integration of Forecasting and Algorithm Selection:** With all the methods above done, we will then develop an integrated framework that combines the predictive power of GPHT with the adaptive capabilities of the Alors system. This integration creates a pipeline, where the GPHT model generates initial forecasts, and the Alors system dynamically selects and applies suitable algorithms to refine the forecasts. Hopefully, this framework will be designed to allow for iterative improvements, where algorithm selection is continuously informed and updated by the performance of previous forecasts.
5.	**Evaluation Metrics and Comparative Analysis:** Several evaluation metrics will be proposed to measure the performance of the integrated framework on the test set. Such performance will then be compared with initial benchmarks like ARIMA and other deep learning models. Evaluation metrics include Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and other relevant financial forecasting metrics that will be proposed by us. Such comparative analysis will provide insights into the relative strengths of the combined approach and its potential advantages over existing methods.

Through these methodological steps, our project aims to deliver a comprehensive, validated approach to stock price prediction, which leverages the latest advancements in time series forecasting and algorithm selection.

## Acknowledgements
### Datasets
[https://larry-better-me.notion.site/Datasets-211bf6cc1d6440ddbbd74af5e8f84a37](https://larry-better-me.notion.site/Datasets-211bf6cc1d6440ddbbd74af5e8f84a37)

### Papers
[Generative Pretrained Hierarchical Transformer (GPHT)](https://dl.acm.org/doi/pdf/10.1145/3637528.3671855)

[Alors: An algorithm recommender system (Alors)](https://www.sciencedirect.com/science/article/pii/S0004370216301436)

### Codes
[Generative Pretrained Hierarchical Transformer (GPHT)](https://github.com/icantnamemyself/GPHT)

[Alors: An algorithm recommender system (Alors)](https://www.lri.fr/~sebag/Alors/src/)

## References
Liu, Z., Yang, J., Cheng, M., Yucong, L., & Li, Z. (2024). Generative Pretrained Hierarchical Transformer for Time Series Forecasting. _Time Series Forecasting_.

Misir, M., & Sebag, M. (2017). Alors: An algorithm recommender system. _Artiﬁcial Intelligence_.
