# Using-ML-DeepL-to-forecasting-stock-prices
 Forecasting the Vietnamese pharmacy  company’s stock prices using Statistic,  Machine Learning and Deep Learning  Models.
 
[FinalResearch.pdf](https://github.com/huuquangg/Using-ML-and-DeepL-to-forecast-stock-prices/files/14582635/FinalResearch.pdf).

[FinalResearch GGDrive](https://drive.google.com/drive/u/0/folders/1NCmJSmATSgjGKhYuA_Vb38rZV4pLnJGJ).


# ABSTRACT
This paper investigates the application of various statistical models and machine learning algorithms in forecasting the stock prices of Vietnamese pharmaceutical companies. Leveraging a diverse array of methodologies, including Linear Regression (LR), Support Vector Regression (SVR), Long Short-Term Memory (LSTM) networks, Autoregressive Integrated Moving Average (ARIMA), Autoregressive Integrated Moving Average with Exogenous Variables (ARIMAX), K-Nearest Neighbors (KNN), and Boosting LSTM, our study aims to provide comprehensive insights into the predictive capabilities of these models within the dynamic and complex pharmaceutical stock market landscape. Additionally, the findings contribute to the ongoing discourse on applying statistical models and machine learning algorithms in predicting stock prices
 within emerging markets, particularly in the context of 3 Vietnamese Pharmaceutical Joint Stock Company (OPC, Vimedimex, and IMEXPHARM).
 
 **Keyword:** <i> Stock price, forecasting, LR, SVR, LSTM, ARIMA, ARIMAX, KNN, Boosting LSTM, HMM,pharma companies, OPC, IMP, VMD.</i>

# **Methodology**
We have decided to approach this problems by introducing to you 8 algorithms below includes Statistical Models, Machine Learning and Deep Learning Models.

### A. LINEAR REGRESSION ( LR )
A multiple linear regression model has the form Eq.(19):
$$x = {b_0+b_1X_1+b_2X_2\ +\ ...+\ b_kX_k+e}$$

### B. AUTOREGRESSIVE INTEGRATED MOVING AVERAGE (ARIMA)

- AutoRegressive - AR(p) is a regression model with lagged I, until p-th time in the past, as predictors. Eq.\eqref{eq}
$$AR(p)={\ \alpha\ +\ \beta_1y_{t-1}\ +\ \beta_2y_{t-2}+\ \cdots\ +\ \beta_py_{t-p}+\ \varepsilon}$$


- Integrated I(d) - The difference is taken d times until the original series becomes stationary. Eq.\eqref{eq}
$${I\left(d\right)=\ \Delta y_{t}= y_{t}-y_{t-1}.}$$

- Moving average MA(q) - A moving average model uses a regression-like model on past forecast errors.\eqref{eq}
$$MA(q)={\ \alpha\ +\ \theta_1\varepsilon_{t-1}\ +\ \theta_2\varepsilon_{t-2}\ +\ \cdots\ +\ \theta_p\varepsilon_{t-p}\ +\ \varepsilon.}$$
- Regression equations for ARIMA(p,d,q): Eq.\eqref{eq}
$${ARIMA\left(p,q,d\right)=\ AR\left(p\right)+\ I\left(d\right)+\ MA\left(q\right)\ \ \ \}$$

### C. SUPPORTVECTORREGRESSION(SVR)

![image](https://github.com/huuquangg/Using-ML-and-DeepL-to-forecaste-stock-prices/assets/98322281/abe8da63-1fdd-4cb1-b2ae-ec735cbb0bd5)

### D. LONGSHORT-TERMMEMORY(LSTM)

![image](https://github.com/huuquangg/Using-ML-and-DeepL-to-forecaste-stock-prices/assets/98322281/cf5c357f-a690-4601-9cf9-33091c220e07)


### E. ARIMA WITH EXOGENOUSVARIABLES(ARIMAX)

$$x_t={ARIMA\left(p,d,q\right)+b_1X_1\left(t\right)+\ ...+\ b_kX_k\left(t\right)+\ \varepsilon\ \left(t\right)}$$

### F. K-NEAREST NEIGHBORS (KNN)
K-Nearest Neighbors (KNN) is one of the simpler prediction/classification techniques: there is no model to be fit (as in regression).

![image](https://github.com/huuquangg/Using-ML-and-DeepL-to-forecaste-stock-prices/assets/98322281/a80959aa-b641-4e86-8fdc-a6f8bcd2b1b4)

### G. BOOSTING LSTM
![image](https://github.com/huuquangg/Using-ML-and-DeepL-to-forecaste-stock-prices/assets/98322281/697204ce-6b4b-44ad-8a99-b15588d1f77f)
 
### H. HIDDEN MARKOVMODEL(HMM)

The model’s observation is a three dimensional vector representing daily stock information,Eq.(19)

$$O_t=(\frac{close-open}{open}\ ,\ \frac{high-open}{open}\ ,\ \frac{open-low}{open}\ )$$

# IV. EXPERIMENT
The data-set is crawled by using vnstock · PyPI (v0.2.8.6) from Pypi library through DNSC stock market in 10th Dec, 2023.
## A. DATA DESCRIPTIVE
There are 3 data-set fetching 3 pharma company in Vietnam, includes OPC Pharmaceutical Joint Stock Company (OPC), 2468 rows; VimedimexPharmaceuticalJoint Stock Company (VMD) 2477 rows and IMEXPHARM Pharmaceutical Joint Stock Company (IMP) with 2467 rows from 2nd Jan, 2014 to 8th Dec, 2023.

| Companies                                  | OPC | VMD | IMP  | 
|--------------------------------------------|---  |---  |---   |
| Count                                      | 2,468 |  2,477 | 2,467 |
| Mean                                       | 16,553 |  18,719 | 38,546   |
| Standard deviation                         |5,180 |7,165 |16,621|
| Min                                        |7,470 |6,140 |10,790|
| 25%                                        |11,172 |14,950 |23,280|
| 50%                                        |16,700 |18,150 |34,890|
| 75%                                        |21,760 |21,000 |52,075|
| Max |27,490 |77,810 |79,290|
| Variance |26,841,383| 51,345,375 |276,290,079 |
| Skewness|-0.14240 |2.37885 |0.50166|
| Kurtosis|-1.35002 |11.85794|-0.86655|

![image](https://github.com/huuquangg/Using-ML-and-DeepL-to-forecaste-stock-prices/assets/98322281/d97cdfcd-67c6-4baf-b180-dbde505753b9)


###  OPC      
       Model            Ratio           RMSE         MAPE          MAE
                     6 -- 3 -- 1      1651.359       6.179      1618.750
        LR           7 -- 2 -- 1      1700.792       5.453      1254.153
                     8 -- 1 -- 1      1603.972       6.365      2218.027
                     6 -- 3 -- 1      5903.303      25.004      7342.278
       ARIMA         7 -- 2 -- 1      1623.326       6.185      1026.099
                     8 -- 1 -- 1      1175.016       3.855       702.727
                     6 -- 3 -- 1      4086.551      13.627      3081.360
        SVR          7 -- 2 -- 1      2695.950       6.981      1650.327
                     8 -- 1 -- 1       593.209       0.790       191.922
                     6 -- 3 -- 1       571.944       1.926       486.944
       LSTM          7 -- 2 -- 1       403.939       1.154       197.606
                     8 -- 1 -- 1       410.994       1.248       209.331
                   **6 -- 3 -- 1**   **200.118**   **0.641**   **142.276**
      ARIMAX         7 -- 2 -- 1       243.033       0.738       165.235
                     8 -- 1 -- 1       200.118       0.641       142.276
                     6 -- 3 -- 1      5691.622      24.026      5281.468
        KNN          7 -- 2 -- 1      1461.386       4.252       996.334
                     8 -- 1 -- 1      1192.508       2.444       586.232
                     6 -- 3 -- 1       724.186       2.229       554.786
      Boosting LSTM  7 -- 2 -- 1       592.866       1.544       339.519
                     8 -- 1 -- 1        737.7        1.878       359.912
                     6 -- 3 -- 1      4387.945      17.882      3899.961
        HMM          7 -- 2 -- 1      12242.473     54.110      12122.201
                     8 -- 1 -- 1      3054.832       9.839      2213.876

### VMD                                  
  --------------- ----------------- ------------- ----------- -------------
       Model            Ratio           RMSE         MAPE          MAE
                     6 -- 3 -- 1      9649.451      16.289      2418.075
        LR           7 -- 2 -- 1      12120.135     21.137      2283.258
                     8 -- 1 -- 1      5714.464      21.078      5916.466
                     6 -- 3 -- 1      9937.794      18.275      2548.507
       ARIMA         7 -- 2 -- 1      12457.71      22.682      2883.917
                     8 -- 1 -- 1      13172.876     55.797      15002.287
                     6 -- 3 -- 1      11438.434     12.914      4997.949
        SVR          7 -- 2 -- 1      14422.118     22.395      8324.811
                     8 -- 1 -- 1       735.555       1.864       460.992
                     6 -- 3 -- 1      1893.718       3.675       474.926
       LSTM          7 -- 2 -- 1      1766.212       3.334       333.847
                     8 -- 1 -- 1       706.622       2.307       308.219
                   **6 -- 3 -- 1**   **406.139**   **0.961**   **239.240**
      ARIMAX         7 -- 2 -- 1       474.067       1.172       309.271
                     8 -- 1 -- 1       484.918       1.514       356.366
                     6 -- 3 -- 1      9837.156      18.329      5614.338
        KNN          7 -- 2 -- 1      12302.448     22.017      7666.507
                     8 -- 1 -- 1      12951.406     54.788      11920.842
                     6 -- 3 -- 1       3295.73       5.526       894.255
     Boosting LSTM   7 -- 2 -- 1      4454.588       6.321      614.9697
                     8 -- 1 -- 1       767.626       2.476       505.563
                     6 -- 3 -- 1      9964.968      20.851      6136.134
        HMM          7 -- 2 -- 1      12427.107     26.853      8515.060
                     8 -- 1 -- 1      9884.588      31.466      8311.402

### IMP                                  
  --------------- ----------------- ------------- ----------- -------------
       Model            Ratio           RMSE         MAPE          MAE
                     6 -- 3 -- 1      14575.349     19.263      5784.508
        LR           7 -- 2 -- 1      18488.891     26.043      6404.986
                     8 -- 1 -- 1      10409.816     12.933      7164.764
                     6 -- 3 -- 1      15721.151     20.404      17852.991
       ARIMA         7 -- 2 -- 1      10833.288     14.187      5648.963
                     8 -- 1 -- 1      11875.223     16.744      13740.165
                     6 -- 3 -- 1      30489.105     40.429      25254.876
        SVR          7 -- 2 -- 1      31617.288     44.872      29261.537
                     8 -- 1 -- 1      2624.939       2.308      1541.152
                     6 -- 3 -- 1      3007.534       4.116      2573.013
       LSTM          7 -- 2 -- 1      1809.724       2.337      1054.136
                     8 -- 1 -- 1      1472.944       1.93        940.072
                   **6 -- 3 -- 1**   **566.148**   **0.723**   **386.874**
      ARIMAX         7 -- 2 -- 1       637.844       0.713       441.870
                     8 -- 1 -- 1       745.529       0.794       489.264
                     6 -- 3 -- 1      22355.075     30.649      18843.429
        KNN          7 -- 2 -- 1      14449.104      19.35      12783.089
                     8 -- 1 -- 1      8747.178      12.626      7198.577
                     6 -- 3 -- 1       2761.74       3.872      1693.198
      Boosting LSTM  7 -- 2 -- 1      2501.154       3.167      1511.618
                     8 -- 1 -- 1      2377.513       3.279      1337.611
                     6 -- 3 -- 1      23104.035     33.584      20096.218
        HMM          7 -- 2 -- 1      26204.521     40.751      25651.172
                     8 -- 1 -- 1      14171.327     16.793      10559.083

### C. INSIGHT

![image](https://github.com/huuquangg/Using-ML-and-DeepL-to-forecaste-stock-prices/assets/98322281/f9611fed-359a-48db-82ad-2243c206d43e)
![image](https://github.com/huuquangg/Using-ML-and-DeepL-to-forecaste-stock-prices/assets/98322281/048ddc40-0dce-4692-becf-ac7875b93642)
![image](https://github.com/huuquangg/Using-ML-and-DeepL-to-forecaste-stock-prices/assets/98322281/1512acee-92c5-47aa-9c5b-8fce3b9856f2)

![image](https://github.com/huuquangg/Using-ML-and-DeepL-to-forecaste-stock-prices/assets/98322281/fe3cdb59-871e-46f7-86f2-963c31142f14)
![image](https://github.com/huuquangg/Using-ML-and-DeepL-to-forecaste-stock-prices/assets/98322281/9c3dee6f-a0f7-435b-9fea-687e638abb02)

As you can see in table 3,4 and 5 
ARIMAX has most less difference between actual value andpredict (RMSE: 200.118vnd,MAPE: 0.641%,MAE: 142.276vnd) value at ratio(6-3-1) for both 3companies.
LSTMModelandBoost ing LSTM model rank 2nd (Ratio:8-1-1,RMSE:767.626vnd, MAPE: 2.307%,MAE: 380.219vnd) and 
3rd(Ratio: 8-1 - 1,RMSE: 767.626vnd,MAPE: 2.4%,MAE: 505.563vnd) respectively.
While ARIMA has big gap between 2compared values (approximately 13k vnd and percent errormostly 56%).














