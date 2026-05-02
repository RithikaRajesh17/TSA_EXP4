# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 02-05-2026



## AIM:
To implement ARMA model in python.
## ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
## PROGRAM:
```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
data=pd.read_csv("/content/POPH.csv")
x=data['value']
N=1000
plt.rcParams['figure.figsize']=[12,6]
plt.plot(x)
plt.title("Originaldataset")
plt.show()
plt.subplot(2, 1, 1)
plot_acf(x, lags=len(x)//4, ax=plt.gca())
plt.title('ACF')
plt.subplot(2, 1, 2)
plot_pacf(x, lags=len(x)//4, ax=plt.gca())
plt.title('PACF')
plt.tight_layout()
plt.show()
arma11_model = ARIMA(x, order=(1, 0, 1)).fit()
phi1 = arma11_model.params['ar.L1']
theta1 = arma11_model.params['ma.L1']
ar1 = np.array([1, -phi1])
ma1 = np.array([1, theta1])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1)')
plt.show()
plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()
arma22_model = ARIMA(x, order=(2, 0, 2)).fit()
phi1 = arma22_model.params['ar.L1']
phi2 = arma22_model.params['ar.L2']
theta1 = arma22_model.params['ma.L1']
theta2 = arma22_model.params['ma.L2']
ar2 = np.array([1, -phi1, -phi2])
ma2 = np.array([1, theta1, theta2])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2)')
plt.show()
plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()

```



 ## OUTPUT:

### ORIGINAL DATASET:

<img width="990" height="528" alt="image" src="https://github.com/user-attachments/assets/8acf36c6-d7cb-4ab5-b413-3e6ef224c4dd" />


<img width="1198" height="590" alt="image" src="https://github.com/user-attachments/assets/d281aff0-0b80-40ea-9b2a-bbb502e76e0a" />



## SIMULATED ARMA(1,1) PROCESS:

<img width="988" height="528" alt="image" src="https://github.com/user-attachments/assets/d13102dc-06f8-4003-aef2-73108fe4ecb5" />


### AutoCorrelation:

<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/909bea49-4d63-4ad5-8e22-e45f6fb2b98b" />

### Partial AutoCorrelation:


<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/340abd28-1c93-4436-8cbc-3d5dbb4b84b8" />





## SIMULATED ARMA(2,2) PROCESS:


<img width="997" height="528" alt="image" src="https://github.com/user-attachments/assets/fb9e515f-6565-4040-8f9a-429d37cc1a2f" />


### AutoCorrelation:


<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/29788c21-38c0-4ea8-9d53-8e50ba7471cd" />


### Partial AutoCorrelation:


<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/79c9fd00-f27f-4e11-a2a8-6734b0c470c3" />


## RESULT:
Thus, a python program is created to fir ARMA Model successfully .
