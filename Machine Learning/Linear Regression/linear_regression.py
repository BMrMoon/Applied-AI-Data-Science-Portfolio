# Imports
import pandas as pd
import numpy as np


### Task
## 1- Construct the linear regression model equation using the given bias and weight. Bias = 275, Weight = 90 (ŷ = b + wx)
# Equation:
# ŷ = 275 + 90x
## 2- Based on the model equation you created, estimate the salary for all years of experience in the table.
## 3- Calculate the MSE, RMSE, and MAE scores to evaluate the model’s performance.


# Functions
def linear_regression(b, w, exp):
    func = b + w*exp
    return func

def main():
    exp = np.array([5, 7, 3, 3, 2, 7, 3, 10, 6, 4, 8, 1, 1, 9, 1])
    salary =  np.array([600, 900, 550, 500, 400, 950, 540, 1200, 900, 550, 1100, 460, 400, 1000, 380])
    df = pd.DataFrame(data={'exp': exp, 'salary':salary})

    ### Task

    df["expected_salary"] = df["exp"].apply(lambda e: linear_regression(275, 90, e))
    df["error"] = (df["salary"] - df["expected_salary"])
    df["squared_error"] = np.square(df["error"])
    df["absolute_error"] = np.abs(df["error"])

    mse = df["squared_error"].mean()
    rmse = (df["squared_error"].mean())**0.5
    mae = df["absolute_error"].mean()
    print('MSE: ', mse)
    print('RMSE: ', rmse)
    print('MAE: ', mae)



if __name__ == '__main__':
    main()