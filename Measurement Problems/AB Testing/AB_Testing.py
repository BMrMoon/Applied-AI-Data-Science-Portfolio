# Imports
import pandas as pd
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 500)


# Functions
def test_result(p, thr):
    if p <= thr:
        print('H0 is rejected, there is a significant difference.')
        return False
    else:
        print('H0 is accepted, there is not a significant difference.')
        return True

def select_test(df_control, df_test, thr):
    normal_test_stat_control, normal_test_pvalue_control = shapiro(df_control)
    normal_test_stat_test, normal_test_pvalue_test = shapiro(df_test)
    var_stat, var_p = levene(df_control, df_test)
    if (normal_test_pvalue_control and normal_test_pvalue_test) > thr:
        if var_p <= thr:
            test_stat, pvalue = ttest_ind(df_control, df_test, equal_var=False)
            return test_stat, pvalue
        if var_p > thr:
            test_stat, pvalue = ttest_ind(df_control, df_test, equal_var=True)
            return test_stat, pvalue
    else:
        test_stat, pvalue = mannwhitneyu(df_control, df_test)
        return test_stat, pvalue

def AB_Testing():
    ### Task 1: Data Preparation and Analysis
    ## Step 1: Read the dataset ab_testing_data.xlsx, which contains the control and test group data. Assign the control and test group data to separate variables.
    df_control = pd.read_excel("./AB_Testing/ab_testing.xlsx", sheet_name="Control Group")
    df_test = pd.read_excel("./AB_Testing/ab_testing.xlsx", sheet_name="Test Group")
    df_control['group'] = 'control'
    df_test['group'] = 'test'
    print(df_control.shape)
    print(df_test.shape)
    control_group = df_control.copy()
    test_group = df_test.copy()

    ## Step 2: Analyze the control and test group data.
    print('Control Group Dataset Insight: \n', control_group.info())
    print('Control Group Descriptive Statistics: \n', control_group.describe().T)
    print('Test Group Dataset Insight: \n', test_group.info())
    print('Test Group Descriptive Statistics: \n', test_group.describe().T)

    ## Step 3: After completing the analysis, combine the control and test group data using the concat method.
    df = pd.concat([control_group, test_group], axis=0).reset_index()

    ### Task 2: Defining the Hypothesis of the A/B Test
    ## Step 1: Define the hypothesis.
    # H0 : M1 = M2, p>0.05
    # H1 : M1!= M2
    # H1: m1 == m2
        # Check the distribution
        # Check the variances

    ## Step 2: Analyze the average purchase (revenue) values for the control and test groups.
    print('Average purchase (revenue) values: \n', df.groupby('group').agg({'Purchase': 'mean'}))

    ## Task 3: Performing the Hypothesis Test
    test_stat_control, pvalue_control = shapiro(df[df['group'] == "control"]['Purchase'])
    print('Test Stat = %.4f, p-value = %.4f' % (test_stat_control, pvalue_control))
    test_stat_test, pvalue_test = shapiro(df[df['group'] == "test"]['Purchase'])
    print('Test Stat = %.4f, p-value = %.4f' % (test_stat_test, pvalue_test))
    test_stat_var, pvalue_var = levene(df[df['group'] == "control"]['Purchase'],
                           df[df['group'] == "test"]['Purchase'])
    print('Test Stat = %.4f, p-value = %.4f' % (test_stat_var, pvalue_var))

    ## Step 2: Based on the results of the normality assumption and variance homogeneity tests, choose the appropriate statistical test.
    test_stat, pvalue = select_test(df[df['group'] == "control"]['Purchase'], df[df['group'] == "test"]['Purchase'], 0.05)

    ## Step 3: Based on the p_value obtained from the test result, interpret whether there is a statistically significant difference between the average purchases of the control and test groups.
    print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
    test_result(pvalue, 0.05)







def main():
    AB_Testing()

if __name__ == '__main__':
    main()