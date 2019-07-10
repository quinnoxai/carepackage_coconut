import pandas as pd
import numpy as np
import final_detection_model as fdm

#from final_detection_model import main_model
#import app_test1 as atm
from sklearn.linear_model import LinearRegression

def sec_model():
    sec_df = pd.read_csv("C:/Users/ShivamM/Desktop/validation/carePackage.csv")
    #print(sec_df)

    X = sec_df.drop('actual_nuts',axis =1)
    Y = sec_df['actual_nuts']

    print("columns",X.columns)

    reg = LinearRegression().fit(X, Y)
    print("Score:",reg.score(X, Y))
    print("coeff;",reg.coef_)
    print("intercept",reg.intercept_)

    green,brown,tree_no = fdm.classification()
    print("Hello")
    print(green)
    print(brown)
    #tree_no = fdm.post()
    green = int(green)
    brown = int(brown)
    tree_no = int(tree_no)
    #brown = fdm.count_brown_new
    pred_count = reg.predict(np.array([[1, green, brown]]))
    print("Tree No:",tree_no)
    print("Final count",pred_count)
    mean_intercept = reg.intercept_/2
    pred_green = green + mean_intercept
    pred_brown = brown + mean_intercept
    green_coconuts=int(pred_green)
    brown_coconuts=int(pred_brown)
    print("pred_green",green_coconuts)
    print("pred_brown",brown_coconuts)

    final_matured_nuts = green_coconuts+brown_coconuts
    print("final_matured_nuts",final_matured_nuts)
    sec_df_new = pd.read_csv("C:/Users/ShivamM/Desktop/validation/Book1.csv")
    print("sec_df_new",sec_df_new)
    print("validation",sec_df_new.columns)
    actual_count_nuts = sec_df_new[sec_df_new['tree_no'] == tree_no]['actual']
    print("actual_count_nuts!!!",actual_count_nuts)
    print("actual_count_nuts type!!!",actual_count_nuts)
    #actual = 49
    mape = np.mean(np.abs(actual_count_nuts - final_matured_nuts)/actual_count_nuts)

    print("actual_count_nuts",actual_count_nuts)
    print("final_matured_nuts",final_matured_nuts)
    print(mape)
    print("Acc %", (1 - mape)*100)
sec_model()
