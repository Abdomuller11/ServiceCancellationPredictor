from typing import TYPE_CHECKING
# void functions
from sklearn.preprocessing import MinMaxScaler  # scaling each feature to a given range
from sklearn.preprocessing import LabelEncoder  # encode labels
from sklearn.model_selection import train_test_split  # train and test our dataset by spliting
import pandas as pd  # data structures and data analysis tools
import numpy as np  ## linear algebra            #scientific computing in Python(linear algebra...)
import matplotlib.pyplot as plt  # plotting, opens figures on your screen,data processing, CSV...
from tkinter import *
from sklearn.metrics import \
    confusion_matrix  # (The confusion matrix is a matrix used to determine the performance of the classification models for a given set)compute confusion matrix to evalute accuracy matrix
from sklearn.metrics import accuracy_score  # output accuracy between actual and predict set of our ai data
import seaborn as sns  # data visualization library based on matplotlib.
from sklearn.linear_model import LogisticRegression  # LogisticRegression algorithm
# from sklearn.linear_model import LinearRegression #linear regression algorithm
from sklearn.svm import SVC  # Support Vector Classification algorithm
from sklearn.naive_bayes import MultinomialNB  # implements the naive Bayes algorithm
from sklearn.tree import DecisionTreeClassifier  # decision tree classifier algorithm
from sklearn.ensemble import RandomForestClassifier  # for Random Forest algorithm
from sklearn.neighbors import KNeighborsClassifier  # K Nearest Classifier algorithm
import plotly.express as px
from PIL import Image
# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def dataoveriew(df):
    print('Overview of the dataset:\n')
    print('Number of rows: ', df.shape[0])
    print("\nNumber of features:", df.shape[1])
    print("\nData Features:")
    print(df.columns.tolist())
    print("\nMissing values:", df.isnull().sum().values.sum())
    print("\nUnique values:")
    print(df.nunique())  # values
    return "\n"


def displaySpecificRow(boolean):
    if boolean == True:
        inp0 = int(input("enter start index: "))
        inp1 = int(input("enter end index: "))
        return dataset.iloc[inp0:inp1]
    return "\n"


def displaySpecificColumn(boolean):
    if boolean == True:
        inp0 = input("enter column name: ")
        return dataset[inp0]
    return "\n"


try:

    dataset = pd.read_csv(r"W:\AinShams uni\Project\resourses\om.csv")  # to read .csv files
    dataoveriew(dataset)

    inp0 = input("Do yo want listing of column names, (quick describe of numeric data):[y/n] ").lower()
    if inp0 == "y":
        print(dataset.columns, end="\n")
        print(dataset.describe())

    print(displaySpecificColumn(True))
    print(displaySpecificRow(True))

    inp1 = input("Do you want reading of top or bottom from dataset before starting:[y/n] ").lower()
    if inp1 == "y":
        inp2 = input("top or bottom:[t/b] ").lower()
        if inp2 == "t":
            print(dataset.head())
        else:
            print(dataset.tail())

    '''
    # def binary_map(feature):
    #     return feature.map({'Yes': 1, 'No': 0})
    # 
    # def ecode(fea):
    #     return  fea.map({{'Yes': 1, 'No': 0,"No internet service": 2}})
    '''
    dataset = dataset.replace({
        "gender": {"Male": 1, "Female": 0},
        "Partner": {"No": 0, "Yes": 1},
        "Dependents": {"No": 0, "Yes": 1},
        "PhoneService": {"No": 0, "Yes": 1},
        "MultipleLines": {"No": 0, "Yes": 1, "No phone service": 2},
        "InternetService": {"No": 0, "DSL": 1, "Fiber optic": 2},
        "OnlineSecurity": {"No": 0, "Yes": 1, "No internet service": 2},
        "OnlineBackup": {"No": 0, "Yes": 1, "No internet service": 2},
        "DeviceProtection": {"No": 0, "Yes": 1, "No internet service": 2},
        "TechSupport": {"No": 0, "Yes": 1, "No internet service": 2},
        "StreamingTV": {"No": 0, "Yes": 1, "No internet service": 2},
        "StreamingMovies": {"No": 0, "Yes": 1, "No internet service": 2},
        "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
        "PaperlessBilling": {"No": 0, "Yes": 1},
        "PaymentMethod": {"Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2,
                          "Credit card (automatic)": 3},
        "Churn": {"No": 0, "Yes": 1}

    })  # convert to numeric data to calc. and instead of unique func
    # customer id col. isnt useful as the feature for identification

    #

    dataset.drop("customerID", axis=1, inplace=True)

    # simpleHistogram:
    serv = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies']
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))  # create subplots with specific rows,cols
    for i, item in enumerate(serv):
        if i < 3:
            ax = dataset[item].value_counts().plot(kind='bar', ax=axes[i, 0], rot=0, color='#0009ff')  # calc values
        elif i >= 3 and i < 6:
            ax = dataset[item].value_counts().plot(kind='bar', ax=axes[i - 3, 1], rot=0, color='#9b9c9a')
        elif i < 9:
            ax = dataset[item].value_counts().plot(kind='bar', ax=axes[i - 6, 2], rot=0, color='#ec838a')
        ax.set_title(item)  # title
    plt.suptitle('simpleHistogram\n', horizontalalignment="center", fontstyle="normal", fontsize=24, fontfamily="serif")
    plt.show()  # display plots

    sns.heatmap(dataset.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.show()

    le = LabelEncoder()

    dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')
    # Fill the missing values with with the median value
    dataset['TotalCharges'] = dataset['TotalCharges'].fillna(dataset['TotalCharges'].median())

    plt.figure(figsize=(15, 10))
    corr = dataset.corr()
    sns.heatmap(corr, annot=True)  # annotation
    plt.show()


    def corrl(dataset, threshold):
        cols = []  # set of all names of corelated col
        corl_matrix = dataset.corr()
        for w in range(len(corl_matrix.columns)):
            for ww in range(w):
                if abs(corl_matrix.iloc[w, ww] > threshold):  # +,-
                    colname = corl_matrix.columns[w]  # getting name of col
                    cols.append(colname)
        return cols


    print("high corrlation column: ", corrl(dataset, 0.85))
    # therefore
    dataset.drop("MonthlyCharges", axis=1)

    x = dataset.drop("Churn", axis=1)
    y = dataset["Churn"]

    x_old = x

    # The Pandas drop() function in Python is used to drop specified labels from rows and columns. Drop is a major function used in data science & Machine Learning to clean the dataset.
    # The range of all features should be normalized so that each feature contributes approximately proportionately to the final distance, so we do feature scaling.
    # feature scaling The range of all features should be normalized so that each feature contributes approximately proportionately to the final distance, so we do feature scaling.
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1),
                          clip=False)  # create an obj(scaler) to fitting(compute min,max) then
    x = scaler.fit_transform(x)
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state=50, test_size=0.2)  # 20%
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import StratifiedKFold

    log = LogisticRegression()
    rfecv = RFECV(estimator=log, cv=StratifiedKFold(10, random_state=50, shuffle=True), scoring="accuracy")
    rfecv.fit(xTrain, yTrain)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.grid()
    plt.xticks(range(1, x.shape[1] + 1))
    plt.xlabel("Number of Selected Features")
    plt.ylabel("CV Score")
    plt.title("Recursive Feature Elimination (RFE)")
    plt.show()

    print("The optimal number of features(next time): {}".format(rfecv.n_features_))

    # now machine learning algorithms and its accuracy (prove by code.)
    # both train and test functions to prove requiremet ("You must prove by code that the chosen algorithm produces a higher accuracy than some of the other algorithms")
    # 1.LogisticRegression algorithm

    logReg = LogisticRegression(solver="liblinear", penalty="l1", tol=0.002, C=100)  # saga for big tables
    logReg.fit(xTrain, yTrain)  # x,y train

    y_pred0 = logReg.predict(xTest)  # predict based on xtest البداية يعم
    acc0 = accuracy_score(yTest, y_pred0)


    def trainLogisticRegression():
        print(f"Accuracy Training of LogisticRegression algo: {logReg.score(xTrain, yTrain)}")
        return "\n"


    def testLogisticRegression():
        # confusion mat
        cm = confusion_matrix(yTest, y_pred0)
        print(f"Accuracy testing of LogisticRegression algo: {acc0}")
        sns.heatmap(cm, annot=True, fmt="d")  # decimal
        plt.xlabel("Predicted")
        plt.ylabel("Truth")
        plt.show()
        return f"confusion matrix: {cm}"

        # 2.SVM


    svm = SVC(kernel="rbf", max_iter=-1)  # create obj
    svm.fit(xTrain, yTrain)
    y_pred1 = svm.predict(xTest)
    acc1 = accuracy_score(yTest, y_pred1)


    def trainSVM():
        print(f"Accuracy Training of SVM algo: {svm.score(xTrain, yTrain)}")  # score()
        return "\n"


    def testSVM():
        print(f"Accuracy testing of SVM algo: {acc1}")
        cm = confusion_matrix(yTest, y_pred1)
        sns.displot(x=dataset['Churn'], bins=2)
        plt.show()
        return f"confusion matrix: {cm}"

        # 3.DecisionTreeClassifier


    decTree = DecisionTreeClassifier(max_depth=20, random_state=50)
    decTree.fit(xTrain, yTrain)
    y_pred2 = decTree.predict(xTest)
    acc2 = accuracy_score(yTest, y_pred2)


    def trainTree():
        print(f"Accuracy Training of DecisionTree algo: {decTree.score(xTrain, yTrain)}")  # score()
        return "\n"


    def testTree():
        print(f"Accuracy testing of DecisionTree algo: {acc2}")
        cm = confusion_matrix(yTest, y_pred2)
        sns.heatmap(cm, annot=True, fmt="d")  # decimal
        plt.xlabel("Predicted")
        plt.ylabel("Truth")
        plt.show()
        return f"confusion matrix: {cm}"

        # 4.K Nearest Classifier (knn)


    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30)
    knn.fit(xTrain, yTrain)
    y_pred3 = knn.predict(xTest)
    acc3 = accuracy_score(yTest, y_pred3)


    def trainKnn():
        print(f"Accuracy Training of KNN algo: {knn.score(xTrain, yTrain)}")  # score()
        return "\n"


    def testKnn():
        print(f"Accuracy testing of KNN algo: {acc3}")
        cm = confusion_matrix(yTest, y_pred3)
        sns.heatmap(cm, annot=True, fmt="d")  # decimal
        plt.xlabel("Predicted")
        plt.ylabel("Truth")
        plt.show()
        return f"confusion matrix: {cm}"
        # 5.naive Bayes


    bayes = MultinomialNB(alpha=1.0, fit_prior=True)
    bayes.fit(xTrain, yTrain)
    y_pred4 = bayes.predict(xTest)
    acc4 = accuracy_score(yTest, y_pred4)


    def trainBayes():
        print(f"Accuracy Training of naive Bayes algo: {bayes.score(xTrain, yTrain)}")  # score()
        return "\n"


    def testBayes():
        print(f"Accuracy testing of naive Bayes algo: {acc4}")
        cm = confusion_matrix(yTest, y_pred4)
        sns.heatmap(cm, annot=True, fmt="d")  # decimal
        plt.xlabel("Predicted")
        plt.ylabel("Truth")
        plt.show()
        return f"confusion matrix: {cm}"
        # 6.random forest


    forest = RandomForestClassifier(n_estimators=10, criterion="entropy")
    forest.fit(xTrain, yTrain)
    y_pred5 = forest.predict(xTest)
    acc5 = accuracy_score(yTest, y_pred5)


    def trainForest():
        print(f"Accuracy Training of random forest algo: {forest.score(xTrain, yTrain)}")  # score()
        return "\n"


    def testForest():
        print(f"Accuracy testing of random forest algo: {acc5}")
        cm = confusion_matrix(yTest, y_pred5)
        sns.heatmap(cm, annot=True, fmt="d")  # decimal
        plt.xlabel("Predicted")
        plt.ylabel("Truth")
        plt.show()
        return f"confusion matrix: {cm}"


    def main_predict():
        # ignore customerID, monthlycharges
        c_gender = e1.get()
        if c_gender == "Male":
            c_gender = 1
        elif c_gender == "Female":
            c_gender = 0
        Label(wind, text=c_gender)

        c_sniorcitizen = e2.get()
        Label(wind, text=c_sniorcitizen)

        c_partner = e3.get()
        if c_partner == "Yes" or c_partner == "yes":
            c_partner = 1
        elif c_partner == "No" or c_partner == "no":
            c_partner = 0
        Label(wind, text=c_partner)

        c_Dependent = e4.get()
        if c_Dependent == "Yes" or c_Dependent == "yes":
            c_Dependent = 1
        elif c_Dependent == "No" or c_Dependent == "no":
            c_Dependent = 0
        Label(wind, text=c_Dependent)

        c_tenure = e5.get()
        Label(wind, text=c_tenure)

        c_phone = e6.get()
        if c_phone == "Yes" or c_phone == "yes":
            c_phone = 1
        elif c_phone == "No" or c_phone == "no":
            c_phone = 0
        Label(wind, text=c_phone)

        c_multiple = e7.get()
        if c_multiple == "Yes" or c_multiple == "yes":
            c_multiple = 1
        elif c_multiple == "No" or c_multiple == "no":
            c_multiple = 0
        elif c_multiple == "No phone service".lower():
            c_multiple = 2
        Label(wind, text=c_multiple)

        c_internet = e8.get()
        if c_internet == "DSL".lower():
            c_internet = 1
        elif c_internet == "No" or c_internet == "no":
            c_internet = 0
        elif c_internet == "Fiber optic".lower():
            c_internet = 2
        Label(wind, text=c_internet)

        c_on_sec = e9.get()
        if c_on_sec == "Yes" or c_on_sec == "yes":
            c_on_sec = 1
        elif c_on_sec == "No" or c_on_sec == "no":
            c_on_sec = 0
        elif c_on_sec == "No internet service".lower():
            c_on_sec = 2
        Label(wind, text=c_on_sec)

        c_on_back = e10.get()
        if c_on_back == "Yes" or c_on_back == "yes":
            c_on_back = 1
        elif c_on_back == "No" or c_on_back == "no":
            c_on_back = 0
        elif c_on_back == "No internet service".lower():
            c_on_back = 2
        Label(wind, text=c_on_back)

        c_device_protect = e11.get()
        if c_device_protect == "Yes" or c_device_protect == "yes":
            c_device_protect = 1
        elif c_device_protect == "No" or c_device_protect == "no":
            c_device_protect = 0
        elif c_device_protect == "No internet service".lower():
            c_device_protect = 2
        Label(wind, text=c_device_protect)

        c_tech = e12.get()
        if c_tech == "Yes" or c_tech == "yes":
            c_tech = 1
        elif c_tech == "No" or c_tech == "no":
            c_tech = 0
        elif c_tech == "No internet service".lower():
            c_tech = 2
        Label(wind, text=c_tech)

        c_tv = e13.get()
        if c_tv == "Yes" or c_tv == "yes":
            c_tv = 1
        elif c_tv == "No" or c_tv == "no":
            c_tv = 0
        elif c_tv == "No internet service".lower():
            c_tv = 2
        Label(wind, text=c_tv)

        c_mov = e14.get()
        if c_mov == "Yes" or c_mov == "yes":
            c_mov = 1
        elif c_mov == "No" or c_mov == "no":
            c_mov = 0
        elif c_mov == "No internet service".lower():
            c_mov = 2
        Label(wind, text=c_mov)

        c_contra = e15.get()
        if c_contra == "Month-to-month" or c_contra == "month":
            c_contra = 0
        elif c_contra == "One year" or c_contra == "one":
            c_contra = 1
        elif c_contra == "Two year" or c_contra == "two":
            c_contra = 2
        Label(wind, text=c_contra)

        c_paper = e16.get()
        if c_paper == "No" or c_paper == "no":
            c_paper = 0
        elif c_paper == "Yes" or c_paper == "yes":
            c_paper = 1
        Label(wind, text=c_paper)

        c_pay = e17.get()
        if c_pay == "Electronic" or c_pay == "electronic":
            c_pay = 0
        elif c_pay == "Mailed" or c_pay == "mailed":
            c_pay = 1
        elif c_pay == "Bank" or c_pay == "bank":
            c_pay = 2
        elif c_pay == "Credit" or c_pay == "credit":
            c_pay = 3
        Label(wind, text=c_pay)

        c_monthly = e18.get()
        Label(wind, text=c_monthly)

        c_total = e19.get()
        Label(wind, text=c_total)

        cells = [[c_gender, c_sniorcitizen, c_partner, c_tenure, c_Dependent, c_phone, c_multiple, c_internet, c_on_sec,
                  c_on_back, c_device_protect, c_tech, c_tv, c_mov, c_contra, c_paper, c_pay, c_monthly, c_total]]

        # cells
        def pred_LogisticRegression_basedon_cells():
            print(f"the prediction of LogisticRegression algo is: ")
            y = logReg.predict(cells)
            print(y)
            if y == 0:
                print("Yes, the customer will terminate the service.")
            elif y == 1:  # حاجة حلوة انها ب 1
                print("'No, the customer is happy with our Services.'")
            return "\n"

        def pred_SVM_basedon_cells():
            print(f"the prediction of SVM algo is: ")
            y = svm.predict(cells)
            if y == 0:
                print("Yes, the customer will terminate the service.")
            elif y == 1:
                print("No, the customer is happy with our Services.")
            return "\n"

        def pred_DecisionTree_basedon_cells():
            print(f"the prediction of DecisionTree algo is: ")
            y = decTree.predict(cells)
            if y == 0:
                print("Yes, the customer will terminate the service.")
            elif y == 1:
                print("No, the customer is happy with our Services.")
            return "\n"

        def pred_KNN_basedon_cells():
            print(f"the prediction of KNN algo is: ")
            y = knn.predict(cells)
            if y == 0:
                print("Yes, the customer will terminate the service.")
            elif y == 1:
                print("No, the customer is happy with our Services.")
            return "\n"

        def pred_naiveBayes_basedon_cells():
            print(f"the prediction of naive Bayes algo is: ")
            y = bayes.predict(cells)
            if y == 0:
                print("Yes, the customer will terminate the service.")
            elif y == 1:
                print("No, the customer is happy with our Services.")
            return "\n"

        def pred_randomForest_basedon_cells():
            print(f"the prediction of randomForest algo is: ")
            y = forest.predict(cells)
            if y == 0:
                print("Yes, the customer will terminate the service.")
            elif y == 1:
                print("No, the customer is happy with our Services.")
            return "\n"

        if (
                var0.get() == 1 and var1.get() == 1 and var2.get() == 1 and var3.get() == 1 and var4.get() == 1 and var5.get() == 1):
            print(pred_LogisticRegression_basedon_cells())
            print(pred_SVM_basedon_cells())
            print(pred_DecisionTree_basedon_cells())
            print(pred_randomForest_basedon_cells())
            print(pred_KNN_basedon_cells())
            print(pred_naiveBayes_basedon_cells())
            print("------------------------------------------------------")

        # .......etc


    def b_train():
        # if i select all checkboxes
        # all
        li = [var0, var1, var2, var3, var4, var5]
        for y in li:
            if y.get() == 1:
                if y == var0:
                    print(trainLogisticRegression())
                if y == var1:
                    print(trainSVM())
                if y == var2:
                    print(trainTree())
                if y == var3:
                    print(trainKnn())
                if y == var4:
                    print(trainBayes())
                if y == var5:
                    print(trainForest())


    def b_test():
        # if i select all checkboxes
        # all
        li = [var0, var1, var2, var3, var4, var5]
        for y in li:
            if y.get() == 1:
                if y == var0:
                    print(testLogisticRegression())
                if y == var1:
                    print(testSVM())
                if y == var2:
                    print(testTree())
                if y == var3:
                    print(testKnn())
                if y == var4:
                    print(testBayes())
                if y == var5:
                    print(testForest())

        # 1


    def first():
        if var6.get() == 0:
            b_train()


    def second():
        if var6.get() == 0:
            b_test()


    wind = Tk()  # obj from gui lib (main app window)    root=master
    wind.title("Service cancellation predictor (AI_project)")
    wind.geometry("1120x550")
    wind.iconbitmap(r"W:\AinShams uni\Project\resourses\w.ico")

    wind.minsize(880, 540)
    txt = Label(wind, text="Methodology:", font=("Arial", 10)).grid()

    var0 = IntVar()  # Create variable on window
    Checkbutton(wind, text="LogisticRegression", variable=var0).grid(row=1, column=1, sticky=W)
    var1 = IntVar()
    Checkbutton(wind, text="SVM", variable=var1).grid(row=1, column=2, sticky=W)
    var2 = IntVar()
    Checkbutton(wind, text="ID3", variable=var2).grid(row=1, column=3, sticky=W)
    var3 = IntVar()
    Checkbutton(wind, text="KNN", variable=var3).grid(row=1, column=4, sticky=W)
    var4 = IntVar()
    Checkbutton(wind, text="NaiveBayes", variable=var4).grid(row=1, column=5, sticky=W)
    var5 = IntVar()
    Checkbutton(wind, text="RandomForest", variable=var5).grid(row=1, column=6, sticky=W)

    # train-test radio button
    var6 = IntVar()
    button1 = Button(wind, text="Train", width=12, command=first).grid(row=4, column=2)
    button2 = Button(wind, text="Test", width=12, command=second).grid(row=4, column=4)
    button3 = Button(wind, text="Predict", width=14, command=main_predict).grid(row=20, column=5)

    Label(wind, text="customerID").grid(row=5, column=1)  # start
    Label(wind, text="gender").grid(row=5, column=3)
    Label(wind, text="SeniorCitizen").grid(row=5, column=5)

    Label(wind, text="Partner").grid(row=6, column=1)  # start
    Label(wind, text="Dependent").grid(row=6, column=3)
    Label(wind, text="Tenure").grid(row=6, column=5)

    Label(wind, text="Phone Service").grid(row=7, column=1)  # start
    Label(wind, text="Multiple Lines").grid(row=7, column=3)
    Label(wind, text="Internet Service").grid(row=7, column=5)

    Label(wind, text="Online Security").grid(row=8, column=1)  # start
    Label(wind, text="Online Backup").grid(row=8, column=3)
    Label(wind, text="Device Protection").grid(row=8, column=5)

    Label(wind, text="Tech Support").grid(row=9, column=1)  # start
    Label(wind, text="Streaming TV").grid(row=9, column=3)
    Label(wind, text="StreamingMovies").grid(row=9, column=5)

    Label(wind, text="Contract").grid(row=10, column=1)  # start
    Label(wind, text="PaperlessBilling").grid(row=10, column=3)
    Label(wind, text="PaymentMethod").grid(row=10, column=5)

    Label(wind, text="Monthly Charges").grid(row=11, column=1)  # start
    Label(wind, text="Total Charges").grid(row=11, column=3)

    # entries to accept inputs

    e0 = Entry(wind)
    e1 = Entry(wind)
    e2 = Entry(wind)
    # grid (figure)
    e0.grid(row=5, column=2)
    e1.grid(row=5, column=4)
    e2.grid(row=5, column=6)

    e3 = Entry(wind)
    e4 = Entry(wind)
    e5 = Entry(wind)
    e3.grid(row=6, column=2)
    e4.grid(row=6, column=4)
    e5.grid(row=6, column=6)

    e6 = Entry(wind)
    e7 = Entry(wind)
    e8 = Entry(wind)
    e6.grid(row=7, column=2)
    e7.grid(row=7, column=4)
    e8.grid(row=7, column=6)

    e9 = Entry(wind)
    e10 = Entry(wind)
    e11 = Entry(wind)
    e9.grid(row=8, column=2)
    e10.grid(row=8, column=4)
    e11.grid(row=8, column=6)

    e12 = Entry(wind)
    e13 = Entry(wind)
    e14 = Entry(wind)
    e12.grid(row=9, column=2)
    e13.grid(row=9, column=4)
    e14.grid(row=9, column=6)

    e15 = Entry(wind)
    e16 = Entry(wind)
    e17 = Entry(wind)
    e15.grid(row=10, column=2)
    e16.grid(row=10, column=4)
    e17.grid(row=10, column=6)

    e18 = Entry(wind)
    e19 = Entry(wind)
    e18.grid(row=11, column=2)
    e19.grid(row=11, column=4)

    wind.mainloop()

except:
    print("Ai Project 2022")

finally:
    # comparison
    ls = [logReg.score(xTrain, yTrain), svm.score(xTrain, yTrain), decTree.score(xTrain, yTrain),
          knn.score(xTrain, yTrain), bayes.score(xTrain, yTrain), forest.score(xTrain, yTrain)]
    ls2 = [acc0, acc1, acc2, acc3, acc4, acc5]
    num0 = None
    num1 = None
    for j, jj in enumerate(ls):
        if (num0 is None or jj > num0):
            num0 = jj
            index0 = j

    for jjj, jjjj in enumerate(ls2):
        if (num1 is None or jjjj > num1):
            num1 = jjjj
            index1 = jjj
    print(f"Best Accuracy Training {num0}, and its index(refer to algorithm respectively) {index0}")
    print(f"Best Accuracy testing {num1}, and its index(refer to algorithm respectively) {index1}")

    im = Image.open(r"W:\AinShams uni\Project\resourses\1_MyKDLRda6yHGR_8kgVvckg.png")
    im.show()
    '''
    #Saving dataframe with optimal features
    print(rfecv.support_)
    x_new = dataset.drop(
        ["gender", "SeniorCitizen", "Partner", "PhoneService", "OnlineBackup", "DeviceProtection", "MonthlyCharges"],
        axis=1)

    # Overview of the optimal features in comparison with the intial dataframe
    print("\"x\" dimension: {}".format(x_old.shape))
    print("\"x\" column list:", x_old.columns.tolist())
    print("\"x_new\" dimension: {}".format(x_new.shape))
    print("\"x_new\" column list:", x_new.columns.tolist())
    print(x_new)
    scaler2 = MinMaxScaler(copy=True, feature_range=(0, 1),
                          clip=False)  # create an obj(scaler) to fitting(compute min,max) then
    x_new = scaler2.fit_transform(x_new)
    xTrain2, xTest2, yTrain2, yTest2 = train_test_split(x_new, y, random_state=50, test_size=0.3)

    logReg2 = LogisticRegression(solver="liblinear", penalty="l1", tol=0.002, C=100)  # saga for big tables
    logReg2.fit(xTrain2, yTrain2)  # x,y train
    y_pred0_2 = logReg2.predict(xTest2)  # predict based on xtest البداية يعم
    acc0_2 = accuracy_score(yTest2, y_pred0_2)
    def trainLogisticRegression2():
        print(f"Accuracy Training of LogisticRegression algo: {logReg2.score(xTrain2,yTrain2)}")
        return "\n"

    def testLogisticRegression2():
        #confusion mat
        cm=confusion_matrix(yTest2,y_pred0_2)
        print(f"Accuracy testing of LogisticRegression algo: {acc0_2}")
        sns.heatmap(cm, annot=True, fmt="d")  # decimal
        plt.xlabel("Predicted")
        plt.ylabel("Truth")
        plt.show()
        return f"confusion matrix: {cm}"


    svm2 = SVC(kernel="rbf", max_iter=-1)  # create obj
    svm2.fit(xTrain2, yTrain2)
    y_pred1_2 = svm2.predict(xTest2)
    acc1_2 = accuracy_score(yTest2, y_pred1_2)


    def trainSVM2():
        print(f"Accuracy Training of SVM algo: {svm2.score(xTrain2, yTrain2)}")  # score()
        return "\n"


    def testSVM2():
        print(f"Accuracy testing of SVM algo: {acc1_2}")
        cm = confusion_matrix(yTest2, y_pred1_2)
        sns.displot(x=dataset['Churn'], bins=2)
        plt.show()
        return f"confusion matrix: {cm}"
    print(testLogisticRegression2())
    print(trainLogisticRegression2())
    print(testSVM2())
    print(trainSVM2())
    '''
    print("-----------------------------------------------------------------------------------")

'''

            #other wayes
    #dataset["TotalCharges"]=pd.to_numeric(dataset["TotalCharges"])
    #dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')#hen invalid parsing will be set as NaN
    # Fill the missing values with with the median value
    #dataset['TotalCharges'] = dataset['TotalCharges'].fillna(dataset['TotalCharges'].median())
    # for column in dataset.columns:
    #     if dataset[column].dtype == np.number:
    #         continue
    #     dataset[column] = LabelEncoder().fit_transform(dataset[column])

    #dataset['TotalCharges'] = dataset['TotalCharges'].astype(float)
    #
    # The customerID column isnt useful as the feature is used for identification of customers.

    # Encode categorical features

    # Defining the map function
    # def binary_map(feature):
    #     return feature.map({'Yes': 1, 'No': 0})
    #
    # def ecode(fea):
    #     return  fea.map({{'Yes': 1, 'No': 0,"No internet service": 2}})

    ## Encoding target feature
    # dataset['Churn'] = dataset[['Churn']].apply(binary_map)
    #
    # # Encoding gender category
    # dataset['gender'] = dataset['gender'].map({'Male': 1, 'Female': 0})
    #
    # # Encoding other binary category
    # binary_list = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    # dataset[binary_list] = dataset[binary_list].apply(binary_map)
    #
    # # Encoding the other categoric features with more than two categories
    # dataset = pd.get_dummies(dataset, drop_first=True)
    #

    # x=dataset.iloc[:,0:19].values
    # y=dataset.iloc[:,19].values #churn

    #dataset["TotalCharges"] = le.fit_transform(dataset["TotalCharges"])
    #dataset["TotalCharges"] = dataset["TotalCharges"].convert_dtypes()

    # #dataset["TotalCharges"].astype(np.int64)
    # for ip,ip6 in enumerate(dataset["TotalCharges"]):
    #     if ip6==" ":
    #         dataset.at[ip,"TotalCharges"]=0

    # print(f"now we check if null values are existing or not \n"
    #       f"{dataset.isnull().sum()}")
    # fig = px.imshow(corr, width=1000, height=1000)
    # fig.show()

'''