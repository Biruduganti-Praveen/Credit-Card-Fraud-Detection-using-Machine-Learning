import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import tree
import urllib.request
from PIL import Image

try:
    @st.cache(suppress_st_warning=True, show_spinner=False, allow_output_mutation=True)
    def long_running_function():
        import pandas as pd
        import numpy as np
        import seaborn as sns
        import warnings
        import joblib

        from sklearn.metrics import confusion_matrix, classification_report
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import precision_recall_fscore_support

        warnings.filterwarnings("ignore")
        sns.set()

        X_test = pd.read_csv("https://raw.githubusercontent.com/Biruduganti-Praveen/Credit-Card-Fraud-Detection-using-Machine-Learning/main/XTest.csv"
                    ,dtype=float)
        y_test = pd.read_csv("https://raw.githubusercontent.com/Biruduganti-Praveen/Credit-Card-Fraud-Detection-using-Machine-Learning/main/YTest.csv"
                ,dtype=float)
        X_test = X_test.drop("Unnamed: 0", axis=1)
        y_test = y_test.drop("Unnamed: 0", axis=1)

        return pd, np, sns, warnings, joblib, confusion_matrix, classification_report, precision_recall_curve, precision_recall_fscore_support, X_test, y_test

    pd, np, sns, warnings, joblib, confusion_matrix, classification_report, precision_recall_curve, precision_recall_fscore_support, X_test, y_test = long_running_function()

    col1, col2, col3 = st.beta_columns([0.5, 3, 0.5])
    col2.title("Credit Card Fraud Detection")
    st.sidebar.markdown('<p style=" color:Red;font-style:italic;font-weight: bold;font-size: 15px;">Made By: &nbsp &nbsp<link href="https://emoji-css.afeld.me/emoji.css" rel="stylesheet"><i class="em em-point_down" aria-role="presentation" aria-label="WHITE DOWN POINTING BACKHAND INDEX"></i></p>', unsafe_allow_html=True)
    st.sidebar.markdown('''<p style=" text-shadow: 0 0 2px #0802bf, 0 0 30px #ffde05;font-weight: bold;font-size: 30px;">Biruduganti Praveen</p>''', unsafe_allow_html=True)
    st.write(" ")
    st.sidebar.write('''LinkedIn Profile:
    https://bit.ly/3z5jvvU''')
    st.sidebar.write('''Code URL:
        https://bit.ly/3kKlfa6''')
    st.sidebar.write('''

        ''')
    st.sidebar.write('''

        ''')

    smodel = None
    MODEL = None
    choose = None


    choose = st.sidebar.selectbox(
        "Choose Your Action", ('Check Model Performance', 'View Dataset', 'Model Comparison'))

    if choose == "Check Model Performance":
        smodel = st.selectbox("Select Model", ('Logistic Regression', 'Naive Bayes',
                                               'SGDClassifier', 'Decision Tree', 'Random Forest'))
        _, _, col3 = st.beta_columns([3.5, 3.5, 3])
        col3.markdown(
            f'<p style="font-family:Times New Roman; font-style:italic;font-weight: bold; color:White; font-size: 15px;">Selected \"{smodel}\"</p>', unsafe_allow_html=True)
        st.write('''

        ''')

        @st.cache(suppress_st_warning=True, show_spinner=False, allow_output_mutation=True)
        def func(smodel, MODEL):

            if smodel == "Logistic Regression":
                MODEL = joblib.load('./ML_Credit_Card_Saved_Models/LogisticReg.sav')

            elif smodel == "Naive Bayes":
                MODEL = joblib.load('./ML_Credit_Card_Saved_Models/NaiveBayes.sav')

            elif smodel == "SGDClassifier":
                MODEL = joblib.load('./ML_Credit_Card_Saved_Models/SGD.sav')

            elif smodel == "Decision Tree":
                MODEL = joblib.load(
                    './ML_Credit_Card_Saved_Models/DecisionTree.sav')

            elif smodel == "Random Forest":
                MODEL = joblib.load(
                    './ML_Credit_Card_Saved_Models/RandomForest.sav')

            pred = MODEL.predict(X_test)
            accuracy = metrics.accuracy_score(pred, y_test[["fraudRisk"]])

            (TN, FP), (FN, TP) = confusion_matrix(y_test[["fraudRisk"]], pred)
            specificity = TN/(TN + FP)
            f1_score = metrics.f1_score(y_test[["fraudRisk"]], pred)
            recall = metrics.recall_score(y_test[["fraudRisk"]], pred)
            precision = metrics.precision_score(y_test[["fraudRisk"]], pred)

            return MODEL, pred, accuracy, specificity, f1_score, recall, precision

        MODEL, pred, accuracy, specificity, f1_score, recall, precision = func(
            smodel, MODEL)

        st.markdown('<p style="font-family:Times New Roman; color:Yellow; font-size: 30px;">Performance:</p>',
                    unsafe_allow_html=True)
        dataframe = pd.DataFrame(
            [[accuracy*100, specificity, f1_score, recall, precision]],
            columns=['Accuracy(%)', 'Specificity', 'F1 Score', 'Recall', 'Precision'], index = [smodel])
        st.table(dataframe)
        st.write('''

        ''')

        st.markdown('<p style="font-family:Times New Roman; color:Yellow; font-size: 30px;">Classification Report:</p>', unsafe_allow_html=True)

        def pandas_classification_report(y_true, y_pred):
            metrics_summary = precision_recall_fscore_support(
                y_true=y_true,
                y_pred=y_pred)

            avg = list(precision_recall_fscore_support(
                y_true=y_true,
                y_pred=y_pred,
                average='weighted'))

            metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
            class_report_df = pd.DataFrame(
                list(metrics_summary),
                index=metrics_sum_index)

            support = class_report_df.loc['support']
            total = support.sum()
            avg[-1] = total

            class_report_df['avg / total'] = avg

            return class_report_df.T

        st.table(pandas_classification_report(y_test[["fraudRisk"]], pred))
        st.text("")
        st.markdown('<p style="font-family:Times New Roman; color:Yellow; font-size: 30px;">Plotting:</p>',
                    unsafe_allow_html=True)

        col1, col2, col3, col4 = st.beta_columns([4, 3, 4, 3])
        do_plot = col1.button(("Precision Recall Curve"))
        if do_plot:
            fig = plt.figure(figsize=(15, 6))
            precision, recall, _ = precision_recall_curve(
                pred, np.array(y_test["fraudRisk"]))
            plt.plot(recall, precision, color='orange', label='Logistic')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Test Data Precision-Recall curve')
            plt.legend()
            st.pyplot(fig)

        do_plot = col2.button(("Fraud Count"))
        if do_plot:
            fig = plt.figure(figsize=(15, 6))
            sns.countplot(pred)
            plt.title('No. of Fraud and Non-Frauds')
            st.pyplot(fig)

        do_plot = col3.button(("Confusion Matrix"))
        if do_plot:
            fig = plt.figure()
            sns.heatmap(confusion_matrix(
                y_test[["fraudRisk"]], pred), annot=True, fmt=".1f")
            st.pyplot(fig)

        if smodel == "Decision Tree":
            do_plot = col4.button(("Decision Tree"))
            if do_plot:
                urllib.request.urlretrieve('https://raw.githubusercontent.com/Biruduganti-Praveen/Credit-Card-Fraud-Detection-using-Machine-Learning/main/Images/DecisionTreePlot.jpg',
                                    "DecisionTreePlot.jpg")
                img = Image.open('DecisionTreePlot.jpg')
                st.image(img, caption='Decision Tree Plot',use_column_width=True)

    elif choose == "View Dataset":
        st.write("Actual Dataset (First 20 Rows):               SHAPE: (10000000, 7)")
        st.table(pd.read_csv("ActualDataset.txt"))

        st.subheader("Showing Dataset After applying StandarScaler")
        st.write("Independent Dataset (First 15 Rows):")
        st.table(X_test.head(15))

        st.write("Dependent Feature (First 15 Rows):")
        st.table(y_test.head(15))



    else:
        st.markdown('<p style="font-family:Times New Roman; font-style:italic;color:Yellow; font-size: 25px;">Accuracy of All Model</p>', unsafe_allow_html=True)
        urllib.request.urlretrieve('https://raw.githubusercontent.com/Biruduganti-Praveen/Credit-Card-Fraud-Detection-using-Machine-Learning/main/Images/accuracy.jpg',
                                    "accuracy.jpg")
        img = Image.open('accuracy.jpg')
        st.image(img, caption='Accuracy of All Models',width=530)

        st.markdown('<p style="font-family:Times New Roman; font-style:italic;color:Yellow; font-size: 25px;">Recall of All Model</p>', unsafe_allow_html=True)
        urllib.request.urlretrieve("https://raw.githubusercontent.com/Biruduganti-Praveen/Credit-Card-Fraud-Detection-using-Machine-Learning/main/Images/recall.jpg"
                                   ,"recall.jpg")
        img = Image.open('recall.jpg')
        st.image(img, caption='Recall of All Models',width=530)

        st.markdown('<p style="font-family:Times New Roman; font-style:italic;color:Yellow; font-size: 25px;">Precision of All Model</p>', unsafe_allow_html=True)
        urllib.request.urlretrieve('https://raw.githubusercontent.com/Biruduganti-Praveen/Credit-Card-Fraud-Detection-using-Machine-Learning/main/Images/precision.jpg',
                                    "accuracy.jpg")
        img = Image.open('accuracy.jpg')
        st.image(img, caption='Precision of All Models',width=530)

        st.markdown('<p style="font-family:Times New Roman; font-style:italic;color:Yellow; font-size: 25px;">F1 Score of All Model</p>', unsafe_allow_html=True)
        urllib.request.urlretrieve('https://raw.githubusercontent.com/Biruduganti-Praveen/Credit-Card-Fraud-Detection-using-Machine-Learning/main/Images/f1score.jpg',
                                    "accuracy.jpg")
        img = Image.open('accuracy.jpg')
        st.image(img, caption='F1 Score of All Models',width=530)

        st.markdown('<p style="font-family:Times New Roman; font-style:italic;color:Yellow; font-size: 25px;">Precision-Recall Curve of All Model</p>', unsafe_allow_html=True)
        urllib.request.urlretrieve('https://raw.githubusercontent.com/Biruduganti-Praveen/Credit-Card-Fraud-Detection-using-Machine-Learning/main/Images/precision-recall.jpg',
                                    "accuracy.jpg")
        img = Image.open('accuracy.jpg')
        st.image(img, caption='Precision-Recall Curve of All Models',use_column_width=True)

        st.write(''' 
        ''')
        st.write('''

        ''')
        st.markdown('<p style=" color:orange;font-style:italic;;font-size: 15px;">**Sensitivity(Recall) is a measure of the proportion of actual positive cases that got predicted as positive (or true positive).</p>', unsafe_allow_html=True)
        st.markdown('<p style=" color:orange;font-style:italic;;font-size: 15px;">**Precision is the ratio of correctly predicted positive observations to the total predicted positive observations.</p>', unsafe_allow_html=True)
except:
    st.title("ERROR 404")
    st.markdown("Sorry for the inconvenience; there is some technical problem that will be fixed soon!!")

