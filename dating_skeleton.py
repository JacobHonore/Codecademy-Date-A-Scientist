import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing, linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import time

# Is there a connection between religion and if you smoke?
# Is there a connection between education level and number of pets?
# Is there a connection between body_type and diet?
# Is there a connection between religion and income?


class Dating_skeleton():
    def __init__(self, csvfilename):
        self.df = pd.read_csv(csvfilename)

    def explore_dataset(self, df):
        mean_incomes = df[(df.income > 0)]  # Exclude non-answers
        mean_incomes = mean_incomes[['diet', 'income']].groupby(['diet']).mean().sort_values(['income'])
        print(mean_incomes)
        x = mean_incomes.index
        y = mean_incomes['income']
        fig, ax = plt.subplots()
        ind = np.arange(len(y))  # the x locations for the groups
        ax.barh(ind, y, color="blue")
        ax.set_yticks(ind)
        ax.set_yticklabels(x, minor=False)
        plt.title('Comparing diet with mean income')
        plt.xlabel('Mean income')
        plt.ylabel('Diet')
        plt.tight_layout()
        plt.show()

        body_types = df['body_type'].replace(np.nan, 'no answer', regex=True).value_counts().sort_values()
        print(body_types)
        x = body_types.index
        y = body_types
        fig, ax = plt.subplots()
        ind = np.arange(len(y))  # the x locations for the groups
        ax.barh(ind, y, color="red")
        ax.set_yticks(ind)
        ax.set_yticklabels(x, minor=False)
        plt.title('Value counts of body type')
        plt.xlabel('Count')
        plt.ylabel('Body type')
        plt.tight_layout()
        plt.show()

    def augment_data(self, df):
        def map_religion(x):
            religion_mapping = {
                "laughing about it": 0,
                "not too serious about it": 1,
                "somewhat serious about it": 3,
                "very serious about it": 4
            }
            group = np.nan
            if (pd.isna(x)):
                return group
            for key in religion_mapping:
                if x and key in x:
                    group = religion_mapping[key]
                    break
                elif x:
                    group = 2
            return group
        body_type_mapping = {
            "average": 1,
            "fit": 2,
            "athletic": 2,
            "thin": 2,
            "curvy": 0,
            "a little extra": 0,
            "skinny": 1,
            "full figured": 0,
            "overweight": 0,
            "jacked": 2,
            "used up": 0,
            "rather not say": np.nan
        }
        df["body_form"] = df.body_type.map(body_type_mapping)

        df["religion_seriousness"] = df.religion.map(map_religion)
        return df

    def normalize_data(self, df):
        feature_data = df[['body_form', 'religion_seriousness']]
        x = feature_data.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
        return feature_data

    def run_naive_bayes(self, df):
        print("Starting Naive bayes")
        df = df.dropna(subset=['body_form', 'religion_seriousness', 'diet'])
        classifier = MultinomialNB()
        X_train, X_test, y_train, y_test = train_test_split(df[['body_form', 'religion_seriousness']], df['diet'], test_size=0.33, random_state=949)
        t0 = time.time()
        classifier.fit(X_train, y_train)
        t1 = time.time()
        predictions = classifier.predict(X_test)
        print(f"Took {t1-t0}s to train Naive bayes")
        print(f"Training set: {len(X_train)} Test set: {len(X_test)}")
        print("Accuracy: ", accuracy_score(y_test, predictions))

    def run_k_neighbors(self, df):
        print("Starting KNeighborsClassifier")
        df = df.dropna(subset=['body_form', 'religion_seriousness', 'diet'])
        classifier = KNeighborsClassifier(n_neighbors=5)
        X_train, X_test, y_train, y_test = train_test_split(df[['body_form', 'religion_seriousness']], df['diet'], test_size=0.33, random_state=33)
        t0 = time.time()
        classifier.fit(X_train, y_train)
        t1 = time.time()
        predictions = classifier.predict(X_test)
        print(f"Took {t1-t0}s to KNeighborsClassifier")
        print(f"Training set: {len(X_train)} Test set: {len(X_test)}")
        print("Accuracy: ", accuracy_score(y_test, predictions))

    def run_linear_regression(self, df):
        print("Starting LinearRegression")
        df = df.dropna(subset=['body_form', 'religion_seriousness', 'age'])
        classifier = linear_model.LinearRegression()
        X_train, X_test, y_train, y_test = train_test_split(df[['body_form', 'religion_seriousness']], df['age'], test_size=0.33, random_state=13)
        t0 = time.time()
        classifier = classifier.fit(X_train, y_train)
        t1 = time.time()
        predictions = classifier.predict(X_test)
        print(f"Took {t1-t0}s to LinearRegression")
        print(f"Training set: {len(X_train)} Test set: {len(X_test)}")
        print("Train score:")
        print(classifier.score(X_train, y_train))
        print("Test score:")
        print(classifier.score(X_test, y_test))
        print("Coefs:")
        print(classifier.coef_)


def main():
    dataprocessing = Dating_skeleton("profiles.csv")
    #dataprocessing.explore_dataset(dataprocessing.df)
    dataprocessing.df = dataprocessing.augment_data(dataprocessing.df)
    normalized_data = dataprocessing.normalize_data(dataprocessing.df)
    dataprocessing.run_naive_bayes(dataprocessing.df)
    dataprocessing.run_k_neighbors(dataprocessing.df)
    dataprocessing.run_linear_regression(dataprocessing.df)


if __name__ == '__main__':
    main()
