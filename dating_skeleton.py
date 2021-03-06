import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing, linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
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
        diet_mapping = {
            "strictly anything": 0,
            "mostly anything": 1,
            "anything": 2,
            "halal": 3,
            "other": 4,
            "kosher": 5,
            "vegetarian": 6,
            "vegan": 7,
            "mostly halal": 8,
            "mostly other": 9,
            "mostly kosher": 10,
            "mostly vegetarian": 11,
            "mostly vegan": 12,
            "strictly halal": 13,
            "strictly kosher": 14,
            "strictly other": 15,
            "strictly vegetarian": 16,
            "strictly vegan": 17,
        }
        df["diet_code"] = df.diet.map(diet_mapping)
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
        X_train, X_test, y_train, y_test = train_test_split(df[['body_form', 'religion_seriousness']], df['diet'], test_size=0.33, random_state=33)
        t0 = time.time()
        classifier.fit(X_train, y_train)
        t1 = time.time()
        predictions = classifier.predict(X_test)
        t2 = time.time()
        print(f"Took {t1-t0}s to train Naive bayes")
        print(f"Took {t2-t1}s to predict on Naive bayes")
        print(f"Training set: {len(X_train)} Test set: {len(X_test)}")
        print("Accuracy: ", accuracy_score(y_test, predictions))

    def run_k_neighbors(self, df, find_best_k=False):
        print("Starting KNeighborsClassifier")
        df = df.dropna(subset=['body_form', 'religion_seriousness', 'diet'])
        X_train, X_test, y_train, y_test = train_test_split(df[['body_form', 'religion_seriousness']], df['diet'], test_size=0.33, random_state=33)
        if find_best_k:
            k_range = range(1, 101)
            best_k = 1
            best_accuracy = 0.0
            accuracies = []
            for k in k_range:
                classifier = KNeighborsClassifier(n_neighbors=k)
                classifier.fit(X_train, y_train)
                accuracy = classifier.score(X_test, y_test)
                accuracies.append(accuracy)
                if (accuracy > best_accuracy):
                    print("Found new best accuracy {0} with k={1}".format(accuracy, k))
                    best_accuracy = accuracy
                    best_k = k
            self.plot_k_accuracy(k_range, accuracies, "KNeighborsClassifier")
            classifier = KNeighborsClassifier(n_neighbors=best_k)
        else:
            classifier = KNeighborsClassifier(n_neighbors=5)
        t0 = time.time()
        classifier.fit(X_train, y_train)
        t1 = time.time()
        predictions = classifier.predict(X_test)
        t2 = time.time()
        print(f"Took {t1-t0}s to train KNeighborsClassifier")
        print(f"Took {t2-t1}s to predict on KNeighborsClassifier")
        print(f"Training set: {len(X_train)} Test set: {len(X_test)}")
        print("Accuracy: ", accuracy_score(y_test, predictions))

    def run_linear_regression(self, df):
        print("Starting LinearRegression")
        df = df.dropna(subset=['body_form', 'religion_seriousness', 'diet_code'])
        classifier = linear_model.LinearRegression()
        X_train, X_test, y_train, y_test = train_test_split(df[['body_form', 'religion_seriousness']], df['diet_code'], test_size=0.33, random_state=33)
        t0 = time.time()
        classifier = classifier.fit(X_train, y_train)
        t1 = time.time()
        predictions = classifier.predict(X_test)
        t2 = time.time()
        print(f"Took {t1-t0}s to train LinearRegression")
        print(f"Took {t2-t1}s to predict on LinearRegression")
        print(f"Training set: {len(X_train)} Test set: {len(X_test)}")
        print("Train score:")
        print(classifier.score(X_train, y_train))
        print("Test score:")
        print(classifier.score(X_test, y_test))
        print("Coefs:")
        print(classifier.coef_)

    def run_k_nearest_regression(self, df, find_best_k=False):
        print("Starting KNeighborsRegressor")
        df = df.dropna(subset=['body_form', 'religion_seriousness', 'diet_code'])
        X_train, X_test, y_train, y_test = train_test_split(df[['body_form', 'religion_seriousness']], df['diet_code'], test_size=0.33, random_state=33)
        if find_best_k:
            k_range = range(1, 101)
            best_k = 1
            best_accuracy = 0.0
            accuracies = []
            for k in k_range:
                classifier = KNeighborsRegressor(n_neighbors=k)
                classifier.fit(X_train, y_train)
                accuracy = classifier.score(X_test, y_test)
                accuracies.append(accuracy)
                if (accuracy > best_accuracy):
                    print("Found new best accuracy {0} with k={1}".format(accuracy, k))
                    best_accuracy = accuracy
                    best_k = k
            self.plot_k_accuracy(k_range, accuracies, "KNeighborsRegressor")
            classifier = KNeighborsRegressor(n_neighbors=best_k)
        else:
            classifier = KNeighborsRegressor(n_neighbors=5)
        t0 = time.time()
        classifier = classifier.fit(X_train, y_train)
        t1 = time.time()
        predictions = classifier.predict(X_test)
        t2 = time.time()
        print(f"Took {t1-t0}s to train KNeighborsRegressor")
        print(f"Took {t2-t1}s to predict on KNeighborsRegressor")
        print(f"Training set: {len(X_train)} Test set: {len(X_test)}")
        print("Train score:")
        print(classifier.score(X_train, y_train))
        print("Test score:")
        print(classifier.score(X_test, y_test))

    def plot_k_accuracy(self, k_range, accuracies, title):
        plt.plot(k_range, accuracies, '.r-')
        plt.title(title + ' Classifier Accuracy')
        plt.xlabel('k')
        plt.ylabel('Validation accuracy')
        plt.show()

def main():
    dataprocessing = Dating_skeleton("profiles.csv")
    #dataprocessing.explore_dataset(dataprocessing.df)
    dataprocessing.df = dataprocessing.augment_data(dataprocessing.df)
    normalized_data = dataprocessing.normalize_data(dataprocessing.df)
    dataprocessing.run_naive_bayes(dataprocessing.df)
    dataprocessing.run_k_neighbors(dataprocessing.df, True)
    dataprocessing.run_linear_regression(dataprocessing.df)
    dataprocessing.run_k_nearest_regression(dataprocessing.df, True)


if __name__ == '__main__':
    main()
