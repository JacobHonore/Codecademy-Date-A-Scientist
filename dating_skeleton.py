import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Is there a connection between religion and if you smoke?
# Is there a connection between education level and number of pets?
# Is there a connection between body_type and diet?
# Is there a connection between religion and income?


class Dating_skeleton():
    def __init__(self, csvfilename):
        self.df = pd.read_csv(csvfilename)

    def explore_dataset(self):
        mean_incomes = self.df[(self.df.income > 0)]  # Exclude non-answers
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

        body_types = self.df['body_type'].replace(np.nan, 'no answer', regex=True).value_counts().sort_values()
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


def main():
    dataprocessing = Dating_skeleton("profiles.csv")
    dataprocessing.explore_dataset()


if __name__ == '__main__':
    main()
