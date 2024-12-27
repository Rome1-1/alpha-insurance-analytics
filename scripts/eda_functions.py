import seaborn as sns
import matplotlib.pyplot as plt

def plot_histogram(df, column):
    df[column].hist(bins=20)
    plt.title(f'Distribution of {column}')
    plt.show()

def plot_correlation(df):
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()