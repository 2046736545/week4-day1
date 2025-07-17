import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    def __init__(self, data):
        self.data = data

    def visualize(self):
        plt.figure(figsize=(12, 6))
        sns.histplot(self.data['Average Price'], bins=20, alpha=0.5, label='Average Price', kde=True)
        plt.title('Average Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Variety', y='Average Price', data=self.data)
        plt.title('Average Price by Variety')
        plt.xticks(rotation=45)
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.barplot(x='Variety', y='Average Price', data=self.data)
        plt.title('Average Price by Variety')
        plt.xticks(rotation=45)
        plt.show()