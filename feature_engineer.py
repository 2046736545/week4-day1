from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class FeatureEngineer:
    def __init__(self, data):
        self.data = data

    def engineer(self):
        numerical_features = ['Month', 'Year', 'Day', 'DayOfWeek']
        categorical_features = [
            'City Name', 'Type', 'Package', 'Variety',
            'Origin', 'Item Size', 'Color', 'Season'
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        return preprocessor, numerical_features, categorical_features