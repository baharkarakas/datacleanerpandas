import pandas as pd
from dataPreprocessing import (CategoricalEncoder, DataTypeConverter, DateTimeHandler,
                               FeatureEngineer, MissingValueHandler, OutlierHandler, Scaler, TextCleaner)

# Example usage
file_path = r'C:\Users\PC\Downloads\synthetic_sample_data.csv'
data = pd.read_csv(file_path)

# Handle missing values
data = MissingValueHandler.fill_with_mean(data, 'Rating')

# Handle outliers
data = OutlierHandler(method="iqr", threshold=1.5).fit_transform(data)

# Scale data
data = Scaler(method="standard").fit_transform(data)

# Clean text data
text_cleaner = TextCleaner(remove_stopwords=True, lemmatize=True)
data['Summary'] = data['Summary'].apply(text_cleaner.clean)

# Encode categorical data
data, _ = CategoricalEncoder.label_encode(data, 'Genre')
data, _ = CategoricalEncoder.one_hot_encode(data, 'Shooting Location')

# Date transformations
data = DateTimeHandler.convert_to_datetime(data, 'Release Date', format='%d/%m/%Y')
data = DateTimeHandler.extract_date_component(data, 'Release Date', 'year')

# Feature engineering
data = FeatureEngineer.add_difference(data, 'Budget in USD', 'Awards', 'Budget_Awards_Diff')
data = FeatureEngineer.add_product(data, 'Rating', 'Popular', 'Rating_Times_Popular')

# Convert data types
data = DataTypeConverter.to_numeric(data, 'Rating')
data = DataTypeConverter.to_categorical(data, 'Genre')

print(data.head())
