import pandas as pd

# Load the CSV file
data = pd.read_csv("output.csv")

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Calculate sizes of each split
total_size = len(data)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)

# Split the data
train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# Save the splits to new CSV files
train_data.to_csv("train.csv", index=False)
val_data.to_csv("val.csv", index=False)
test_data.to_csv("test.csv", index=False)
