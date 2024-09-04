# %%
import pandas as pd
import numbers as np


# %%
df = pd.read_csv("/Users/simonlee/Downloads/deltasUntitled spreadsheet - Sheet1 (1).csv")
# Make the first row the header
df.columns = df.iloc[0]  # Set the first row as column names
df = df.drop(0).reset_index(drop=True)  # Drop the first row and reset index
df = df.drop("Task",axis=1)
df = df.dropna()
# Function to keep the first five digits of every cell
def first_five_digits(val):
    return str(val)[:5]  # Convert to string and take first 5 characters

# Apply the function to each cell in the DataFrame
df = df.applymap(first_five_digits)
df = df.drop("Models",axis=1)
df = df.astype(float)
df

# %%
# Subtract the 'Frozen' column from all other columns
df_cleaned = df.copy()
for col in df_cleaned.columns:
    if col != 'Frozen':  # Don't subtract from the Frozen column itself
        df_cleaned[col] = df_cleaned[col] - df_cleaned['Frozen']

df_cleaned 

# %%
latex_code = df_cleaned.to_latex(index=False)  # Set index=False if you don't want the index in the LaTeX output

# Print the LaTeX code
print(latex_code)

# %%
df = pd.read_csv("/Users/simonlee/Downloads/deltasUntitled spreadsheet - Sheet2.csv")
# Make the first row the header
df.columns = df.iloc[0]  # Set the first row as column names
df = df.drop(0).reset_index(drop=True)  # Drop the first row and reset index
df = df.drop("Task",axis=1)
df = df.dropna()
# Function to keep the first five digits of every cell
def first_five_digits(val):
    return str(val)[:5]  # Convert to string and take first 5 characters

# Apply the function to each cell in the DataFrame
df = df.applymap(first_five_digits)
df = df.drop("Models",axis=1)
df = df.astype(float)
# Subtract the 'Frozen' column from all other columns
df_cleaned = df.copy()
for col in df_cleaned.columns:
    if col != 'Frozen':  # Don't subtract from the Frozen column itself
        df_cleaned[col] = df_cleaned[col] - df_cleaned['Frozen']

df_cleaned 

# %%
latex_code = df_cleaned.to_latex(index=False)  # Set index=False if you don't want the index in the LaTeX output

# Print the LaTeX code
print(latex_code)

# %%



