# Pandas Basics

Pandas is a data manipulation library for Python. It provides two main
data structures: Series (1D) and DataFrame (2D).

## Loading data

Use `pd.read_csv()` to load a CSV file. Use `pd.read_excel()` for Excel.
Both return a DataFrame.

## Selecting rows

Use `.loc` for label-based selection and `.iloc` for integer-position
selection. Note that .loc slicing is inclusive on both ends, while
.iloc follows standard Python slicing (exclusive on the end).

## Filtering

Boolean masks work the same as in NumPy. Use & for AND, | for OR,
and always wrap each condition in parentheses.