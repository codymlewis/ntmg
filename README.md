# NTMG

A fast and simple data management library for machine learning.


## Installation

Simply run `pip install git+https://github.com/codymlewis/ntmg.git`

## Usage

```python
import ntmg

X_train, Y_train, X_test, Y_test = load_data() # Loaded into numpy arrays
data = {
    "train": {"X": X_train, "Y": Y_train},
    "test": {"X": X_test, "Y": Y_test}
}
dataset = ntmg.Dataset(data)

print("Default data")
print(data)

print("Subset of the data")
print(data.select({"train": [1, 2, 3], "test": [4, 5, 6]}))

print("Normalised data")
print(data.normalise())
```