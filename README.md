# Hierarchical Approximate Minimum Degree

A hierarchical approach to the Approximate Minimum Degree algorithm for fill-reducing matrix ordering

## Usage

```
# Standard ParAMD
./paramd --matrix matrix.mtx --algo paramd

# Hierarchical AMD
./paramd --matrix matrix.mtx --algo paramd --hierarchical

# With custom parameters
./paramd --matrix matrix.mtx --algo paramd --hierarchical \
 --partition-threshold 5000 --recursion-depth 5 --balance-factor 0.5
```
