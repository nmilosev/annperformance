# annperformance

(c) 2017 Nemanja Milosevic (nmilosev [at] dmi.rs)

This repository contains the performance comparison between different neural network training environments:

- keras (Python)
- gobrain (Go Language)
- MATLAB
- R with nnet package

Please see `paper` and `results` directories for more detail.

Directory structure is as follows:

```
root
├── code
│   ├── gobrain
│   ├── keras
│   ├── keras-optimized
│   ├── matlab
│   └── R
├── data
│   └── doc
├── LICENSE
├── normalizer.py
├── paper
└── results

```

More information about used datasets can be found in `data/doc`. 

R scripts are to be run with `RScript`, MATLAB examples have batch runners.

For running Go and Python examples, use `go run [name]` and `python3 [name]`.

# License

This work is licensed with the included license file (GPLv3). Please notify me if you use these results for your research.
