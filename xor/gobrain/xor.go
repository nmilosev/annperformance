package main

import (
    "github.com/goml/gobrain"
    "math/rand"
)

func main() {
    rand.Seed(0)

    patterns := [][][]float64{
      {{0, 0}, {0}},
      {{0, 1}, {1}},
      {{1, 0}, {1}},
      {{1, 1}, {0}},
    }

    ff := &gobrain.FeedForward{}

    // initialize the Neural Network;
    // the networks structure will contain:
    // 2 inputs, 2 hidden nodes and 1 output.
    ff.Init(2, 2, 1)

    // the training will run for 1000 epochs
    // the learning rate is set to 0.1 and the momentum factor to 1
    ff.Train(patterns, 1000, 0.1, 1, false)

    // testing the network
    ff.Test(patterns)

    // predicting a value
    inputs := []float64{1, 1}
    ff.Update(inputs)
}
