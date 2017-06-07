package main

import (
    "github.com/goml/gobrain"
    "math/rand"
    "bufio"
    "encoding/csv"
    "os"
    //"fmt"
    "io"
    "strconv"
)

func main() {
    rand.Seed(0)

    f, _ := os.Open("../../data/xor.csv")

    num_inputs  := 2
    num_outputs := 1

    r := csv.NewReader(bufio.NewReader(f))

    patterns := [][][]float64{}

    for {
        record, err := r.Read()

        if err == io.EOF {
            break
        }

	pattern := [][] float64{{}, {}}

	// for XOR output is first
	inputs := record[1:]
	outputs := record[:1]

	input_float := make([]float64, num_inputs)
	output_float := make([]float64, num_outputs)

	for i := range inputs {
	    f, _ := strconv.ParseFloat(inputs[i], 64)
	    input_float[i] = f
	}

        for i := range outputs {
            f, _ := strconv.ParseFloat(outputs[i], 64)
            output_float[i] = f 
        }

	pattern[0] = input_float;
	pattern[1] = output_float;

        patterns = append(patterns, pattern)
    }


    ff := &gobrain.FeedForward{}

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
