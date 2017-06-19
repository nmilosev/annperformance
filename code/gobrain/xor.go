package main

import (
    "github.com/goml/gobrain"
    "math/rand"
    "bufio"
    "encoding/csv"
    "os"
    "io"
    "strconv"
    "time"
    "fmt"
)

func main() {

    start := time.Now()
    
    rand.Seed(0)

    f, _ := os.Open("../../data/xor.csv")

    num_inputs  := 2
    num_hidden  := 2
    num_outputs := 1
    num_epochs  := 10000

    r := csv.NewReader(bufio.NewReader(f))

    patterns := [][][]float64{}

    for {
        record, err := r.Read()

        if err == io.EOF {
            break
        }

	pattern := [][] float64{{}, {}}

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

    fmt.Println("Loaded data in: ", time.Since(start))
    start = time.Now()

    ff := &gobrain.FeedForward{}

    ff.Init(num_inputs, num_hidden, num_outputs)

    fmt.Println("Constructed network in: ", time.Since(start))
    start = time.Now()
    
    // the learning rate is set to 0.1 and the momentum factor to 1
    ff.Train(patterns, num_epochs, 0.1, 1, false)

    fmt.Println("Trained network in: ", time.Since(start))
    start = time.Now()
    
    // testing the network
    ff.Test(patterns)
    
    fmt.Println("Tested network in: ", time.Since(start))
    
    bufio.NewReader(os.Stdin).ReadBytes('\n')
    
}
