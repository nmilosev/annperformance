# Neural Net (nnet) with R
# load:
memory.limit(size=8192000)

library(tictoc)
tic()

rm(list=ls())

setwd("C:\\code\\annperformance-master\\code\\R")

library(nnet)

mnist_inputs <- read.csv("..\\..\\data\\mnist_train_inputs.csv", header=FALSE)
mnist_outputs <- read.csv("..\\..\\data\\mnist_train_outputs.csv", header=FALSE)

set.seed(0)

nnetout <- data.frame(mnist_outputs)
nnetin  <- data.frame(mnist_inputs)

nnetdata <- data.frame(nnetout, nnetin)

toc()
tic()

# construct:
hidden_layer_size <- 30 # 100 runs out of memory :(
max_iter <- 1

toc()
tic()

# train:
nnetstr <- nnet(nnetout, nnetin, data=nnetdata, size=hidden_layer_size, linout=T, maxit=max_iter, decay=1.0e-5, MaxNWts=100000)

toc()
tic()

# eval:
invisible(predict(nnetstr, nnetin))

toc()

dummy <- readLines("stdin", n=1)

