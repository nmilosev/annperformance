# Neural Net (nnet) with R
# load:
library(tictoc)
tic()

rm(list=ls())

setwd("C:\\code\\annperformance-master\\code\\R")

library(nnet)

nndata <- read.csv("xor.csv")

set.seed(0)

nnetout <- data.frame(nndata$output)
nnetin  <- data.frame(nndata$input1, nndata$input2)

nnetdata <- data.frame(nnetout, nnetin)

toc()
tic()

# construct:
hidden_layer_size <- 50
max_iter <- 50

toc()
tic()

# train:
nnetstr <- nnet(nnetout, nnetin, data=nnetdata, size=hidden_layer_size, linout=T, maxit=max_iter, decay=1.0e-5)

toc()
tic()

# eval:
invisible(predict(nnetstr, nnetin))

toc()

dummy <- readLines("stdin", n=1)

