# Import
require(readr)
require(ggplot2)
require(dplyr)
require(forecast)

# Read from CSV
df <- read.csv(file="count.csv", colClasses=c('Date','integer'), header=FALSE, sep=",", stringsAsFactors=FALSE)
print(names(df))
print(sapply(df, class))
print(df$V1)

# Take subset of data
df <- subset(df, V1 <= as.Date('2014-12-01') )
h_ts <- ts(df$V2, frequency=12)

# Model
model <- auto.arima(h_ts, include.drift = FALSE)
tsdiag(model)
summary(model)
plot(forecast(model, h=12))
