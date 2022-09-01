library(academictwitteR)
library(tidyverse)

# bind data
train <- read_csv("train.csv", col_types = "c", col_select = id)
test <- read_csv("test.csv", col_types = "c")
users <- train |> bind_rows(test)

# run the loop
for (i in 1:nrow(users)) {

academictwitteR::get_all_tweets(users = users$id[i],
                                start_tweets = "2018-05-01T00:00:00Z",
                                end_tweets = "2022-01-01T00:00:00Z",
                                data_path = "./tweets",
                                n = 300,
                                page_n = 300,
                                bind_tweets = FALSE)

print(paste("progress:", i, "/", nrow(users)))


}



# bind tweets
tweets <- bind_tweets(data_path = "./tweets")
tweets_flat <- jsonlite::flatten(tweets)
tweets_flat <- tweets_flat |> distinct(id, .keep_all = TRUE)

# save rds
write_rds(x = tweets_flat, file = "./data/tweets_raw_full.rds")

