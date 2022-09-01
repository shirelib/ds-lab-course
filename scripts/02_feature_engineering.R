library(tidyverse)
library(moments) # for kurtosis, skewness
library(sentimentr) # for sentiment analysis
library(quanteda.dictionaries) # for dictionary analysis
library(text) # for word embeddings
library(janitor) # for some column name cleaning
library(corrplot) # for the correlation matrix

# Pre Proccessing Downloaded tweets ----
## Read raw data
tweets_raw_full <- read_rds("./data/tweets_raw_full.rds")

## Removing retweets
tweets_no_RT <- tweets_raw_full |>
  filter(purrr::map(tweets_raw_full$referenced_tweets, 1) != 'retweeted') # or quoted/repiled to

## count number of tweets after removal, filter users who have sub-50 tweets left
tweet_no_rt_count <- tweets_no_RT |> group_by(author_id) |> tally() 
tweets_count <- tweets_raw_full |> 
  group_by(author_id) |> 
  tally() |> 
  left_join(tweet_no_rt_count, by = "author_id") |> 
  mutate(n.y = ifelse(is.na(n.y), 0, n.y)) |> 
  filter(n.y < 50)

## take all tweets of sub-50 users
tweets_unfiltered_users <- tweets_raw_full |> filter(author_id %in% tweets_count$author_id)

## Create the final tweet:
## take the no RT tweet df, bind with sub-50 users, then take the top 200 tweets from every user
tweets_final <- tweets_no_RT |> 
  bind_rows(tweets_unfiltered_users) |> 
  distinct() |> 
  mutate(exposure = public_metrics.like_count + public_metrics.retweet_count + public_metrics.reply_count) |> 
  group_by(author_id) |>
  slice_max(n = 200, order_by = desc(exposure)) |>
  ungroup() |> 
  arrange(author_id)

write_rds(x = tweets_final, file = "./data/tweets_final.rds")


# Feature Engineering ----
tweets_features <- read_rds("./data/tweets_final.rds")

## Basic Metrics ----
### Add new columns indicating no. of urls\mentions\hashtags
### note: using the purr package we map the length of the embedded df's.
### this is a workaround to return the number of rows in that df.
tweets_features <- tweets_features |> 
  rowwise() |> 
  mutate(n_url = mean(purrr::map_dbl(entities.urls, length)),
         n_mention = mean(purrr::map_dbl(entities.mentions, length)),
         n_hashtag = mean(purrr::map_dbl(entities.hashtags, length))) |> 
  mutate(n_url = ifelse(is.na(n_url), yes = 0, no = n_url),
         n_mention = ifelse(is.na(n_mention), yes = 0, no = n_mention),
         n_hashtag = ifelse(is.na(n_hashtag), yes = 0, no = n_hashtag)) |> 
  ungroup()
  

### Add number of words and characters
tweets_features <- tweets_features |>
  mutate(n_entities = sapply(purrr::map(tweets_features$entities.annotations, 1), length),
         n_words = sapply(text, function(s) stringr::str_count(s, "\\w+")) # letter/digit/underscore only
  )

## Sentiment Analysis ----
### Add sentiment analysis with "sentimentr" package
sentiments <- sentimentr::sentiment(text.var = sentimentr::get_sentences(tweets_features$text)) |> 
  group_by(element_id) |> 
  summarise(mean_sentiment = mean(sentiment))

### add the mean sentiment for every tweet
tweets_features <- tweets_features |> 
  mutate(sentiment_score = sentiments$mean_sentiment)


## Media Reliability features ----
### Read the 'media_reliability' dataframe and normalize the urls
media_reliability <- read_tsv("./data/media_reliability.tsv") |> 
  mutate(url_name = str_extract(source_url_normalized, pattern = "[^.]+"))

### Create new columns in the df to store the factuality and bias
tweets_features$url_fact <- NA
tweets_features$url_bias <- NA

### find the index of tweets with >0 urls to iterate over them in the loop
index <- which(tweets_features$n_url > 0)

### run the loop
for (i in index) {
  # Pluck the url with purrr and normalize it
  url_clean <- purrr::pluck(tweets_features$entities.urls, i, "display_url", 1) |> 
                str_extract(pattern = "[^.]+")
  # fetch the factuality and bias metrics
  fact <- media_reliability$fact[media_reliability$url_name == url_clean]
  bias <- media_reliability$bias[media_reliability$url_name == url_clean]
  
  # if the values above are not empty, impute them into the dataframe
  if (length(fact) > 0) {
    tweets_features$url_fact[i] <- fact
    tweets_features$url_bias[i] <- bias
  }
}

### mutate the variables to integers
tweets_features <- tweets_features |> 
  mutate(url_bias = case_when(url_bias == "left" ~ "-2",
                              url_bias == "left-center" ~ "-1",
                              url_bias == "center" ~ "0",
                              url_bias == "right-center" ~ "1",
                              url_bias == "right" ~ "2",
                              url_bias == "extreme-right" ~ "3"),
         url_fact = case_when(url_fact == "high" ~ 1,
                              url_fact == "mixed" ~ 0,
                              url_fact == "low" ~ -1),
         url_bias = as.numeric(url_bias),
         url_fact = as.numeric(url_fact))

## Dictionary ----
### Use the liwcalike function to compute the dictionary simliarity scores
dict <- tweets_features$text |> 
  quanteda.dictionaries::liwcalike(quanteda.dictionaries::data_dictionary_LaverGarry)

### Bind the results with the main dataframe
tweets_features <- tweets_features |> 
  bind_cols(dict |> select(culture:values.liberal)) |> 
  bind_cols(dict |> select(Sixltr))

## Word Embeddings ----

### Because computing word embeddings takes ages on our machine,
### and because we still wanted to incorporate them, we decided to
### compute on the top tweet of each user.
most_exposure <- tweets_features |>
  group_by(author_id) |>
  slice(which.max(exposure)) |> 
  select(author_id, text) |> 
  ungroup()

### Clean the text to feed into the transformer model:
### Removing @mentions, #hashtags, and "amp" html code
most_exposure <- most_exposure |> 
  mutate(clean_text = str_remove_all(text, pattern = "@\\w+"),
         clean_text = str_remove_all(clean_text, pattern = "#\\w+"), 
         clean_text = str_remove_all(clean_text, pattern = "amp"),
         clean_text = tolower(clean_text))

### Compute the word embeddings and save them
for (i in 1:nrow(most_exposure)) {
  
full_word_embeddings_decon <- text::textEmbed(most_exposure$clean_text[i],
                                              contexts = TRUE,
                                              model = "cardiffnlp/twitter-roberta-base",
                                              decontexts = TRUE)

  write_rds(full_word_embeddings_decon, paste0("./data/word_embeddings/","we_",i,".rds"))
  print(paste(i,"_",nrow(most_exposure)))
}

# Read all the files and run a loop that combines them and attaches an identifaction row
files <- fs::dir_ls(path = "./data/word_embeddings/", glob = "*rds")

for (i in 1:length(files)){
  we_temp <- read_rds(files[i], id)[["x"]] |> 
  mutate(row = paste(str_extract_all(files[i], "[:digit:]")[[1]],sep="",collapse=""))

    if (i == 1) {we <- we_temp}
      else {we <- we |> bind_rows(we_temp)}
}

WE_most_exposure <- we |> 
  mutate(row = as.numeric(row)) |> 
  arrange(row)

### To reduce the dimensionality of the Word Embeddings
### we have decided to conduct a PCA and select the top 6
### after visually inspecting the scree plot
pca_embeds <- prcomp(WE_most_exposure |> select(!row))
factoextra::fviz_eig(pca_embeds) # view the percentage of variances explained by each principal component
selected_pcs <- select(as_tibble(pca_embeds$x), PC1:PC6)

# Save the engineered rds ----
write_rds(tweets_features, file = "./data/tweets_features.rds")



# Summarising features ----
tweets_features <- read_rds("./data/tweets_features.rds")

## Select the relevant columns and rename them
metric_data <- tweets_features |> 
  select(author_id, 
         "rt" = public_metrics.retweet_count, 
         "like" = public_metrics.like_count, 
         n_url,
         url_fact,
         url_bias,
         n_mention,
         n_hashtag,
         n_entities,
         n_words,
         sentiment_score,
         culture:Sixltr) |> 
  janitor::clean_names()

## Calculate statistics - Mean, Median, SD, Kurtosis, Skewness for every metric
### note: id's without any rt\reply\like\quote create NaN and NA's in some statistics
### note: for now, we are using mean only for simplicity's sake
metric_data <- metric_data |> 
  group_by(author_id) |> 
  summarise(across(rt:sixltr, list("mean" = ~mean(.x, na.rm=T)
                                            #"median" = ~median(.x, na.rm=T),                                             # "median" = ~median(.x, na.rm=T), 
                                            #"sd" = ~sd(.x, na.rm=T),
                                            #"kurtosis" = ~kurtosis(.x, na.rm=T), 
                                            #"skewness" = ~skewness(.x, na.rm=T)
                                            ))
  )

## Replace NAs with 0s
metric_data <- metric_data |>
  mutate(across(rt_mean:sixltr_mean, ~replace(., is.nan(.), 0))) 

# Visual examination ----
## examine the histograms of the variables, perhaps some should be dropped or transformed
histograms <- metric_data |> 
  pivot_longer(cols = rt_mean:sixltr_mean, names_to = "metric", values_to = "value")

histograms |> 
  ggplot(aes(x = value)) + geom_histogram(bins = 100) + facet_wrap(~metric, scales = "free")

## After the visual examination we have decided to drop 
## "culture_sport_mean" and "law_and_order_law_liberal_mean" due to very low variance
metric_data <- metric_data |> 
  select(!c(culture_sport_mean, law_and_order_law_liberal_mean))

## Furthermore, we decided to log-transform "rt_mean" and "like_mean"
## due to extreme values
metric_data <- metric_data |> 
  mutate(rt_mean = ifelse(rt_mean != 0, yes = log(rt_mean), no = log(rt_mean + 0.001)),
         like_mean = ifelse(like_mean != 0, yes = log(like_mean), no = log(like_mean + 0.001)))


## Finally, combine the PCs with the metric data
### Merge the word embedding PCs with the main df
metric_data <- metric_data |> 
  bind_cols(selected_pcs)

## Examine corrplot
cor <- cor(as.matrix(metric_data |> select(!author_id)))
corrplot(cor)

# Save final product
write_rds(x = metric_data, file = "./data/metric_data.rds")
