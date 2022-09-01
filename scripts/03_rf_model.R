library(caret)
library(tidyverse)
library(corrplot)

# Read the train data
train <- read_csv("./data/train.csv", col_types = "c")


# Read the feature engineered data and filter for train IDs only
metric_data <- read_rds("./data/metric_data.rds") |> 
  filter(author_id %in% train$id) |>
  left_join(train, by = c("author_id" = "id"))

# for now, impute 0 to NA and NAN values
#metric_data[is.na(metric_data)] <- 0

customSummary <- function (data, lev = NULL, model = NULL) {
  out <- cor(data$obs, data$pred)  
  names(out) <- "PearsonR"
  out
}


### FULL RF MODEL
training <- metric_data |> select(!author_id)

tg <- expand.grid(list(mtry = seq(from = 0, to = ncol(training)-1, by = 2)))

tc <- trainControl(method = "cv", 
                   number = 10,
                   summaryFunction = customSummary,
                   verboseIter = TRUE)

model <- caret::train(
  score ~ .,
  data = training,
  preProcess = c("center", "scale"),
  trControl = tc,
  tuneGrid = tg,
  metric = "PearsonR",
  method = "rf"
)

write_rds(model, "FULL_RF_MODEL_FINAL.rds")

### BASIC RF MODEL
training <- metric_data |> select(rt_mean, like_mean, n_url_mean, n_mention_mean, 
                                  n_hashtag_mean, n_entities_mean, n_words_mean,
                                  score)

tg <- expand.grid(list(mtry = seq(from = 3, to = ncol(training) - 1, by = 2)))

tc <- trainControl(method = "cv", 
                   number = 10,
                   summaryFunction = customSummary,
                   verboseIter = TRUE)

model_basic <- caret::train(
  score ~ .,
  data = training,
  preProcess = c("center", "scale"),
  trControl = tc,
  tuneGrid = tg,
  metric = "PearsonR",
  method = "rf"
)

write_rds(model_basic, "BASIC_RF_MODEL_FINAL.rds")



########### Model comparison
# tuning
df <- (model_basic$results |> mutate(model_type = "basic")) |> 
  bind_rows(model$results |> mutate(model_type = "full"))

ggplot(df, aes(x = mtry, y = PearsonR, color = model_type, group = model_type)) +
  geom_point()+
  geom_line()+
  labs(title = "Model Comparison",
       subtitle = "Full (NLP methods included) vs. Basic (No NLP methods)",
       y = "Cross-Validated Pearson's R",
       x = "No. of Randomly Selected Predictors Hyperparameter") +
  theme_bw()

# SHAP
#devtools::install_github('ModelOriented/treeshap')
library(treeshap)

model_unified <- randomForest.unify(model$finalModel,
                                    metric_data |> select(!author_id))

treeshap_res <- treeshap(model_unified, metric_data |> select(!author_id))

write_rds(treeshap_res, "data/treeshap_res.rds")

plot_contribution(treeshap_res, obs = )

#### MAKE PREDICTIONS

# Read the test data
test <- read_csv("./data/test.csv", col_types = "c")

# Read the feature engineered data and filter for test IDs only
testing <- read_rds("./data/metric_data.rds") |> 
  filter(author_id %in% test$id)



# Make predictions
testing$pred <- predict(model, newdata = testing)

submission <- test |> 
  left_join(testing |> select(author_id, pred), by = c("id" = "author_id")) |> 
  rename("score" = pred)

submission$score[is.na(submission$score)] <- median(submission$score, na.rm = TRUE)


submission <- submission |> mutate(id = 1:n())

write_csv(submission, "submission_full_pearsonR.csv")

