library(caret)
library(dplyr)
library(ggplot2)

set.seed(123)

train_data <- read.csv("train_ch.csv")
test_data <- read.csv("test_ch.csv")

# head(train_data)      # View the first few rows of the data
# summary(train_data)   # Get summary statistics for each variable
# str(train_data)       # View the structure of the data frame

# Check for missing values in the training data
# sum(is.na(train_data))

# Check for missing values in the test data
# sum(is.na(test_data))

# Scatter plots of variables against the response variable using ggplot2
ggplot(train_data, aes(x = v1, y = Y)) +
  geom_point() +
  ggtitle("v1 vs. Y")

ggplot(train_data, aes(x = v2, y = Y)) +
  geom_point() +
  ggtitle("v2 vs. Y")

ggplot(train_data, aes(x = v3, y = Y)) +
  geom_point() +
  ggtitle("v3 vs. Y")

ggplot(train_data, aes(x = v4, y = Y)) +
  geom_point() +
  ggtitle("v4 vs. Y")

ggplot(train_data, aes(x = v5, y = Y)) +
  geom_point() +
  ggtitle("v5 vs. Y")

ggplot(train_data, aes(x = v6, y = Y)) +
  geom_point() +
  ggtitle("v6 vs. Y")

ggplot(train_data, aes(x = v7, y = Y)) +
  geom_point() +
  ggtitle("v7 vs. Y")

ggplot(train_data, aes(x = v8, y = Y)) +
  geom_point() +
  ggtitle("v8 vs. Y")

ggplot(train_data, aes(x = v9, y = Y)) +
  geom_point() +
  ggtitle("v9 vs. Y")


# Parametric approach
lm_model <- train(Y ~ v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9,
                  data = train_data,
                  method = "lm",
                  trControl = trainControl(method = "cv"))

lm_predictions <- predict(lm_model, newdata = test_data)

# summary(lm_model)

lm_model_v3 <- train(Y ~ v3,
                      data = train_data,
                      method = "lm",
                      trControl = trainControl(method = "cv"))

lm_predictions_v3 <- predict(lm_model_v3, newdata = test_data)
# lm_predictions_v3_train <- predict(lm_model_v3, newdata = train_data)

# summary(lm_model_v3)

# Nonparametric approach
knn_model <- train(Y ~ v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9,
                   data = train_data,
                   method = "knn",
                   trControl = trainControl(method = "cv"),
                   tuneGrid = expand.grid(k = 1:10))


knn_predictions <- predict(knn_model, newdata = test_data)

knn_model_v3 <- train(Y ~ v3,
                   data = train_data,
                   method = "knn",
                   trControl = trainControl(method = "cv"),
                   tuneGrid = expand.grid(k = 1:10))

knn_predictions_v3 <- predict(knn_model_v3, newdata = test_data)
# knn_predictions_v3_train <- predict(knn_model_v3, newdata = train_data)


predictions <- data.frame(pred_knn = knn_predictions_v3, pred_lm = lm_predictions_v3)

write.csv(predictions, file = "predictions.csv", row.names = FALSE)

print(knn_model_v3$results)
print(lm_model_v3$results)


