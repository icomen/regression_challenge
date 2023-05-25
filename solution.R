library(caret)
library(dplyr)
library(ggplot2)
library(corrplot)
library(car)
library(FactoMineR)
library(factoextra)
library(visreg)
library(FNN)
library(e1071)


set.seed(123)

## Step 1: Data Exploration and Analysis

train_data <- read.csv("train_ch.csv")
test_data <- read.csv("test_ch.csv")

# summary(train_data)   # Get summary statistics for each variable
# str(train_data)       # View the structure of the data frame
#
# summary(test_data)   # Get summary statistics for each variable
# str(test_data)       # View the structure of the data frame

vars <- c("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "Y")
vars_of_interest <- c("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9")
vars_no_corr <- c("v1", "v2", "v3", "v4", "v6", "v8", "v9")

## Step 2: Plot Predictors and Check Collinearity

# Scatter plots of variables against the response variable using ggplot2
for (var in vars_of_interest) {
  plot_title <- paste(var, "vs. Y")
  p <- ggplot(train_data, aes(x = get(var), y = Y)) +
    geom_point() +
    ggtitle(plot_title)
  print(p)
}

# Create a pair plot using the 'pairs()' function
pairs(train_data[, vars])

# Correlation matrix
cor_matrix <- cor(train_data[, vars_of_interest])
# print(cor_matrix)

# Plot the correlation matrix using corrplot
corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)

# Create a heatmap using the 'heatmap()' function
heatmap(cor_matrix,
        col = colorRampPalette(c("blue", "white", "red"))(100),
        main = "Correlation Heatmap",
        xlab = "Variables",
        ylab = "Variables")

# Loop through each variable and create distribution plots
for (var in vars) {
  # Create a histogram using the 'hist()' function
  hist(train_data[[var]], main = paste("Histogram of", var), xlab = var)

  # Alternatively, create a density plot using 'geom_density()' in ggplot2
  ggplot(train_data, aes(x = get(var))) +
    geom_density(fill = "lightblue", alpha = 0.6) +
    labs(title = paste("Density Plot of", var), x = var, y = "Density")
}

# Perform Principal Component Analysis (PCA)
pca_result <- PCA(train_data[, vars_of_interest], graph = FALSE)

# Create a correlation circle plot
corr_circle <- fviz_pca_var(pca_result, axes = c(1, 2), col.var = "black")

# Display the correlation circle plot
print(corr_circle)

## Step 3: Parametric Linear Regression Model

lm_model <- lm(Y ~ v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9, data = train_data)

# lm_model <- train(Y ~ v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9,
#                   data = train_data,
#                   method = "lm",
#                   trControl = trainControl(method = "cv"))

# View the summary of the linear regression model
summary(lm_model)

# Calculate the Variance Inflation Factor (VIF)
vif_values <- vif(lm_model)
print(vif_values)

lm_predictions <- predict(lm_model, newdata = test_data)

## Step 4: Check for Nonlinearity in Predictors

# We can assess nonlinearity using the scatterplot of the predictor against the response variable and density plots

for (var in vars_of_interest) {
   # Plot the relationship between the predictor and the response variable
  visreg(lm_model, var, scale = "response", line = list(col = "blue"), partial = FALSE,
         xlab = var, ylab = "Response")
}

## Step 5: Check Violation of Assumptions and Synergic Effects (if applicable)

# Check linearity assumption
# Residual plot
plot(lm_model, which = 1)

# Check normality of residuals assumption
# Q-Q plot
plot(lm_model, which = 2)

# Check homoscedasticity assumption
# Residuals vs. Fitted values plot
plot(lm_model, which = 3)

# Check influential observations
# Leverage plot
plot(lm_model, which = 5)

# Alternatively, you can also use the 'car' package for leverage plots
avPlots(lm_model)

# Compute and analyze the Cook's distance
cooksd <- cooks.distance(lm_model)
plot(cooksd, pch = "*", cex = 1.5, main = "Influential Observations by Cook's distance")

# Identify influential observations
influential <- as.numeric(names(cooksd)[cooksd > 4 / length(cooksd)])
print(influential)

# lm_model_v3 <- train(Y ~ v3,
#                       data = train_data,
#                       method = "lm",
#                       trControl = trainControl(method = "cv"))
#
# lm_predictions_v3 <- predict(lm_model_v3, newdata = test_data)

# # Tune and select the optimal value of k (number of neighbors)
# k <- tune.knn(train = train_data[, -response_column_index], test = test_data, cl = train_data$response, k = 1:10)$best.neighbors
#
# # Fit KNN model with the selected k value
# knn_model <- knn(train = train_data[, -response_column_index], test = test_data, cl = train_data$response, k = k)
#
# # View the predictions from the KNN model
# knn_model



# Nonparametric approach
knn_model <- train(Y ~ v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9,
                   data = train_data,
                   method = "knn",
                   trControl = trainControl(method = "cv"),
                   tuneGrid = expand.grid(k = 1:10))


knn_predictions <- predict(knn_model, newdata = test_data)

# knn_model_v3 <- train(Y ~ v3,
#                    data = train_data,
#                    method = "knn",
#                    trControl = trainControl(method = "cv"),
#                    tuneGrid = expand.grid(k = 1:10))
#
# knn_predictions_v3 <- predict(knn_model_v3, newdata = test_data)


predictions <- data.frame(pred_knn = knn_predictions, pred_lm = lm_predictions)

write.csv(predictions, file = "predictions.csv", row.names = FALSE)

# print(knn_model_v3$results)
# print(lm_model_v3$results)


