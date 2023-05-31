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
library(mgcv)


set.seed(123)

## Step 1: Data Exploration and Analysis

train_data <- read.csv("train_ch.csv")
test_data <- read.csv("test_ch.csv")
test_data_new <- read.csv("test_completo_ch.csv")

# Extract the target variable (Y) from the datasets
y_train <- train_data$Y
y_test <- test_data_new$Y

# summary(train_data)   # Get summary statistics for each variable
# str(train_data)       # View the structure of the data frame
#
# summary(test_data)   # Get summary statistics for each variable
# str(test_data)       # View the structure of the data frame

vars <- c("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "Y")
vars_of_interest <- c("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9")
vars_no_coll <- c("v1", "v2", "v3", "v4", "v6", "v8", "v9")

subset_train <- train_data[, c("v1", "v2", "v3", "v4", "v6", "v8", "v9", "Y")]

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
for (var in vars_of_interest) {
  # Create a histogram using the 'hist()' function
  hist(train_data[[var]], main = paste("Histogram of", var), xlab = var)

  # Alternatively, create a density plot using 'geom_density()' in ggplot2
  # ggplot(train_data, aes(x = get(var))) +
  #   geom_density(fill = "lightblue", alpha = 0.6) +
  #   labs(title = paste("Density Plot of", var), x = var, y = "Density")
}

# Perform Principal Component Analysis (PCA)
pca_result <- PCA(train_data[, vars_of_interest], graph = FALSE)

# Create a correlation circle plot
corr_circle <- fviz_pca_var(pca_result, axes = c(1, 2), col.var = "black")

# Display the correlation circle plot
print(corr_circle)

## Step 3: Parametric Linear Regression Model

lm_model <- lm(Y ~ v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9, data = train_data)

# Create an empty list to store the regression models
regression_models <- list()

# Iterate over each predictor and fit a linear regression model
for (predictor in vars_of_interest) {
  formula <- as.formula(paste("Y ~", predictor))

  # Fit the linear regression model
  lm_model_v <- lm(formula, data = train_data)

  # Store the model in the list
  regression_models[[predictor]] <- lm_model_v
}

lm_model_v1 <- regression_models[["v1"]]
lm_model_v2 <- regression_models[["v2"]]
lm_model_v3 <- regression_models[["v3"]]
lm_model_v4 <- regression_models[["v4"]]
lm_model_v5 <- regression_models[["v5"]]
lm_model_v6 <- regression_models[["v6"]]
lm_model_v7 <- regression_models[["v7"]]
lm_model_v8 <- regression_models[["v8"]]
lm_model_v9 <- regression_models[["v9"]]

# View the summary of the linear regression model
# summary(lm_model)
# summary(lm_model_v1)
# summary(lm_model_v2)
# summary(lm_model_v3)
# summary(lm_model_v4)
# summary(lm_model_v5)
# summary(lm_model_v6)
# summary(lm_model_v7)
# summary(lm_model_v8)
# summary(lm_model_v9)

# Calculate the Variance Inflation Factor (VIF)
vif_values <- vif(lm_model)
# print(vif_values)


## Step 4: Check for Nonlinearity in Predictors

# We can assess nonlinearity using the scatterplot of the predictor against the response variable and histogram plots

for (var in vars_of_interest) {
   # Plot the relationship between the predictor and the response variable
  visreg(lm_model, var, scale = "response", line = list(col = "blue"), partial = FALSE,
         xlab = var, ylab = "Response")
}

## Step 5: Check Violation of Assumptions and Synergic Effects

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
# print(influential)

# Solve violations of assumptions:

# Logarithmic transformation
model_log <- lm(Y ~ log(v1 + v2 + v3 + v4 + v6 + v8 + v9), data = subset_train)

# Square root transformation
model_sqrt <- lm(Y ~ sqrt(v1 + v2 + v3 + v4 + v6 + v8 + v9), data = subset_train)

# Polynomial transformation
model_poly <- lm(Y ~ I(v1^4) + I(v2^4) + I(v3^2)+ v4 + v6 + v8 + I(v9^4) , data = subset_train)

# Compare model fits
anova(lm_model, model_log, model_sqrt, model_poly)

# Fit a GAM
model_gam <- gam(Y ~ s(v1) + s(v2) + v3 + s(v4) + s(v6) + s(v8) + s(v9), data = subset_train)

# Compare model fits
AIC(lm_model,model_log, model_sqrt, model_poly, model_gam)

# Results:
plot(model_poly, which = 1)
plot(model_poly, which = 2)
plot(model_poly, which = 3)
plot(model_poly, which = 5)
avPlots(model_poly)
cooksd <- cooks.distance(model_poly)
plot(cooksd, pch = "*", cex = 1.5, main = "Influential Observations by Cook's distance")
influential <- as.numeric(names(cooksd)[cooksd > 4 / length(cooksd)])
# print(influential)


model_lm0 <- train(Y ~ v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9,
                    data = train_data,
                    method = "lm",
                    trControl = trainControl(method = "cv"))

model_lm0$results

model_lm <- train(Y ~ v1 + v2 + v3 + v4 + v6 + v8 + v9,
                    data = subset_train,
                    method = "lm",
                    trControl = trainControl(method = "cv"))

model_lm$results



poly_model <- train(Y ~ I(v1^4) + I(v2^4) + I(v3^2)+ v4 + v6 + v8 + I(v9^4),
                    data = subset_train,
                    method = "lm",
                    trControl = trainControl(method = "cv"))

poly_model$results


poly_model2 <- train(Y ~ I(v1^2) + I(v2^2) + I(v3^2)+ I(v4^2) + I(v6^2) + I(v8^2) + I(v9^2),
                    data = subset_train,
                    method = "lm",
                    trControl = trainControl(method = "cv"))

poly_model2$results


knn_model0 <- train(Y ~ v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9,
                   data = train_data,
                   method = "knn",
                   trControl = trainControl(method = "cv"),
                   tuneGrid = expand.grid(k = 1:30))


knn_model0$results

knn_model1 <- train(Y ~ v1 + v2 + v3 + v4 + v6 + v8 + v9,
                   data = subset_train,
                   method = "knn",
                   trControl = trainControl(method = "cv"),
                   tuneGrid = expand.grid(k = 1:30))


knn_model1$results

knn_model <- train(Y ~ I(v1^4) + I(v2^4) + I(v3^2) + v4 + v6 + v8 + I(v9^4),
                   data = subset_train,
                   method = "knn",
                   trControl = trainControl(method = "cv"),
                   tuneGrid = expand.grid(k = 1:30))


knn_model$results


lm_predictions <- predict(model_lm, newdata = test_data)
poly_predictions <- predict(poly_model, newdata = test_data)
knn_predictions <- predict(knn_model, newdata = test_data)


mse <- mean((poly_predictions - y_test)^2)
print(mse)



mse_knn <- mean((knn_predictions - y_test)^2)
print(mse_knn)




predictions <- data.frame(pred_knn = knn_predictions, pred_lm = poly_predictions)

write.csv(predictions, file = "predictions.csv", row.names = FALSE)


