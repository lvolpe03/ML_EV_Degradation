###############################################
# PECN 6180 - Regression Project - Luke Volpe #
###############################################

# Goal: predict EV battery health (SoH_Percent) from all other variables
# Dataset URL: https://www.kaggle.com/datasets/bertnardomariouskono/electric-vehicle-ev-battery-degradation-and-charge/data

library(tidyverse)
library(rsample)
library(caret)
library(glmnet)
library(splines)
library(ggplot2)

# Load in .csv
EVdf <- read.csv("ev_battery_degradation_v1.csv")
# Convert categorical features to factors
EVdf$Car_Model <- as.factor(EVdf$Car_Model)
EVdf$Battery_Type <- as.factor(EVdf$Battery_Type)
EVdf$Driving_Style <- as.factor(EVdf$Driving_Style)
EVdf$Battery_Capacity_kWh <- as.factor(EVdf$Battery_Capacity_kWh)
# Remove variables we won't regress on (Car ID and a classification target)
EVdf <- select(EVdf, -c(Vehicle_ID, Battery_Status))

# Here, we can see that Car_Model perfectly explains both Battery_Type and 
# Battery_Capacity_kWh. Also, Internal_Resistance_Ohm is near perfectly 
# correlated with Y. Therefore, we can drop them all as well. 
table(EVdf$Car_Model, EVdf$Battery_Type)
table(EVdf$Car_Model, EVdf$Battery_Capacity_kWh)
cor(EVdf$Internal_Resistance_Ohm, EVdf$SoH_Percent)
EVdf <- select(EVdf, -c(Battery_Type, Battery_Capacity_kWh, Internal_Resistance_Ohm))

# Generate new features:
# Degrees above optimal operating range (27 C)
EVdf$temp_high <- pmax(0, EVdf$Avg_Temperature_C - 27)
# Interaction b/t Age and Total Cycles 
EVdf$age_cycles <- EVdf$Vehicle_Age_Months * EVdf$Total_Charging_Cycles

# Summary Statistics
sum_stats <- summary(EVdf)
sum_stats

set.seed(100)

# Train-Test Split
split <- initial_split(EVdf, prop = 0.75)
train_data <- training(split)
test_data <- testing(split)

# Set-up CV control
cv10 <- trainControl(method = "cv", number = 10)

# Reproducibility
set.seed(100)

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
# MODEL SELECTION
# Model 1: Basic OLS (all linear variables)
set.seed(100)
ols_1 <- train(SoH_Percent ~ . - age_cycles - temp_high, data = train_data, method = "lm", trControl = cv10)

# Model 2: OLS plus age*cycles interaction and temp_high
set.seed(100)
ols_2 <- train(SoH_Percent ~ . , data = train_data, method = "lm", trControl = cv10)

# Model 3: OLS plus age_cycles and temp^2
set.seed(100)
ols_3 <- train(SoH_Percent ~ . + (Avg_Temperature_C^2) - temp_high, data = train_data, method = "lm", trControl = cv10)

# Model 4: Ridge (alpha = 0, lambda=0.1) 
set.seed(100)
ridge_l01 <- train(SoH_Percent ~ . , data = train_data, method = "glmnet", trControl = cv10, 
                   tuneGrid = data.frame(alpha = 0, lambda = 0.1))

# Model 5: Ridge (alpha = 0, lambda=5) 
set.seed(100)
ridge_l5 <- train(SoH_Percent ~ . , data = train_data, method = "glmnet", trControl = cv10,
                   tuneGrid = data.frame(alpha = 0, lambda = 5))

# Model 6: Lasso (alpha = 1, lambda=0.1) 
set.seed(100)
lasso_l01 <- train(SoH_Percent ~ . , data = train_data, method = "glmnet", trControl = cv10,
                   tuneGrid = data.frame(alpha = 1, lambda = 0.1))

# Model 7: Lasso (alpha = 1, lambda=5) 
set.seed(100)
lasso_l5 <- train(SoH_Percent ~ . , data = train_data, method = "glmnet", trControl = cv10,
                   tuneGrid = data.frame(alpha = 1, lambda = 5))

# Model 8: Elastic Net, alpha = 0.25 (closer to ridge), lambda = 0.1
set.seed(100)
enet_a25_l01 <- train(SoH_Percent ~ . , data = train_data, method = "glmnet", trControl = cv10,
                      tuneGrid = data.frame(alpha = 0.25, lambda = 0.1))

# Model 9: Elastic Net, alpha = 0.25 (closer to ridge), lambda = 5 
set.seed(100)
enet_a25_l5 <- train(SoH_Percent ~ . , data = train_data, method = "glmnet", trControl = cv10,
                      tuneGrid = data.frame(alpha = 0.25, lambda = 5))

# Model 10: Elastic Net, alpha = 0.75 (closer to lasso), lambda = 0.1 
set.seed(100)
enet_a75_l01 <- train(SoH_Percent ~ . , data = train_data, method = "glmnet", trControl = cv10,
                      tuneGrid = data.frame(alpha = 0.75, lambda = 0.1))

# Model 11: Elastic Net, alpha = 0.75 (closer to lasso), lambda = 5
set.seed(100)
enet_a75_l5 <- train(SoH_Percent ~ . , data = train_data, method = "glmnet", trControl = cv10,
                      tuneGrid = data.frame(alpha = 0.75, lambda = 5))

# Model 12: Cubic Spline of Temperature
set.seed(100)
spline_formula1 <- SoH_Percent ~ ns(Avg_Temperature_C, df = 4) + 
  Car_Model + Fast_Charge_Ratio + Total_Charging_Cycles + 
    Avg_Discharge_Rate_C + Driving_Style + age_cycles + temp_high + Vehicle_Age_Months
spline_1 <- train(spline_formula1,
                   data = train_data,
                   method = "lm",
                   trControl = cv10)

# Model 13: Quartic Spline of Temperature
set.seed(100)
spline_formula2 <- SoH_Percent ~ ns(Avg_Temperature_C, df = 5) + 
  Car_Model + Fast_Charge_Ratio + Total_Charging_Cycles +
  Avg_Discharge_Rate_C + Driving_Style + age_cycles + temp_high + Vehicle_Age_Months
spline_2 <- train(spline_formula2,
                  data = train_data,
                  method = "lm",
                  trControl = cv10)

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
### MODEL SELECTION - TRAINING RMSE
cat("OLS:                       ", round(ols_1$results$RMSE, 4), "\n")
cat("OLS (inter + high):        ", round(ols_2$results$RMSE, 4), "\n")
cat("OLS (inter + temp2):       ", round(ols_3$results$RMSE, 4), "\n")
cat("Ridge (lambda = 0.1):      ", round(ridge_l01$results$RMSE, 4), "\n")
cat("Ridge (lambda = 5):        ", round(ridge_l5$results$RMSE, 4), "\n")
cat("Lasso (lambda = 0.1):      ", round(lasso_l01$results$RMSE, 4), "\n")
cat("Lasso (lambda = 5):        ", round(lasso_l5$results$RMSE, 4), "\n")
cat("Elastic Net (a=.25, l=.1): ", round(enet_a25_l01$results$RMSE, 4), "\n")
cat("Elastic Net (a=.25, l=5):  ", round(enet_a25_l5$results$RMSE, 4), "\n")
cat("Elastic Net (a=.75, l=.1): ", round(enet_a75_l01$results$RMSE, 4), "\n")
cat("Elastic Net (a=.75, l=5):  ", round(enet_a75_l5$results$RMSE, 4), "\n")
cat("Temp Spline (Cubic):       ", round(spline_1$results$RMSE, 4), "\n")
cat("Temp Spline (Quartic):     ", round(spline_2$results$RMSE, 4), "\n")

trainRMSE_table <- data.frame(
  Model = c("OLS (linear terms only)", "OLS (age_cycles + temp_high)", "OLS (age_cycles + temp^2)", 
                  "Ridge (lambda = 0.1)", "Ridge (lambda = 5)", "Lasso (lambda = 0.1)", 
                  "Lasso (lambda = 5)", "Elastic Net (a=.25, l=.1)", "Elastic Net (a=.25, l=5)",
                  "Elastic Net (a=.75, l=.1)", "Elastic Net (a=.75, l=5)",
                  "Avg_Temp Spline (Cubic)", "Avg_Temp Spline (Quartic)"),
  Training_RMSE = c(round(ols_1$results$RMSE, 4), round(ols_2$results$RMSE, 4), round(ols_3$results$RMSE, 4),
                    round(ridge_l01$results$RMSE, 4), round(ridge_l5$results$RMSE, 4),
                    round(lasso_l01$results$RMSE, 4), round(lasso_l5$results$RMSE, 4),
                    round(enet_a25_l01$results$RMSE, 4), round(enet_a25_l5$results$RMSE, 4),
                    round(enet_a75_l01$results$RMSE, 4), round(enet_a75_l5$results$RMSE, 4),
                    round(spline_1$results$RMSE, 4), round(spline_2$results$RMSE, 4)
                    ))
trainRMSE_table

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
### Best Model Selection
best_model <- spline_1
best_ols <- ols_2
best_ridge <- ridge_l01
best_lasso <- lasso_l01
best_enet <- enet_a25_l01

# Get test RMSEs
ols_preds <- predict(best_ols, newdata = test_data)
ols_rmse <- sqrt(mean((test_data$SoH_Percent - ols_preds)^2))
ridge_preds <- predict(best_ridge, newdata = test_data)
ridge_rmse <- sqrt(mean((test_data$SoH_Percent - ridge_preds)^2))
lasso_preds <- predict(best_lasso, newdata = test_data)
lasso_rmse <- sqrt(mean((test_data$SoH_Percent - lasso_preds)^2))
enet_preds <- predict(best_enet, newdata = test_data)
enet_rmse <- sqrt(mean((test_data$SoH_Percent - enet_preds)^2))
best_preds <- predict(best_model, newdata = test_data)
best_rmse <- sqrt(mean((test_data$SoH_Percent - best_preds)^2))

cat("Best OLS Test Set RMSE:          ",ols_rmse,"\n")
cat("Best Ridge Test Set RMSE:        ",ridge_rmse,"\n")
cat("Best Lasso Test Set RMSE:        ",lasso_rmse,"\n")
cat("Best Elastic Net Test Set RMSE:  ",enet_rmse,"\n")
cat("Best Spline Test Set RMSE:       ",best_rmse,"\n")

# Number of predictors
ols_coefcount <- length(coef(best_ols$finalModel)) - 1
ols_coefcount
ridge_coefcount <- length(coef(best_ridge$finalModel, s = best_ridge$bestTune$lambda)) - 1
ridge_coefcount
lasso_coefs <- coef(best_lasso$finalModel, s = best_lasso$bestTune$lambda)
lasso_coefcount <- sum(lasso_coefs != 0) - 1
lasso_coefcount
enet_coefs <- coef(best_enet$finalModel, s = best_enet$bestTune$lambda)
enet_coefcount <- sum(enet_coefs != 0) - 1
enet_coefcount
spline_coefcount <- length(coef(spline_1$finalModel)) - 4
spline_coefcount

# Build results table
results_table <- data.frame(
  Best_Models = c("OLS", "Ridge", "Lasso", "Elastic Net", "Cubic Spline"),
  Training_RMSE = c(round(best_ols$results$RMSE, 4), round(best_ridge$results$RMSE, 4), 
                    round(best_lasso$results$RMSE, 4), round(best_enet$results$RMSE, 4),
                    round(best_model$results$RMSE, 4)),
  Test_RMSE = c(round(ols_rmse, 4), round(ridge_rmse, 4), round(lasso_rmse, 4), 
                round(enet_rmse, 4), round(best_rmse, 4)),
  Predictors = c(ols_coefcount, ridge_coefcount, lasso_coefcount, enet_coefcount, spline_coefcount))
results_table

# Final coefficients
summary(best_model)

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### VISUALIZATIONS #####
# Spline Plot
Temp_range <- seq(min(train_data$Avg_Temperature_C), max(train_data$Avg_Temperature_C), length.out = 200)
# We can't keep categorical control variables at a mean, so we must get the 
# mode and use that for the spline. I made this short function to calculate it:
Mode <- function(x) {
  names(sort(table(x), decreasing = TRUE))[1]}

spline_data <- data.frame(
  Avg_Temperature_C = Temp_range,
  Car_Model = Mode(train_data$Car_Model),
  Vehicle_Age_Months = mean(train_data$Vehicle_Age_Months),
  Fast_Charge_Ratio = mean(train_data$Fast_Charge_Ratio),
  Total_Charging_Cycles = mean(train_data$Total_Charging_Cycles),
  Avg_Discharge_Rate_C = mean(train_data$Avg_Discharge_Rate_C),
  Driving_Style = Mode(train_data$Driving_Style)
)
spline_data$temp_high  <- pmax(0, spline_data$Avg_Temperature_C - 27)
spline_data$age_cycles <- spline_data$Vehicle_Age_Months * spline_data$Total_Charging_Cycles

spline_data$predicted_SoH <- predict(spline_1, newdata = spline_data)
spline_plot <- ggplot() +
  geom_point(data = train_data, aes(x = Avg_Temperature_C, y = SoH_Percent),
             color = "gray50", size = 1, alpha = 0.5) +
  geom_line(data = spline_data, aes(x = Avg_Temperature_C, y = predicted_SoH),
            color = "#16A085", linewidth = 1.2) +
  geom_vline(xintercept = 27, linetype = "dashed", color = "black") +
  labs(x = "Average Operating Temperature (Celsius)",
       y = "Predicted Battery Health (% of new)",
       title = "Semi-Parametric Model: Estimated Effect of 
       Operating Temperature on Battery Health",
       subtitle = "Natural spline curve (df=4); other predictors held at means/modes") +
  theme_minimal()
spline_plot

# Actual vs. Predicted SoH (Spline model)
AvP_df <- data.frame(
  Actual = test_data$SoH_Percent,
  Predicted = best_preds)
AvP_plot <- ggplot(AvP_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.4, color = "turquoise3") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(
    title = "Predicted vs. Actual State of Health",
    x = "Actual SoH (%)",
    y = "Predicted SoH (%)",
    subtitle = "EV Battery Capacity (% of Original)") + theme_minimal()
AvP_plot

# Residual Plot
AvP_df$Residuals <- AvP_df$Actual - AvP_df$Predicted
Resid_plot <- ggplot(AvP_df, aes(x = Predicted, y = Residuals)) +
  geom_point(alpha = 0.4) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title = "Residual Plot – Cubic Spline Model",
    x = "Predicted SoH (%)",
    y = "Residuals"
  ) +
  theme_minimal()
Resid_plot

# Descriptive boxplot of SoH by Car Model
car_boxplot <- ggplot(train_data, aes(x = Car_Model, y = SoH_Percent)) +
  geom_boxplot() +
  labs(
    title = "Battery Health Distribution by Car Model",
    x = "Car Model",
    y = "State of Health (%)"
  ) + theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))
car_boxplot

# Basic Temp SoH scatterplot 
TempSoH <- ggplot() + geom_point(data = train_data, aes(x = Avg_Temperature_C, y = SoH_Percent),
                                color = "chocolate2", size = 1, alpha = 0.5) + labs(
                                  title = "Battery Health vs Vehicle Age",
                                  x = "# Charging Cycles",
                                  y = "State of Health (%)")
TempSoH

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 