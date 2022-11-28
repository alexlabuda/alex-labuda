---
title: "Predicting customer lifetime value"
subtitle: ""
excerpt: "using tidymodels to predict customer lifetime value and identify churned customers"
date: 2022-11-27
author: "Alex Labuda"
draft: false
thumbnail_left: true # for list-sidebar only
show_author_byline: true
images: 
series:
tags: ["machine learning", "data wrangling"]
categories:
layout: single
---


# ACE Fitness Customer Lifetime Value ----
# CLV with Machine Learning using RFM Features ----
# **** ----

# Libraries ----

library(tidymodels)
library(vip)
library(tidyverse)
library(timetk)
library(lubridate)
library(gt)

# Data Prep ----

ace_raw <- read_csv("data/ace_cust_sales.csv")

ace_raw

# * Data Cleanup -----
cdnow_tbl <- ace_raw %>%
    select(date, customer_id, order_total) %>%
    set_names(
        c("date", "customer_id", "price")
    ) %>%
    mutate(date = mdy(as.character(date))) %>%
    mutate(date = ymd(as.character(date))) %>%
    filter(date >= ymd("2022-02-26"))            # get exactly 180 days of data
    # mutate(price = as.double(price)) %>%
    drop_na()



# 2.0 COHORT ANALYSIS ----
# - Only the customers that have joined at the specific business day

# * Get Range of Initial Purchases ----
cdnow_first_purchase_tbl <- cdnow_tbl %>%
    group_by(customer_id) %>%
    slice_min(date) %>%
    ungroup()

cdnow_first_purchase_tbl %>%
    pull(date) %>%
    range()

# "2022/01/13" "2022/08/09"

as.double(difftime(ymd("2022/02/26"),
                   ymd("2022/05/27")))

# * Set Cohort Span ----
#   - Set initial purchase: 1/26/2022 "5/27/2022
ids_in_cohort <- cdnow_first_purchase_tbl %>%
    filter_by_time(
        .start_date = "2022-02-26",
        .end_date   = "2022-05-27"
    ) %>%
    distinct(customer_id) %>%
    pull(customer_id)

cdnow_cohort_tbl <- cdnow_tbl %>%
    filter(customer_id %in% ids_in_cohort)


# * Visualize: Spend Distribution ----
tidymodels_prefer()

cdnow_cohort_tbl %>%
    select(price) %>%
    filter(price > 1000) %>%
    ggplot(aes(x = price)) +
    geom_histogram(binwidth = 250, col = "white")

cdnow_cohort_tbl %>%
    select(price) %>%
    filter(price > 0) %>%
    ggplot(aes(x = price)) +
    geom_histogram(bins = 50, col = "white") +
    scale_x_log10()

ymin <- min(cdnow_cohort_tbl$price)

y25 <- quantile(cdnow_cohort_tbl$price, 0.25)

y50 <- median(cdnow_cohort_tbl$price)

y75 <- quantile(cdnow_cohort_tbl$price, 0.75)

ymax <- max(cdnow_cohort_tbl$price)

iqr <- y75 - y25

upper_fence <- y75 + 8 * iqr


cdnow_cohort_tbl <- cdnow_cohort_tbl %>%
    mutate(price = ifelse(price > upper_fence, upper_fence, price))

cdnow_cohort_tbl %>% skimr::skim()

# * Visualize: Total Cohort Purchases ----

cdnow_cohort_tbl %>%
    summarize_by_time(
        total_price = sum(price, na.rm = TRUE),
        .by   = "month"
    ) %>%
    plot_time_series(date, total_price, .y_intercept = 0)

# * Visualize: Individual Customer Purchases ----
n    <- 1:10
ids  <- unique(cdnow_cohort_tbl$customer_id)[n]

cdnow_cohort_tbl %>%
    filter(customer_id %in% ids) %>%
    group_by(customer_id) %>%
    plot_time_series(
        date, price,
        .y_intercept = 0,
        .smooth      = FALSE,
        .facet_ncol  = 2,
        .interactive = FALSE,
        .title = "Customer Purchase Behavior"
    ) +
    geom_point(color = "#2c3e50")

# 3.0 MACHINE LEARNING ----
#  Questions:
#  - What will the customers spend in the next 90-Days? (Regression)
#  - What is the probability of a customer to make a purchase in next 90-days? (Classification)


# 3.1 SPLITTING (2-Stages) ----

# ** Stage 1: Random Splitting by Customer ID ----

set.seed(123)
ids_train <- cdnow_cohort_tbl %>%
    pull(customer_id) %>%
    unique() %>%
    sample(size = round(0.8*length(.))) %>%
    sort()

split_1_train_tbl <- cdnow_cohort_tbl %>%
    filter(customer_id %in% ids_train)

split_1_test_tbl  <- cdnow_cohort_tbl %>%
    filter(!customer_id %in% ids_train)


# ** Stage 2: Time Splitting ----

splits_2_train <- time_series_split(
    split_1_train_tbl,
    assess     = "90 days",
    cumulative = TRUE
)

splits_2_train %>%
    tk_time_series_cv_plan() %>%
    plot_time_series_cv_plan(date, price)

splits_2_test <- time_series_split(
    split_1_test_tbl,
    assess     = "90 days",
    cumulative = TRUE
)

splits_2_test %>%
    tk_time_series_cv_plan() %>%
    plot_time_series_cv_plan(date, price)


# 3.2 FEATURE ENGINEERING (RFM) ----
#   - Need to think about what features to include

# ** Make in-sample targets from training data ----
targets_train_tbl <- testing(splits_2_train) %>%
    group_by(customer_id) %>%
    summarise(
        spend_90_total = sum(price)
        # ,
        # spend_90_flag    = 1
    ) %>%
    mutate(spend_90_flag = ifelse(spend_90_total > 0, 1, 0))

# ** Make out-sample targets from testing(splits_2) ----
targets_test_tbl <- testing(splits_2_test) %>%
    group_by(customer_id) %>%
    summarise(
        spend_90_total = sum(price)
        # ,
        # spend_90_flag    = 1
    ) %>%
    mutate(spend_90_flag = ifelse(spend_90_total > 0, 1, 0))

# ** Make Training Data ----
#    - RFM: Recency, Frequency, Monetary

max_date_train <- training(splits_2_train) %>%
    pull(date) %>%
    max()

train_tbl <- training(splits_2_train) %>%
    group_by(customer_id) %>%
    summarise(
        recency   = (max(date) - max_date_train) / ddays(1),
        frequency = n(),
        price_sum   = sum(price, na.rm = TRUE),
        price_mean  = mean(price, na.rm = TRUE)
    ) %>%
    left_join(
        targets_train_tbl
    ) %>%
    replace_na(replace = list(
        spend_90_total = 0,
        spend_90_flag  = 0
    )
    ) %>%
    mutate(spend_90_flag = as.factor(spend_90_flag))

# ** Make Testing Data ----
#    - Repeat for testing data
#    - Need full customer history: training splits 1 and 2
test_tbl <- training(splits_2_test) %>%
    group_by(customer_id) %>%
    summarise(
        recency     = (max(date) - max_date_train) / ddays(1),
        frequency   = n(),
        price_sum   = sum(price, na.rm = TRUE),
        price_mean  = mean(price, na.rm = TRUE)
    ) %>%
    left_join(
        targets_test_tbl
    ) %>%
    replace_na(replace = list(
        spend_90_total = 0,
        spend_90_flag  = 0
    )
    ) %>%
    mutate(spend_90_flag = as.factor(spend_90_flag))

# 3.3 RECIPES ----

# ** Model 1: 90-Day Spend Prediction ----
recipe_spend_total <- recipe(spend_90_total ~ ., data = train_tbl) %>%
    step_center() %>%
    step_rm(spend_90_flag, customer_id)


# ** Model 2: 90-Day Spend Probability ----
recipe_spend_prob <- recipe(spend_90_flag ~ ., data = train_tbl) %>%
    step_center() %>%
    step_rm(spend_90_total, customer_id)

recipe_spend_prob %>% prep() %>% juice() %>% glimpse()

summary(recipe_spend_prob)

# 3.4 MODELS ----

# ** Model 1: 90-Day Spend Prediction ----
wflw_spend_total_xgb <- workflow() %>%
    add_model(
        boost_tree(
            mode = "regression"
        ) %>%
            set_engine("xgboost")
    ) %>%
    add_recipe(recipe_spend_total) %>%
    fit(train_tbl)

# ** Model 2: 90-Day Spend Probability ----
wflw_spend_prob_xgb <- workflow() %>%
    add_model(
        boost_tree(
            mode = "classification"
        ) %>%
            set_engine("xgboost")
    ) %>%
    add_recipe(recipe_spend_prob) %>%
    fit(train_tbl)

# 3.5 TEST SET EVALUATION ----

# * Make Test Predictions ----

predictions_test_tbl <-  bind_cols(

    predict(wflw_spend_total_xgb, test_tbl) %>%
        rename(.pred_total = .pred),

    predict(wflw_spend_prob_xgb, test_tbl, type = "prob") %>%
        select(.pred_1) %>%
        rename(.pred_prob = .pred_1)
) %>%
    bind_cols(test_tbl) %>%
    select(starts_with(".pred"), starts_with("spend_"), everything())

# * Model Test Accuracy ----

predictions_test_tbl %>%
    yardstick::mae(spend_90_total, .pred_total)

predictions_test_tbl %>%
    yardstick::roc_auc(spend_90_flag, .pred_prob, event_level = "second")

predictions_test_tbl %>%
    yardstick::roc_curve(spend_90_flag, .pred_prob, event_level = "second")%>%
    autoplot()

# 3.6 FEATURE IMPORTANCE ----

# * Probability Model ----
vip(wflw_spend_prob_xgb$fit$fit,
    aesthetics = list(color = "grey50", fill = "grey50"))

# * Spend Model ----
vip(wflw_spend_total_xgb$fit$fit,
    aesthetics = list(color = "grey50", fill = "grey50"))


# 3.7 SAVE WORK ----

fs::dir_create("ace/artifacts")

wflw_spend_prob_xgb %>% write_rds("ace/artifacts/ace_model_prob.rds")
wflw_spend_total_xgb %>% write_rds("ace/artifacts/ace_model_spend.rds")

vi_model(wflw_spend_prob_xgb$fit$fit) %>% write_rds("ace/artifacts/ace_vi_prob.rds")
vi_model(wflw_spend_total_xgb$fit$fit) %>% write_rds("ace/artifacts/ace_vi_spend.rds")

all_tbl <- bind_rows(train_tbl, test_tbl)
predictions_all_tbl <- bind_cols(
    predict(wflw_spend_total_xgb, all_tbl) %>%
        rename(.pred_total = .pred),
    predict(wflw_spend_prob_xgb, all_tbl, type = "prob") %>%
        select(.pred_1) %>%
        rename(.pred_prob = .pred_1)
) %>%
    bind_cols(all_tbl) %>%
    select(starts_with(".pred"), starts_with("spend_"), everything())

predictions_all_tbl %>% write_rds("ace/artifacts/ace_predictions_all_tbl.rds")


# 4.0 PUTTING THE ANALYSIS TO WORK ----

# ** Which customers have the highest spend probability in next 90-days? ----
#    - Target for new products similar to what they have purchased in the past
predictions_test_tbl %>%
    arrange(desc(.pred_prob))

# ** Which customers have recently purchased but are unlikely to buy? ----
#    - Incentivize actions to increase probability
#    - Provide discounts, encourage referring a friend, nurture by letting them know what's coming
predictions_test_tbl %>%
    filter(
        recency    > -50,
        .pred_prob < 0.2
    ) %>%
    arrange(.pred_prob)

# ** Missed opportunities: Big spenders that could be unlocked ----
#    - Send bundle offers encouraging volume purchases
#    - Focus on missed opportunities
predictions_test_tbl %>%
    arrange(desc(.pred_total)) %>%
    filter(
        spend_90_total == 0
    )


# 5.0   REPORT-READY DATA TABLES ----

prediction_tbl_1 <- readr::read_rds("ace/artifacts/ace_predictions_all_tbl.rds")


prediction <-prediction_tbl_1 %>%
    arrange(desc(.pred_total)) %>%
    filter(spend_90_total < 0.6 * .pred_total) %>%
    filter(.pred_total > 45) %>%
    select(spend_90_total,  .pred_total, customer_id, recency, frequency, price_sum, price_mean) %>%

    rename(
        "90-Day Spend (Predict)"      = .pred_total,
        "90-Day Spend (Actual)"       = spend_90_total,
        # "Predicted Spend Probability" = .pred_prob,
        "Cust ID"                     = customer_id,
        "Monetary (Total)"            = price_sum,
        "Monetary (Avg)"              = price_mean
    ) %>%
    rename_all(
        .funs = ~ str_replace_all(., "_", " ") %>%
            str_to_title()

    )


big_spenders <- prediction %>%
    # filter("90-Day Spend (Predict)" >= 150.00) %>%
    gt() %>%
    tab_header(
        title = md("**ACE Fitness -** Customer Analytics"),
        subtitle = md("**Predicted Big Spenders**")
    ) %>%
    fmt_number(
        columns = c(Recency, Frequency)
    ) %>%
    fmt_currency(
        columns = c("90-Day Spend (Actual)", "90-Day Spend (Predict)",
                    "Monetary (Total)", "Monetary (Avg)")
    ) %>%
    fmt_passthrough(
        columns = c("Cust Id")
    ) %>%
    tab_source_note(
        source_note = md("Source: Tealium EventDB")
    )
