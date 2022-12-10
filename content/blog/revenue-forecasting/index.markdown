---
title: "Bootstrapped Resampling Regression & Revenue Forecasting"
subtitle: ""
excerpt: "We'll analyze regression coefficients with bootstrapped resamples, visualize most important features and forecast future revenue"
date: 2022-11-28
author: "Alex Labuda"
draft: false
thumbnail_left: true # for list-sidebar only
show_author_byline: true
images: 
series:
tags: ["machine learning", "regression", "feature importance", "time-series forecasting"]
categories:
layout: single
---

![](featured.jpg)

In this post, I’ll begin by visualizing marketing related time-series data which includes revenue and spend for several direct marketing channels.

I will fit a simple regression model and examine the effect of each channel on revenue.

In order to better understand the reliability of each coefficient, I will fit many bootstrapped resamples to develop a more robust estimate of each coefficient.

Lastly, I will create a forecasting model to predict future revenue. Enjoy!

``` r
library(modeltime)
library(timetk)
library(lubridate)

interactive <- FALSE
```

# The Data

``` r
df <- 
  read_csv(
    file = "data/marketing_data.csv"
  ) %>% 
  mutate(
    date = mdy(date)
  )
```

# Visualize the Data

First lets visualize our target variable, `revenue`

``` r
df %>% 
  ggplot(aes(date, revenue)) +
  geom_line() +
  labs(title = "Revenue",
       x = "")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-3-1.png" width="768" />

# Fit: Simple Regression Model

First we’ll start by fitting a simple regression model to the data. This is just the grand mean of our data.

``` r
df %>%
    plot_time_series_regression(
        .date_var     = date,
        .formula      = revenue ~ as.numeric(date),
        .interactive  = interactive,
        .show_summary = FALSE
    )
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-4-1.png" width="768" />

We can make a more accurate model by using month of year as independent variables for our model

``` r
df %>%
    plot_time_series_regression(
        .date_var     = date,
        .formula      = revenue ~ as.numeric(date) + month(date, label = TRUE),
        .interactive  = interactive,
        .show_summary = FALSE
    )
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-5-1.png" width="768" />

We can also introduce our spend variables as input to our model to study each channel’s effect on revenue

``` r
df %>%
    plot_time_series_regression(
        .date_var     = date,
        .formula      = revenue ~ as.numeric(date) + month(date, label = TRUE) + tv_spend +
          billboard_spend + print_spend + search_spend + facebook_spend + competitor_sales,
        .interactive  = interactive,
        .show_summary = FALSE
    )
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-6-1.png" width="768" />

# A Simple Linear Model

Now lets take a look at our simple model output that includes direct marketing features and examine their effect of revenue

We can see that:

- TV spend, print spend, competitor sales and Facebook spend are statistically significant
- Print spend has the largest positive effect at increasing revenue of each of the direct marketing inputs
- A large positive intercept suggests that this company already has a strong baseline of sales

``` r
# forcing intercept to 0 to just show the gap from 0 instead of a base
revenue_fit <- lm(revenue ~ as.numeric(date) + tv_spend + billboard_spend + print_spend + 
                    search_spend + facebook_spend + competitor_sales, data = df)
summary(revenue_fit)
```

    ## 
    ## Call:
    ## lm(formula = revenue ~ as.numeric(date) + tv_spend + billboard_spend + 
    ##     print_spend + search_spend + facebook_spend + competitor_sales, 
    ##     data = df)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -440321  -86030  -59261  -12997 1618906 
    ## 
    ## Coefficients:
    ##                    Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)       7.345e+05  1.070e+06   0.687   0.4932    
    ## as.numeric(date) -3.416e+01  5.754e+01  -0.594   0.5534    
    ## tv_spend          5.007e-01  9.069e-02   5.521 1.04e-07 ***
    ## billboard_spend   4.183e-02  1.216e-01   0.344   0.7313    
    ## print_spend       8.755e-01  3.950e-01   2.216   0.0278 *  
    ## search_spend      5.364e-01  6.683e-01   0.803   0.4231    
    ## facebook_spend    3.577e-01  2.095e-01   1.707   0.0893 .  
    ## competitor_sales  2.868e-01  1.154e-02  24.846  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 264700 on 200 degrees of freedom
    ## Multiple R-squared:  0.8681, Adjusted R-squared:  0.8634 
    ## F-statistic:   188 on 7 and 200 DF,  p-value: < 2.2e-16

## Viz: SLM coefficients

We can plot the coefficient and confidence intervals to make it easier to see the reliability of each estimate

- Print spend has the largest positive coefficient, but also the widest confidence interval
- Our model is very confident in the estimated effect of competitor sales

``` r
library(broom)
library(dotwhisker)

tidy(revenue_fit) %>% 
  filter(p.value < 0.1) %>% 
  mutate(
         term = fct_reorder(term, -estimate)) %>% 
  dwplot(
    vars_order = levels(.$term),
    dot_args = list(size = 3, color = "darkred"),
  whisker_args = list(color = "darkred", alpha = 0.75)
  ) +
  labs(x = "Coefficient by channel",
       y = NULL,
       title = "Simple Linear Model Coefficients")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-8-1.png" width="768" />

# Fit: Bootstrap Resampling

## How reliable are our coefficients?

We can fit a model using bootstrapped resamples of our data. This allows us to determine the stability of our coefficient estimates. Essentially this is fitting many small models to examine the variation of each channel’s coefficient.

- By default `reg_intervals` uses 1,001 bootstrap samples for t-intervals and 2,001 for percentile intervals.

``` r
library(rsample)

revenue_intervals <-
  reg_intervals(revenue ~ as.numeric(date) + tv_spend + billboard_spend + print_spend + 
                    search_spend + facebook_spend + competitor_sales,
                data = df, keep_reps = TRUE)

revenue_intervals
```

    ## # A tibble: 7 × 7
    ##   term                .lower .estimate .upper .alpha .method         .replicates
    ##   <chr>                <dbl>     <dbl>  <dbl>  <dbl> <chr>     <list<tibble[,2]>
    ## 1 as.numeric(date) -138.      -35.1    64.7     0.05 student-t       [1,001 × 2]
    ## 2 billboard_spend    -0.123     0.0403  0.167   0.05 student-t       [1,001 × 2]
    ## 3 competitor_sales    0.263     0.287   0.304   0.05 student-t       [1,001 × 2]
    ## 4 facebook_spend     -0.0439    0.368   0.707   0.05 student-t       [1,001 × 2]
    ## 5 print_spend         0.263     0.875   1.42    0.05 student-t       [1,001 × 2]
    ## 6 search_spend       -0.639     0.575   1.71    0.05 student-t       [1,001 × 2]
    ## 7 tv_spend            0.199     0.513   0.763   0.05 student-t       [1,001 × 2]

## Viz: Bootstrapped Resampled Coefficients

### Crossbar chart

- Here we can see the wide confidence interval surrounding `search_spend`.
- This makes sense since the coefficient for this channel in our first linear model was not statistically significant.

``` r
revenue_intervals %>%
  filter(!term == "as.numeric(date)") %>% 
    mutate(
        term = fct_reorder(term, .estimate)
    ) %>%
    ggplot(aes(.estimate, term)) +
    geom_crossbar(aes(xmin = .lower, xmax = .upper),
                  color = "darkred", alpha = 0.8) +
  labs(x = "Coefficient by channel", 
       y = NULL, 
       title = "Bootstrapped Resampled Model Coefficients", 
       subtitle = "95% Confidence Intervals")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-10-1.png" width="768" />

# Time-Series Forecasting

Lets build our forecasting model!

``` r
df_ts <- 
  df %>% 
  select(date, revenue)

df_ts %>% 
  ggplot(aes(date, revenue)) +
  geom_line() +
  labs(title = "Revenue",
       x = "")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-11-1.png" width="768" />

## Training & Testing Splits

First, I’ll split the data into training and testing sets

``` r
splits <- 
  df_ts %>% 
  time_series_split(assess = "12 months", cumulative = TRUE)

splits %>% 
  tk_time_series_cv_plan() %>% 
  plot_time_series_cv_plan(date, revenue, .interactive = interactive) +
  labs(title = "Timeseries - Training / Testing Split")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-12-1.png" width="768" />

## Modeling

Now we can create the recipe for our model.

After we `bake`, we can see what the output looks like.

``` r
library(tidymodels)
recipe_spec_timeseries <- 
  recipe(revenue ~., data = training(splits)) %>%
    step_timeseries_signature(date) 

bake(prep(recipe_spec_timeseries), new_data = training(splits))
```

    ## # A tibble: 156 × 29
    ##    date        revenue date_in…¹ date_…² date_…³ date_…⁴ date_…⁵ date_…⁶ date_…⁷
    ##    <date>        <dbl>     <dbl>   <int>   <int>   <int>   <int>   <int>   <int>
    ##  1 2018-11-23 2754372.    1.54e9    2018    2018       2       4      11      10
    ##  2 2018-11-30 2584277.    1.54e9    2018    2018       2       4      11      10
    ##  3 2018-12-07 2547387.    1.54e9    2018    2018       2       4      12      11
    ##  4 2018-12-14 2875220     1.54e9    2018    2018       2       4      12      11
    ##  5 2018-12-21 2215953.    1.55e9    2018    2018       2       4      12      11
    ##  6 2018-12-28 2569922.    1.55e9    2018    2018       2       4      12      11
    ##  7 2019-01-04 2171507.    1.55e9    2019    2019       1       1       1       0
    ##  8 2019-01-11 2464132.    1.55e9    2019    2019       1       1       1       0
    ##  9 2019-01-18 2012520     1.55e9    2019    2019       1       1       1       0
    ## 10 2019-01-25 1738912.    1.55e9    2019    2019       1       1       1       0
    ## # … with 146 more rows, 20 more variables: date_month.lbl <ord>,
    ## #   date_day <int>, date_hour <int>, date_minute <int>, date_second <int>,
    ## #   date_hour12 <int>, date_am.pm <int>, date_wday <int>, date_wday.xts <int>,
    ## #   date_wday.lbl <ord>, date_mday <int>, date_qday <int>, date_yday <int>,
    ## #   date_mweek <int>, date_week <int>, date_week.iso <int>, date_week2 <int>,
    ## #   date_week3 <int>, date_week4 <int>, date_mday7 <int>, and abbreviated
    ## #   variable names ¹​date_index.num, ²​date_year, ³​date_year.iso, ⁴​date_half, …

### Preprocessing Steps

Now we’ll add a few additional preprocessing steps to our recipe:

- Convert a date column into a fourier series
- Remove date
- Normalize the inputs
- one-hot encode categorical inputs (add dummies)

``` r
recipe_spec_final <- recipe_spec_timeseries %>%
    step_fourier(date, period = 365, K = 1) %>%
    step_rm(date) %>%
    step_rm(contains("iso"), contains("minute"), contains("hour"),
            contains("am.pm"), contains("xts")) %>%
    step_normalize(contains("index.num"), date_year) %>%
    step_dummy(contains("lbl"), one_hot = TRUE) 

recipe_spec_prophet <- recipe_spec_timeseries

juice(prep(recipe_spec_final))
```

    ## # A tibble: 156 × 39
    ##     revenue date_index…¹ date_…² date_…³ date_…⁴ date_…⁵ date_…⁶ date_…⁷ date_…⁸
    ##       <dbl>        <dbl>   <dbl>   <int>   <int>   <int>   <int>   <int>   <int>
    ##  1 2754372.        -1.71   -2.14       2       4      11      23       0       6
    ##  2 2584277.        -1.69   -2.14       2       4      11      30       0       6
    ##  3 2547387.        -1.67   -2.14       2       4      12       7       0       6
    ##  4 2875220         -1.65   -2.14       2       4      12      14       0       6
    ##  5 2215953.        -1.62   -2.14       2       4      12      21       0       6
    ##  6 2569922.        -1.60   -2.14       2       4      12      28       0       6
    ##  7 2171507.        -1.58   -1.01       1       1       1       4       0       6
    ##  8 2464132.        -1.56   -1.01       1       1       1      11       0       6
    ##  9 2012520         -1.54   -1.01       1       1       1      18       0       6
    ## 10 1738912.        -1.51   -1.01       1       1       1      25       0       6
    ## # … with 146 more rows, 30 more variables: date_mday <int>, date_qday <int>,
    ## #   date_yday <int>, date_mweek <int>, date_week <int>, date_week2 <int>,
    ## #   date_week3 <int>, date_week4 <int>, date_mday7 <int>, date_sin365_K1 <dbl>,
    ## #   date_cos365_K1 <dbl>, date_month.lbl_01 <dbl>, date_month.lbl_02 <dbl>,
    ## #   date_month.lbl_03 <dbl>, date_month.lbl_04 <dbl>, date_month.lbl_05 <dbl>,
    ## #   date_month.lbl_06 <dbl>, date_month.lbl_07 <dbl>, date_month.lbl_08 <dbl>,
    ## #   date_month.lbl_09 <dbl>, date_month.lbl_10 <dbl>, …

## Model Specs

1.  linear regression
2.  xgboost

``` r
model_spec_lm <- linear_reg(mode = "regression") %>%
    set_engine("lm")

model_spec_xgb <- boost_tree(mode = "regression") %>% 
  set_engine("xgboost")

model_spec_prophet <- prophet_reg(mode = "regression") %>% 
  set_engine("prophet")
```

## Workflow

We can add our recipe and model to a workflow

``` r
workflow_lm <- workflow() %>%
    add_recipe(recipe_spec_final) %>%
    add_model(model_spec_lm)

workflow_xgb <- workflow() %>% 
  add_recipe(recipe_spec_final) %>% 
  add_model(model_spec_xgb)

workflow_prophet <- workflow() %>% 
  add_recipe(recipe_spec_prophet) %>% 
  add_model(model_spec_prophet)
```

## Fit our Model

``` r
workflow_fit_lm <- 
  workflow_lm %>% 
  fit(data = training(splits))

workflow_fit_xgb <- 
  workflow_xgb %>% 
  fit(data = training(splits))

workflow_fit_prophet <-
  workflow_prophet %>% 
  fit(data = training(splits))
```

``` r
model_table <- modeltime_table(
  workflow_fit_lm,
  workflow_fit_xgb,
  workflow_fit_prophet
) 

calibration_table <- model_table %>%
  modeltime_calibrate(testing(splits))
```

## Measure Accuracy

``` r
calibration_table %>% 
    modeltime_accuracy(acc_by_id = FALSE) %>% 
    table_modeltime_accuracy(.interactive = FALSE)
```

<div id="ircgvwpzfb" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>html {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
}

#ircgvwpzfb .gt_table {
  display: table;
  border-collapse: collapse;
  margin-left: auto;
  margin-right: auto;
  color: #333333;
  font-size: 16px;
  font-weight: normal;
  font-style: normal;
  background-color: #FFFFFF;
  width: auto;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #A8A8A8;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #A8A8A8;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
}

#ircgvwpzfb .gt_heading {
  background-color: #FFFFFF;
  text-align: center;
  border-bottom-color: #FFFFFF;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#ircgvwpzfb .gt_caption {
  padding-top: 4px;
  padding-bottom: 4px;
}

#ircgvwpzfb .gt_title {
  color: #333333;
  font-size: 125%;
  font-weight: initial;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-color: #FFFFFF;
  border-bottom-width: 0;
}

#ircgvwpzfb .gt_subtitle {
  color: #333333;
  font-size: 85%;
  font-weight: initial;
  padding-top: 0;
  padding-bottom: 6px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-color: #FFFFFF;
  border-top-width: 0;
}

#ircgvwpzfb .gt_bottom_border {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#ircgvwpzfb .gt_col_headings {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#ircgvwpzfb .gt_col_heading {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  padding-left: 5px;
  padding-right: 5px;
  overflow-x: hidden;
}

#ircgvwpzfb .gt_column_spanner_outer {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  padding-top: 0;
  padding-bottom: 0;
  padding-left: 4px;
  padding-right: 4px;
}

#ircgvwpzfb .gt_column_spanner_outer:first-child {
  padding-left: 0;
}

#ircgvwpzfb .gt_column_spanner_outer:last-child {
  padding-right: 0;
}

#ircgvwpzfb .gt_column_spanner {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 5px;
  overflow-x: hidden;
  display: inline-block;
  width: 100%;
}

#ircgvwpzfb .gt_group_heading {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  text-align: left;
}

#ircgvwpzfb .gt_empty_group_heading {
  padding: 0.5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: middle;
}

#ircgvwpzfb .gt_from_md > :first-child {
  margin-top: 0;
}

#ircgvwpzfb .gt_from_md > :last-child {
  margin-bottom: 0;
}

#ircgvwpzfb .gt_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  margin: 10px;
  border-top-style: solid;
  border-top-width: 1px;
  border-top-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  overflow-x: hidden;
}

#ircgvwpzfb .gt_stub {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
}

#ircgvwpzfb .gt_stub_row_group {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
  vertical-align: top;
}

#ircgvwpzfb .gt_row_group_first td {
  border-top-width: 2px;
}

#ircgvwpzfb .gt_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#ircgvwpzfb .gt_first_summary_row {
  border-top-style: solid;
  border-top-color: #D3D3D3;
}

#ircgvwpzfb .gt_first_summary_row.thick {
  border-top-width: 2px;
}

#ircgvwpzfb .gt_last_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#ircgvwpzfb .gt_grand_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#ircgvwpzfb .gt_first_grand_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: double;
  border-top-width: 6px;
  border-top-color: #D3D3D3;
}

#ircgvwpzfb .gt_striped {
  background-color: rgba(128, 128, 128, 0.05);
}

#ircgvwpzfb .gt_table_body {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#ircgvwpzfb .gt_footnotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#ircgvwpzfb .gt_footnote {
  margin: 0px;
  font-size: 90%;
  padding-left: 4px;
  padding-right: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#ircgvwpzfb .gt_sourcenotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#ircgvwpzfb .gt_sourcenote {
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#ircgvwpzfb .gt_left {
  text-align: left;
}

#ircgvwpzfb .gt_center {
  text-align: center;
}

#ircgvwpzfb .gt_right {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

#ircgvwpzfb .gt_font_normal {
  font-weight: normal;
}

#ircgvwpzfb .gt_font_bold {
  font-weight: bold;
}

#ircgvwpzfb .gt_font_italic {
  font-style: italic;
}

#ircgvwpzfb .gt_super {
  font-size: 65%;
}

#ircgvwpzfb .gt_footnote_marks {
  font-style: italic;
  font-weight: normal;
  font-size: 75%;
  vertical-align: 0.4em;
}

#ircgvwpzfb .gt_asterisk {
  font-size: 100%;
  vertical-align: 0;
}

#ircgvwpzfb .gt_indent_1 {
  text-indent: 5px;
}

#ircgvwpzfb .gt_indent_2 {
  text-indent: 10px;
}

#ircgvwpzfb .gt_indent_3 {
  text-indent: 15px;
}

#ircgvwpzfb .gt_indent_4 {
  text-indent: 20px;
}

#ircgvwpzfb .gt_indent_5 {
  text-indent: 25px;
}
</style>
<table class="gt_table">
  <thead class="gt_header">
    <tr>
      <td colspan="9" class="gt_heading gt_title gt_font_normal gt_bottom_border" style>Accuracy Table</td>
    </tr>
    
  </thead>
  <thead class="gt_col_headings">
    <tr>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" scope="col" id=".model_id">.model_id</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id=".model_desc">.model_desc</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id=".type">.type</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" scope="col" id="mae">mae</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" scope="col" id="mape">mape</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" scope="col" id="mase">mase</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" scope="col" id="smape">smape</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" scope="col" id="rmse">rmse</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" scope="col" id="rsq">rsq</th>
    </tr>
  </thead>
  <tbody class="gt_table_body">
    <tr><td headers=".model_id" class="gt_row gt_right">1</td>
<td headers=".model_desc" class="gt_row gt_left">LM</td>
<td headers=".type" class="gt_row gt_left">Test</td>
<td headers="mae" class="gt_row gt_right">242994.3</td>
<td headers="mape" class="gt_row gt_right">16.19</td>
<td headers="mase" class="gt_row gt_right">0.90</td>
<td headers="smape" class="gt_row gt_right">15.14</td>
<td headers="rmse" class="gt_row gt_right">313485.5</td>
<td headers="rsq" class="gt_row gt_right">0.80</td></tr>
    <tr><td headers=".model_id" class="gt_row gt_right">2</td>
<td headers=".model_desc" class="gt_row gt_left">XGBOOST</td>
<td headers=".type" class="gt_row gt_left">Test</td>
<td headers="mae" class="gt_row gt_right">260391.4</td>
<td headers="mape" class="gt_row gt_right">18.67</td>
<td headers="mase" class="gt_row gt_right">0.97</td>
<td headers="smape" class="gt_row gt_right">16.33</td>
<td headers="rmse" class="gt_row gt_right">352541.8</td>
<td headers="rsq" class="gt_row gt_right">0.76</td></tr>
    <tr><td headers=".model_id" class="gt_row gt_right">3</td>
<td headers=".model_desc" class="gt_row gt_left">PROPHET W/ REGRESSORS</td>
<td headers=".type" class="gt_row gt_left">Test</td>
<td headers="mae" class="gt_row gt_right">306623.9</td>
<td headers="mape" class="gt_row gt_right">21.62</td>
<td headers="mase" class="gt_row gt_right">1.14</td>
<td headers="smape" class="gt_row gt_right">18.85</td>
<td headers="rmse" class="gt_row gt_right">384166.2</td>
<td headers="rsq" class="gt_row gt_right">0.79</td></tr>
  </tbody>
  
  
</table>
</div>

## Forcasting Results

Lets visualize each model’s prediction accuracy

``` r
calibration_table %>%
  modeltime_forecast(actual_data = df_ts) %>%
  plot_modeltime_forecast(.interactive = FALSE) +
  labs(
    y = "Revenue",
    title = "Revenue",
    subtitle = "Actual vs Predicted"
  ) +
  theme(legend.position = "none")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-20-1.png" width="768" />

**Sources:** *Matt Dancho, Julia Silge*
