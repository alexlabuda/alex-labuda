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



# Packages


```r
library(modeltime)
library(timetk)
library(lubridate)

interactive <- FALSE
```

# The Data

We will analyze weekly marketing spend and revenue figures a company.


```r
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

```r
df %>% 
  ggplot(aes(date, revenue)) +
  geom_line() +
  geom_smooth(se = FALSE, color = "darkred") +
  labs(title = "Revenue",
       x = "")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-3-1.png" width="768" />

# Fit: Simple Regression Model

First we'll start by fitting a simple regression model to the data


```r
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


```r
df %>%
    plot_time_series_regression(
        .date_var     = date,
        .formula      = revenue ~ as.numeric(date) + month(date, label = TRUE),
        .interactive  = interactive,
        .show_summary = FALSE
    )
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-5-1.png" width="768" />

We can also introduce our spend variables as input to our model to study each channels effect on revenue


```r
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

Lets take a look at our simple model output that includes direct marketing features
- TV spend, print spend, competitor sales and Facebook spend are statistically significant
- Print spend has the largest positive effect at increasing revenue


```r
# forcing intercept to 0 to just show the gap from 0 instead of a base
revenue_fit <- lm(revenue ~ as.numeric(date) + tv_spend + billboard_spend + print_spend + 
                    search_spend + facebook_spend + competitor_sales, data = df)
summary(revenue_fit)
```

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
## Multiple R-squared:  0.8681,	Adjusted R-squared:  0.8634 
## F-statistic:   188 on 7 and 200 DF,  p-value: < 2.2e-16
```

## Viz: SLM coefficients

```r
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
  labs(x = "Coefficient by channel", y = NULL, title = "Simple Linear Model Coefficients", subtitle = "95% Confidence Intervals")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-8-1.png" width="768" />

# Fit: Bootstrap Resampling 

## How reliable are our coefficients?

We can fit many bootstrapped resampled models to determine the stability of our coefficient estimates

- By default `reg_intervals` uses 1,001 bootstrap samples for t-intervals and 2,001 for percentile intervals.


```r
library(rsample)

revenue_intervals <-
  reg_intervals(revenue ~ as.numeric(date) + tv_spend + billboard_spend + print_spend + 
                    search_spend + facebook_spend + competitor_sales,
                data = df, keep_reps = TRUE)

revenue_intervals
```

```
## # A tibble: 7 × 7
##   term                .lower .estimate .upper .alpha .method         .replicates
##   <chr>                <dbl>     <dbl>  <dbl>  <dbl> <chr>     <list<tibble[,2]>
## 1 as.numeric(date) -136.      -31.9    63.7     0.05 student-t       [1,001 × 2]
## 2 billboard_spend    -0.118     0.0425  0.168   0.05 student-t       [1,001 × 2]
## 3 competitor_sales    0.263     0.287   0.305   0.05 student-t       [1,001 × 2]
## 4 facebook_spend     -0.0147    0.356   0.697   0.05 student-t       [1,001 × 2]
## 5 print_spend         0.273     0.864   1.40    0.05 student-t       [1,001 × 2]
## 6 search_spend       -0.665     0.539   1.75    0.05 student-t       [1,001 × 2]
## 7 tv_spend            0.198     0.513   0.759   0.05 student-t       [1,001 × 2]
```

## Viz: Bootstrapped Resampled Coefficients

### Crossbar chart


```r
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

Lets build our forecasting model

```r
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


```r
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


```r
library(tidymodels)
recipe_spec_timeseries <- 
  recipe(revenue ~., data = training(splits)) %>%
    step_timeseries_signature(date) 

bake(prep(recipe_spec_timeseries), new_data = training(splits))
```

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
```

### Preprocessing Steps

- Convert a date column into a Fourier series
- Remove date
- Normalize
- one-hot encoding (add dummies)

```r
recipe_spec_final <- recipe_spec_timeseries %>%
    step_fourier(date, period = 365, K = 1) %>%
    step_rm(date) %>%
    step_rm(contains("iso"), contains("minute"), contains("hour"),
            contains("am.pm"), contains("xts")) %>%
    step_normalize(contains("index.num"), date_year) %>%
    step_dummy(contains("lbl"), one_hot = TRUE) 

juice(prep(recipe_spec_final))
```

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
```

## Model Specs

- Linear model

```r
model_spec_lm <- linear_reg(mode = "regression") %>%
    set_engine("lm")
```

## Workflow

```r
workflow_lm <- workflow() %>%
    add_recipe(recipe_spec_final) %>%
    add_model(model_spec_lm)

workflow_lm
```

```
## ══ Workflow ════════════════════════════════════════════════════════════════════
## Preprocessor: Recipe
## Model: linear_reg()
## 
## ── Preprocessor ────────────────────────────────────────────────────────────────
## 6 Recipe Steps
## 
## • step_timeseries_signature()
## • step_fourier()
## • step_rm()
## • step_rm()
## • step_normalize()
## • step_dummy()
## 
## ── Model ───────────────────────────────────────────────────────────────────────
## Linear Regression Model Specification (regression)
## 
## Computational engine: lm
```

## Fit our Model

```r
workflow_fit_lm <- workflow_lm %>% fit(data = training(splits))
```



```r
model_table <- modeltime_table(
  workflow_fit_lm
) 

calibration_table <- model_table %>%
  modeltime_calibrate(testing(splits))
```

## Forcasting Results


```r
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

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-19-1.png" width="768" />








