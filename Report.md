Markowitz Mean-Variance Optimization (MVO) Portfolio
================
Rudraksha Dev Sharma

# Fetching Necessary Libraries

In this step, we simply fetch the necessary libraries

``` r
library(tidyquant)
```

    ## Registered S3 method overwritten by 'quantmod':
    ##   method            from
    ##   as.zoo.data.frame zoo

    ## ── Attaching core tidyquant packages ─────────────────────── tidyquant 1.0.11 ──
    ## ✔ PerformanceAnalytics 2.0.8      ✔ TTR                  0.24.4
    ## ✔ quantmod             0.4.28     ✔ xts                  0.14.1
    ## ── Conflicts ────────────────────────────────────────── tidyquant_conflicts() ──
    ## ✖ zoo::as.Date()                 masks base::as.Date()
    ## ✖ zoo::as.Date.numeric()         masks base::as.Date.numeric()
    ## ✖ PerformanceAnalytics::legend() masks graphics::legend()
    ## ✖ quantmod::summary()            masks base::summary()
    ## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

``` r
library(ggplot2)
library(corrplot)
```

    ## corrplot 0.95 loaded

``` r
library(dplyr)
```

    ## 
    ## ######################### Warning from 'xts' package ##########################
    ## #                                                                             #
    ## # The dplyr lag() function breaks how base R's lag() function is supposed to  #
    ## # work, which breaks lag(my_xts). Calls to lag(my_xts) that you type or       #
    ## # source() into this session won't work correctly.                            #
    ## #                                                                             #
    ## # Use stats::lag() to make sure you're not using dplyr::lag(), or you can add #
    ## # conflictRules('dplyr', exclude = 'lag') to your .Rprofile to stop           #
    ## # dplyr from breaking base R's lag() function.                                #
    ## #                                                                             #
    ## # Code in packages is not affected. It's protected by R's namespace mechanism #
    ## # Set `options(xts.warn_dplyr_breaks_lag = FALSE)` to suppress this warning.  #
    ## #                                                                             #
    ## ###############################################################################
    ## 
    ## Attaching package: 'dplyr'
    ## 
    ## The following objects are masked from 'package:xts':
    ## 
    ##     first, last
    ## 
    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag
    ## 
    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(quadprog)
```

# Symbols

The symbols that we have decided to use here are SPY (for the S&P500),
FXI (for iShares China Large-Cap ETF), GLD (for Gold), and EPI (for
Nifty50 ETF) and we’ll be training it based on 17 years of data

``` r
ticker_FXI <- "FXI" #iShares China Large-Cap
ticker_SnP <- "SPY"
ticker_Nifty50 <- "EPI"
ticker_Gold <- "GLD"

start_date = as.Date("2008-09-17")
end_date = as.Date("2025-01-01")

TR_FXI <- tq_get(ticker_FXI, from=start_date, to=end_date)
TR_SnP <- tq_get(ticker_SnP, from=start_date, to=end_date)
TR_Nifty <- tq_get(ticker_Nifty50, from = start_date, to=end_date)
TR_Gold <- tq_get(ticker_Gold, from = start_date, to = end_date)
```

# Sanity check

Purpose of this cell is to get all the returns of the 4 assets so that
we can later put them all into the dataframe and also just to see
whether the returns are normally distributed or not (of course they are,
but I just like looking at the Law of Large Numbers come into play).

``` r
rets_SnP <- TR_SnP %>%
  arrange(date) %>%
  mutate(ret = adjusted/lag(adjusted) - 1) %>%
  filter(!is.na(ret))

rets_FXI <- TR_FXI %>%
  arrange(date) %>%
  mutate(ret = adjusted/lag(adjusted) - 1) %>%
  filter(!is.na(ret))

rets_Gold <- TR_Gold %>%
  arrange(date) %>%
  mutate(ret = adjusted/lag(adjusted) - 1) %>%
  filter(!is.na(ret))

rets_Nifty <- TR_Nifty %>%
  arrange(date) %>%
  mutate(ret = adjusted/lag(adjusted) - 1) %>%
  filter(!is.na(ret))

ggplot(rets_SnP, aes(x=ret)) +
  geom_histogram(bins = 10, binwidth = 0.001) +
  labs(
    title = "Returns of SnP500",
    x = "P[t]/P[t-1] - 1",
    y = "Frequency"
  ) +
  theme_minimal()
```

![](Report_files/figure-gfm/Verifying%20returns%20being%20normal-1.png)<!-- -->

``` r
ggplot(rets_Nifty, aes(x=ret)) +
  geom_histogram(bins = 10, binwidth = 0.001) +
  labs(
    title = "Returns of Nifty50",
    x = "P[t]/P[t-1] - 1",
    y = "Frequency"
  ) +
  theme_minimal()
```

![](Report_files/figure-gfm/Verifying%20returns%20being%20normal-2.png)<!-- -->

``` r
ggplot(rets_Gold, aes(x=ret)) +
  geom_histogram(bins = 10, binwidth = 0.001) +
  labs(
    title = "Returns of Gold",
    x = "P[t]/P[t-1] - 1",
    y = "Frequency"
  ) +
  theme_minimal()
```

![](Report_files/figure-gfm/Verifying%20returns%20being%20normal-3.png)<!-- -->

``` r
ggplot(rets_FXI, aes(x=ret)) +
  geom_histogram(bins = 10, binwidth = 0.001) +
  labs(
    title = "Returns of FXI",
    x = "(P[t]/P[t-1]) - 1",
    y = "Frequency"
  ) +
  theme_minimal()
```

![](Report_files/figure-gfm/Verifying%20returns%20being%20normal-4.png)<!-- -->
\# Function for Markowitz Model This is the main function that runs the
Markowitz model for portfolio optimization. We take a small epsilon
nudge to avoid the corner portfolios and use quadratic programming to
minimize t(w) \* Cov \* w with the constraint that each individual
weight of an asset lies between 0 and 1.

``` r
markowitz <- function(Cov, Mu, Npoints)
{
  epsilon <- 0.0001
  Nu <- length(Mu)

  mu_min <- min(Mu) + epsilon*(mean(Mu) - min(Mu))
  mu_max <- max(Mu) - epsilon*(max(Mu) - mean(Mu))
  
  iota <- matrix(1, Nu, 1)
  zero <- matrix(0, Nu, 1)
  Amat <- cbind(iota, Mu, diag(1, Nu), diag(-1, Nu))
  meq <- 2
  
  mu_dist <- seq(from = mu_min, to = mu_max, length = Npoints)
  
  sigma_vec <- matrix(0, Npoints, 1)
  weights_mat <- matrix(0, Nu, Npoints)
  
  for(k in 1:Npoints)
  {
    bvec <- c(1, mu_dist[k], zero, -iota)
    opt <- solve.QP(Cov, zero, Amat, bvec, meq = meq)
    sigma_vec[k] <- sqrt(2* opt$value)
    weights_mat[,k] <- opt$solution
  }
  
  mu_p <- t(weights_mat) %*% Mu
  sigma_p <- sqrt(diag(t(weights_mat) %*% Cov %*% weights_mat))
  
  markowitz <- data.frame(mu_p, sigma_p, weights = t(weights_mat))
}
```

# Daily returns datframe

The chunk performs inner join on all the returns vectors of the assets
and aligns them by date. This avoids any error that may arise due to
missing dates if we were to simply use the cbind() function.

``` r
rets_all <- rets_FXI %>%
  select(date, FXI = ret) %>%
  inner_join(rets_SnP   %>% select(date, SnP   = ret), by = "date") %>%
  inner_join(rets_Nifty %>% select(date, Nifty = ret), by = "date") %>%
  inner_join(rets_Gold  %>% select(date, Gold  = ret), by = "date")

R_daily <- rets_all %>% select(-date)  # numeric matrix-like data frame (T x 4)
print(head(R_daily))
```

    ## # A tibble: 6 × 4
    ##       FXI      SnP   Nifty     Gold
    ##     <dbl>    <dbl>   <dbl>    <dbl>
    ## 1  0.136   0.0297   0.0706 -0.0311 
    ## 2  0.113   0.0397   0.0689  0.0384 
    ## 3 -0.0625 -0.0226  -0.0446  0.0372 
    ## 4 -0.0238 -0.0228  -0.0196 -0.00964
    ## 5  0.0126  0.00321  0      -0.0182 
    ## 6  0.0271  0.0156   0.0200 -0.00300

# Daily returns and covariance

Here, we calculate the mean daily returns of each asset and also the
covariance matrix of the daily returns.

``` r
Mu_daily  <- colMeans(R_daily)
Cov_daily <- cov(R_daily)

print(Mu_daily)
```

    ##          FXI          SnP        Nifty         Gold 
    ## 0.0003004844 0.0005487703 0.0004574960 0.0003096713

``` r
print(Cov_daily)
```

    ##                FXI          SnP        Nifty         Gold
    ## FXI   4.255364e-04 1.767336e-04 2.357054e-04 2.451972e-05
    ## SnP   1.767336e-04 1.569148e-04 1.527091e-04 9.316582e-06
    ## Nifty 2.357054e-04 1.527091e-04 3.071904e-04 2.659195e-05
    ## Gold  2.451972e-05 9.316582e-06 2.659195e-05 1.108717e-04

``` r
print(cor(R_daily))
```

    ##             FXI        SnP     Nifty       Gold
    ## FXI   1.0000000 0.68394116 0.6519246 0.11288513
    ## SnP   0.6839412 1.00000000 0.6955511 0.07063407
    ## Nifty 0.6519246 0.69555112 1.0000000 0.14409073
    ## Gold  0.1128851 0.07063407 0.1440907 1.00000000

``` r
corrplot(cor(R_daily), type = "lower")
```

![](Report_files/figure-gfm/Daily%20vectors-1.png)<!-- --> \#
Annualizing Mu and Cov Earlier we had gotten the daily Mu and Covariance
matrix, now we annualize them by multiplying each by 252 (number of days
in the year). If we wished to annualize the volatility alone, we would
have multiplied by sqrt(252), but for covariance and variance we would
have to multiply by 252 (because it is essentially multiplying by
(sqrt(252))^2).

``` r
Mu      <- as.numeric(Mu_daily) * 252
Cov_ann <- Cov_daily * 252

print(Mu)
```

    ## [1] 0.07572207 0.13829011 0.11528900 0.07803716

``` r
print(Cov_ann)
```

    ##               FXI         SnP       Nifty        Gold
    ## FXI   0.107235162 0.044536857 0.059397762 0.006178969
    ## SnP   0.044536857 0.039542530 0.038482687 0.002347779
    ## Nifty 0.059397762 0.038482687 0.077411987 0.006701172
    ## Gold  0.006178969 0.002347779 0.006701172 0.027939672

``` r
asset_names <- colnames(R_daily)
names(Mu) <- asset_names
```

# Running Markowitz Function

This chunk runs the Markowitz function Npoints number of times to create
an efficient frontier. “frontier” stores the matrix of weights so that
it can later be used to plot out the efficient frontier.

``` r
Npoints <- 100
frontier <- markowitz(Cov_ann, Mu, Npoints = Npoints)

w_cols <- 3:ncol(frontier)
colnames(frontier)[w_cols] <- paste0("w_", asset_names)
```

# Monte Carlo Simulation

Now, we do Monte Carlo Simulation of various weights. We do the
simulation Nsims number of times, each time taking a new random
arrangement of weights. We also set a seed for the results so that they
can be reproducible.

``` r
set.seed(42)
Nsims <- 20000
n <- length(Mu)

W <- matrix(rgamma(Nsims * n, shape = 1, rate = 1), nrow = Nsims, ncol = n)
W <- W / rowSums(W)
print(head(W))
```

    ##            [,1]       [,2]        [,3]      [,4]
    ## [1,] 0.50400277 0.07731022 0.217087792 0.2015992
    ## [2,] 0.02683410 0.63816063 0.081083605 0.2539217
    ## [3,] 0.15066345 0.34400277 0.004031006 0.5013028
    ## [4,] 0.02043194 0.39041925 0.277408527 0.3117403
    ## [5,] 0.30482697 0.05202177 0.459686874 0.1834644
    ## [6,] 0.02113369 0.02959300 0.422841579 0.5264317

``` r
mu_sim <- as.numeric(W %*% Mu)
sig_sim <- sqrt(rowSums((W %*% Cov_ann) * W))   # efficient diag(W %*% Cov %*% t(W))
```

# Sharpe Ratio Calculation

Using the portfolios we stored in the weight matrix W and calculating
the sharpe ratio of each portfolio by using mu_sim and sigma_sim that we
obtained in the previous chunk. We can then easily find the portfolio
with the highest sharpe.

``` r
# (Optional) Sharpe ratio ranking (We can set rf if we want)
rf <- 0
sharpe_sim <- (mu_sim - rf) / sig_sim

sim_df <- data.frame(
  mu_p = mu_sim,
  sigma_p = sig_sim,
  sharpe = sharpe_sim,
  W
)
colnames(sim_df)[4:(3+n)] <- paste0("w_", asset_names)

best_idx <- which.max(sim_df$sharpe)
best_mc <- sim_df[best_idx, ]

print("Best (approx) Sharpe portfolio from Monte Carlo:")
```

    ## [1] "Best (approx) Sharpe portfolio from Monte Carlo:"

``` r
print(best_mc)
```

    ##           mu_p   sigma_p   sharpe      w_FXI     w_SnP     w_Nifty    w_Gold
    ## 2420 0.1142353 0.1413842 0.807978 0.00147089 0.5962747 0.007362629 0.3948918

# Plotting of graphs

The purpose of this chunk is purely for visualisation purposes. The red
line shows the efficient frontier we calculated using our Markowitz
function, the black dots show the various portfolios we obtained by
doing Monte Carlo simulation, and the big green dot shows the simulated
portfolio with the highest sharpe.

``` r
frontier2 <- frontier %>%
  dplyr::mutate(mu_p = as.numeric(mu_p),
                sigma_p = as.numeric(sigma_p)) %>%
  dplyr::arrange(mu_p)

i_gmv <- which.min(frontier2$sigma_p)
frontier_eff <- frontier2[i_gmv:nrow(frontier2), ] %>%
  dplyr::arrange(sigma_p)

ggplot(sim_df, aes(x = sigma_p, y = mu_p)) +
  geom_point(alpha = 0.2) +
  geom_point(data = best_mc, aes(x = sigma_p, y = mu_p), size = 3, color = "green") +
  geom_path(data = frontier2, aes(x = sigma_p, y = mu_p), linewidth = 1, color = "red") +
  labs(
    title = "Monte Carlo Portfolios + Efficient Frontier",
    x = "Annualized Volatility (sigma)",
    y = "Annualized Expected Return (mu)"
  ) +
  theme_minimal()
```

![](Report_files/figure-gfm/Plotting%20graphs%20of%20Monte%20Carlo%20Simulation-1.png)<!-- -->

# Comparing the Results

Now, we would like to compare how the two models of ours (Monte Carlo
and Markowitz) perform with respect to eachother.

``` r
sharpe_markowitz <- (frontier2$mu_p - rf)/frontier2$sigma_p

best_markowitz_sharpe <- max(sharpe_markowitz)

print(paste("Best sharpe by Markowitz model is: ", best_markowitz_sharpe))
```

    ## [1] "Best sharpe by Markowitz model is:  0.811797407145061"

``` r
print(paste("Best sharpe by Monte Carlo model is: ", best_mc$sharpe))
```

    ## [1] "Best sharpe by Monte Carlo model is:  0.807978014218997"

``` r
deviation_pct <- abs(best_markowitz_sharpe - best_mc$sharpe)/best_markowitz_sharpe * 100

print(paste("Our Monte Carlo simulation deviates from the Monte Carlo model by ", deviation_pct ,"%"))
```

    ## [1] "Our Monte Carlo simulation deviates from the Monte Carlo model by  0.470485972540394 %"
