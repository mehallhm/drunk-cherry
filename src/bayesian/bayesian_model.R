library(MASS)
library(ggplot2)
library(gridExtra)
# for char replacement
library(stringr)

# set verbosity... I know this is bad but ohh well (I really don't feel like adding it as a parameter to every function even if it'd be quick)
verbose <- TRUE
set.seed(1337)

DATA_PATH <- "../../all_trails.csv"
MODEL_OUT_DIR <- "../models/bayesian/"
PLOT_DIR <- "../models/bayesian/plots/"

FEATURE_COLS <- c("elevation_gain", "elevation_loss", "average_grade", "max_grade")
CLASS_LEVELS <- c("Easy", "Intermediate", "Intermediate/Difficult", "Difficult")
FEAT_NAMES <- c("Intercept", FEATURE_COLS)

N_ITER <- 400000
BURNIN_RATIO <- 0.18
BURNIN <- floor(N_ITER * BURNIN_RATIO)

# THIN <- 300
THIN <- 200

PROPOSAL_SD <- 0.05
TRAIN_RATIO <- 0.80
SIGMA_PRIOR <- 3.0
class_count <- 4

DIFFICULTY_MAP <- list(
  "Easy" = "Easy",
  "Easy/Intermediate" = "Easy",
  "Intermediate" = "Intermediate",
  "Intermediate/Difficult" = "Intermediate/Difficult",
  "Difficult" = "Difficult",
  "Very_Difficult" = "Difficult"
)

CLASS_COLORS <- c("red", "blue", "green")

# ggplot2 theme similar to the lecture R files
theme_bayes <- function() {
  theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 13, margin = margin(b = 6)),
      plot.subtitle = element_text(size = 10, color = "grey", margin = margin(b = 8)),
      axis.title = element_text(size = 11),
      axis.text = element_text(size = 9),
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold", size = 10),
      legend.position = "bottom"
    )
}

load_and_preprocess <- function() {
  raw <- read.csv(DATA_PATH, stringsAsFactors = FALSE)

  raw$difficulty_mapped <- sapply(raw$difficulty, function(difficulty_label) {
    mapped <- DIFFICULTY_MAP[[difficulty_label]]
    if (is.null(mapped)) NA else mapped
  })
  raw <- raw[!is.na(raw$difficulty_mapped), ]

  # One row per trail since features are trail-constant
  trail_df <- do.call(rbind, lapply(split(raw, raw$trail_id), `[`, 1, ))
  rownames(trail_df) <- NULL
  trail_df <- trail_df[complete.cases(trail_df[, c(FEATURE_COLS, "difficulty_mapped")]), ]
  trail_df$label <- match(trail_df$difficulty_mapped, CLASS_LEVELS)

  if (verbose) cat(sprintf("Trails after grouping & cleaning: %d\n", nrow(trail_df)))

  trail_df
}

balance_classes <- function(trail_df) {
  class_indices <- lapply(seq_len(class_count), function(class_num) which(trail_df$label == class_num))
  min_count <- min(sapply(class_indices, length))
  if (verbose) {
    cat(sprintf("Minority class size: %d. All classes undersampled to this size\n", min_count))
  }

  balanced_idx <- sample(unlist(lapply(class_indices, function(class_idx)
    sample(class_idx, min_count, replace = FALSE))))
  df_bal <- trail_df[balanced_idx, ]

  df_bal
}

split_and_standardize <- function(df_bal) {
  # remove column names and divide by s.d.
  standardize <- function(X, mu, sigma) sweep(sweep(X, 2, mu, "-"), 2, sigma, "/")

  X_raw <- as.matrix(df_bal[, FEATURE_COLS])
  y <- df_bal$label

  # random sample training and make the rest of them test
  n_total <- nrow(X_raw)
  n_train <- floor(TRAIN_RATIO * n_total)
  train_idx <- sample(seq_len(n_total), n_train)
  test_idx <- setdiff(seq_len(n_total), train_idx)

  X_train_raw <- X_raw[train_idx, ]
  X_test_raw <- X_raw[test_idx, ]
  y_train <- y[train_idx]
  y_test <- y[test_idx]

  feat_mean <- colMeans(X_train_raw)
  feat_sd <- apply(X_train_raw, 2, sd)
  feat_sd[feat_sd == 0] <- 1

  # standardize using training and add the intercept column
  X_train <- cbind(1, standardize(X_train_raw, feat_mean, feat_sd))
  X_test <- cbind(1, standardize(X_test_raw, feat_mean, feat_sd))
  feature_count <- ncol(X_train)

  if (verbose) {
    cat(sprintf("Train: %d  |  Test: %d  |  Features + intercept: %d\n",
                n_train, length(test_idx), feature_count))
  }

  list(X_train = X_train, X_test = X_test,
       y_train = y_train, y_test = y_test,
       feat_mean = feat_mean, feat_sd = feat_sd,
       feature_count = feature_count)
}


# helpers for the multinomial log. reg. model
# using reference class of K (Difficult)
# W: P x (K-1); log P(y=k | x) = x'w_k - log-sum-exp;  log P(y=K | x) = 0 - lse
# Prior: w ~ N(0, sigma_prior^2 * I)

# converts logits to probabilites
softmax_row <- function(eta) {
  eta <- eta - max(eta)
  p <- exp(eta)
  p / sum(p)
}

log_likelihood <- function(W, X, y) {
  eta <- cbind(X %*% W, 0) # last col is reference class = 0
  row_max <- apply(eta, 1, max)
  log_sum <- log(rowSums(exp(eta - row_max))) + row_max
  sum(eta[cbind(seq_len(nrow(eta)), y)] - log_sum)
}

# equivalent to L2 regularization
log_prior <- function(W) -0.5 * sum(W^2) / SIGMA_PRIOR^2

# log(W|data) = log(data|W) + log(W)
log_posterior <- function(W, X, y) log_likelihood(W, X, y) + log_prior(W)

# metrpolis hastings sampler. We do everything in log space so some formulas are gonna be different from class.
run_mcmc <- function(X_train, y_train, feature_count, existing_weights_file) {
  if (verbose) {
    cat(sprintf("%d iterations | burn-in: %d | thin: %d\n",
                N_ITER, BURNIN, THIN))
  }

  W_dim <- feature_count * (class_count - 1)

  if (file.exists(existing_weights_file)) {
    prior_weights_df <- read.csv(existing_weights_file)
    W_current <- as.matrix(prior_weights_df[, -1])
    if (verbose) {
      cat(sprintf("Weights loaded from %s\n", existing_weights_file))
    }
  } else {
    # starts at zero weights if we don't have existing weights from previous runs
    W_current <- matrix(0, feature_count, class_count - 1)
    if (verbose) {
      cat("Starting from zero weights\n")
    }
  }

  current_log_probability <- log_posterior(W_current, X_train, y_train)

  n_keep <- floor((N_ITER - BURNIN) / THIN)
  # tracker for posterior samples
  W_samples <- array(NA, dim = c(n_keep, feature_count, class_count - 1))
  sample_idx <- 0
  n_accept <- 0

  if (verbose) cat(sprintf("Keeping %d samples.\n", n_keep))

  for (iter in seq_len(N_ITER)) {
    # proposal: Gaussian (using rnorm prebuilt sampler: https://piazza.com/class/mk0avxqgwuv2bj/post/215)
    epsilon <- matrix(rnorm(W_dim, 0, PROPOSAL_SD), feature_count, class_count - 1)
    W_prop <- W_current + epsilon

    # acceptance ratio. log(r) = log(W'|data) - log(W|data)
    log_probability <- log_posterior(W_prop, X_train, y_train)
    alpha <- log_probability - current_log_probability

    # accept / reject
    if (alpha > log(runif(1))) {
      W_current <- W_prop
      current_log_probability <- log_probability
      n_accept <- n_accept + 1
    }

    # skip the early samples and only keep the thinned out samples
    # it'll count samples IFF we're also on a factor of the thin rate
    if (iter > BURNIN && ((iter - BURNIN) %% THIN == 0)) {
      sample_idx <- sample_idx + 1
      W_samples[sample_idx, , ] <- W_current
    }
  }

  # we should see about %23 (iirc)
  acceptance_rate <- n_accept / N_ITER
  cat(sprintf("Final acceptance rate: %.2f%%\n", 100 * acceptance_rate))

  post_chain <- W_samples[seq_len(sample_idx), , , drop = FALSE]

  list(post_chain = post_chain,
       sample_idx = sample_idx,
       acceptance_rate = acceptance_rate)
}

# post chain will have axis 1: iteration, axis 2: feature, and axis 3: class
plot_mcmc <- function(post_chain, sample_idx) {
  feature_count <- dim(post_chain)[2]

  for (feature_idx in seq_len(feature_count)) {
    feat_label <- FEAT_NAMES[feature_idx]
    file_label <- str_replace(feat_label, "/", "_")

    # match the style of the convergence plots in sampling_mcmc_drg.R
    png(file.path(PLOT_DIR, sprintf("%s_chain.png", file_label)),
        width = 800, height = 300 * (class_count - 1))
    # make the plot have class_count-1 rows (3) to show the chains for easy, intermediate, and intermediate/difficult classes
    par(mfrow = c(class_count - 1, 1))

    for (class_idx in seq_len(class_count - 1)) {
      chain <- post_chain[, feature_count, class_idx]
      plot(chain, type = "l",
           col = CLASS_COLORS[class_idx],
           main = sprintf("Chain — %s | Class: %s", feat_label, CLASS_LEVELS[class_idx]),
           xlab = "Sample Index", ylab = "Weight Value")
    }
    dev.off()

    # follows the same pattern as sampling_mcmc_drg.R from class; full chain then thinned.
    png(file.path(PLOT_DIR, sprintf("acf%s.png", file_label)),
        width = 800, height = 300 * (class_count - 1) * 2)
    par(mfrow = c((class_count - 1) * 2, 1), mar = c(4, 4, 3, 1))

    for (class_idx in seq_len(class_count - 1)) {
      chain <- post_chain[, feature_count, class_idx]
      thinned_chain <- chain[seq(1, length(chain), by = THIN)]

      # full chain ACF
      acf(chain,
          main = sprintf("ACF — %s | %s (full chain)",
                            feat_label, CLASS_LEVELS[class_idx]))

      # thinned chain ACF
      acf(thinned_chain,
          main = sprintf("ACF — %s | %s (thinned every %dx)",
                            feat_label, CLASS_LEVELS[class_idx], THIN))
    }
    dev.off()
  }

  if (verbose) {
    cat(sprintf("Saved chain + ACF plots for all %d features\n", P))
  }
}


plot_posteriors <- function(post_chain, sample_idx) {
  feature_count <- dim(post_chain)[2]

  for (feature_idx in seq_len(feature_count)) {
    feat_label <- FEAT_NAMES[feature_count]
    file_label <- str_replace(feat_label, "/", "_")

    dens_data <- data.frame(
      value = as.vector(
        sapply(seq_len(class_count - 1), function(class_idx) post_chain[, feature_count, class_idx])
      ),
      class = factor(rep(CLASS_LEVELS[seq_len(class_count - 1)], each = sample_idx),
                     levels = CLASS_LEVELS[seq_len(class_count - 1)])
    )

    ci_df <- do.call(rbind, lapply(seq_len(class_count - 1), function(class_idx) {
      weight_samples <- post_chain[, feature_count, class_idx]
      data.frame(
        class = factor(CLASS_LEVELS[class_idx], levels = CLASS_LEVELS[seq_len(class_count - 1)]),
        mean = mean(weight_samples),
        lo = quantile(weight_samples, 0.025),
        hi = quantile(weight_samples, 0.975)
      )
    }))

    p_dens <- ggplot(dens_data, aes(x = value, fill = class, color = class)) +
      geom_density(alpha = 0.30, linewidth = 0.9) +
      geom_vline(data = ci_df, aes(xintercept = mean, color = class),
                 linetype = "solid", linewidth = 1.0) +
      geom_vline(data = ci_df, aes(xintercept = lo, color = class),
                 linetype = "dashed", linewidth = 0.6) +
      geom_vline(data = ci_df, aes(xintercept = hi, color = class),
                 linetype = "dashed", linewidth = 0.6) +
      scale_fill_manual(values = CLASS_COLORS) +
      scale_color_manual(values = CLASS_COLORS) +
      facet_wrap(~ class, ncol = 1, scales = "free_y") +
      labs(title = sprintf("Posterior Distribution — %s", feat_label),
           subtitle = "Solid = posterior mean | Dashed = 95% credible interval",
           x = "Weight Value", y = "Density") +
      theme_bayes() + theme(legend.position = "none")

    ggsave(file.path(PLOT_DIR, sprintf("posterior_density_%s.png", file_label)),
           p_dens, width = 7, height = 6, dpi = 150)
  }

  coef_rows <- lapply(seq_len(P), function(feature_count) {
    lapply(seq_len(class_count - 1), function(class_idx) {
      weight_samples <- post_chain[, feaure_count, class_idx]
      data.frame(feature = FEAT_NAMES[feature_count],
                 class = CLASS_LEVELS[class_idx],
                 mean = mean(weight_samples),
                 lo = quantile(weight_samples, 0.025),
                 hi = quantile(weight_samples, 0.975))
    })
  })
  coef_df <- do.call(rbind, do.call(c, coef_rows))
  coef_df$feature <- factor(coef_df$feature, levels = rev(FEAT_NAMES))
  coef_df$class <- factor(coef_df$class, levels = CLASS_LEVELS[seq_len(class_count - 1)])

  if (verbose) cat("Saved posterior density plots.\n")
}

# predict and evaluate the model
predict_probs <- function(W, X) {
  eta <- cbind(X %*% W, 0)
  t(apply(eta, 1, softmax_row))
}

predict <- function(W_samps, X) {
  # average predicted probabilities over all posterior samples
  num_samples <- dim(W_samps)[1]
  probability_accumulator <- matrix(0, nrow(X), class_count)
  for (s in seq_len(num_samples)) probability_accumulator <- probability_accumulator + predict_probs(W_samps[s, , ], X)
  probability_accumulator / num_samples
}

conf_matrix <- function(y_true, y_pred) {
  cm <- matrix(0, class_count, class_count,
               dimnames = list(Actual = CLASS_LEVELS, Predicted = CLASS_LEVELS))
  for (i in seq_along(y_true))
    cm[y_true[i], y_pred[i]] <- cm[y_true[i], y_pred[i]] + 1
  cm
}

class_metrics <- function(cm) {
  do.call(rbind, lapply(seq_len(class_count), function(i) {
    true_pos <- cm[i, i]
    false_pos <- sum(cm[, i]) - true_pos
    false_neg <- sum(cm[i, ]) - true_pos

    precision <- if (true_pos + false_pos == 0) 0 else true_pos / (true_pos + false_pos)
    recall <- if (true_pos + false_neg == 0) 0 else true_pos / (true_pos + false_neg)
    f1 <- if (precision + recall == 0) 0 else 2 * precision * recall / (precision + recall)
    data.frame(Class = CLASS_LEVELS[i], Precision = precision,
               Recall = recall, F1 = f1, Support = sum(cm[i, ]))
  }))
}

log_loss_fn <- function(y_true, probs) {
  log_clip <- 1e-15
  -mean(sapply(seq_along(y_true),
               function(i) log(max(probs[i, y_true[i]], log_clip))))
}

evaluate_and_plot <- function(post_chain, X_train, X_test, y_train, y_test, sample_idx) {
  W_post_mean <- apply(post_chain, c(2, 3), mean)
  W_post_sd <- apply(post_chain, c(2, 3), sd)

  if (verbose) cat("Averaging over posterior samples...\n")
  train_probs <- predict(post_chain, X_train)
  test_probs <- predict(post_chain, X_test)
  train_pred <- apply(train_probs, 1, which.max)
  test_pred <- apply(test_probs, 1, which.max)

  cm_train <- conf_matrix(y_train, train_pred)
  cm_test <- conf_matrix(y_test, test_pred)
  met_train <- class_metrics(cm_train)
  met_test <- class_metrics(cm_test)
  acc_train <- sum(diag(cm_train)) / sum(cm_train)
  acc_test <- sum(diag(cm_test)) / sum(cm_test)
  ll_train <- log_loss_fn(y_train, train_probs)
  ll_test <- log_loss_fn(y_test, test_probs)

  # confusion matrix heatmap
  cm_df <- as.data.frame(as.table(cm_test))
  colnames(cm_df) <- c("Actual", "Predicted", "Count")
  cm_df$Actual <- factor(cm_df$Actual, levels = rev(CLASS_LEVELS))
  cm_df$Predicted <- factor(cm_df$Predicted, levels = CLASS_LEVELS)

  p_cm <- ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Count)) +
    geom_tile(color = "white", linewidth = 0.8) +
    geom_text(aes(label = Count), size = 5, fontface = "bold") +
    scale_fill_gradient(low = "#f7f7f7", high = "#2166ac") +
    labs(title = "Confusion Matrix — Test Set",
         subtitle = sprintf("Accuracy: %.1f%% | Log-loss: %.4f",
                            100 * acc_test, ll_test),
         x = "Predicted", y = "Actual") +
    theme_bayes() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))

  ggsave(file.path(PLOT_DIR, "confusion_matrix_test.png"),
         p_cm, width = 7, height = 5, dpi = 150)

  # per-class F1 — train vs test
  f1_long <- rbind(
    data.frame(Class = met_train$Class, F1 = met_train$F1, Split = "Train"),
    data.frame(Class = met_test$Class, F1 = met_test$F1, Split = "Test")
  )
  f1_long$Class <- factor(f1_long$Class, levels = CLASS_LEVELS)

  p_f1 <- ggplot(f1_long, aes(x = Class, y = F1, fill = Split)) +
    geom_bar(stat = "identity", position = position_dodge(0.7),
             width = 0.6, color = "white") +
    geom_text(aes(label = sprintf("%.2f", F1)),
              position = position_dodge(0.7), vjust = -0.4, size = 3.2) +
    scale_fill_manual(values = c("Train" = "#4393c3", "Test" = "#d6604d")) +
    ylim(0, 1.1) +
    labs(title = "Per-Class F1 Score — Train vs. Test",
         x = "Difficulty Class", y = "F1 Score", fill = "Split") +
    theme_bayes() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))

  ggsave(file.path(PLOT_DIR, "per_class_f1.png"),
         p_f1, width = 7, height = 4.5, dpi = 150)

  # --- Plot: Predicted probability grid (true class x predicted class) ---
  prob_df <- as.data.frame(test_probs)
  colnames(prob_df) <- CLASS_LEVELS
  prob_df$TrueClass <- CLASS_LEVELS[y_test]

  prob_long <- reshape(prob_df,
                       varying = CLASS_LEVELS,
                       v.names = "Probability",
                       timevar = "PredClass",
                       times = CLASS_LEVELS,
                       direction = "long")
  prob_long$TrueClass <- factor(prob_long$TrueClass, levels = CLASS_LEVELS)
  prob_long$PredClass <- factor(prob_long$PredClass, levels = CLASS_LEVELS)

  p_prob <- ggplot(prob_long, aes(x = Probability, fill = PredClass)) +
    geom_histogram(bins = 30, color = "white", alpha = 0.85) +
    facet_grid(TrueClass ~ PredClass, scales = "free_y") +
    scale_fill_manual(values = c("#4dac26", "#b8e186", "#f1b6da", "#d01c8b")) +
    labs(title = "Predicted Class Probabilities by True Class — Test Set",
         subtitle = "Rows = true class | Columns = predicted class probability",
         x = "Predicted Probability", y = "Count") +
    theme_bayes() +
    theme(legend.position = "none",
          axis.text.x = element_text(size = 7))

  ggsave(file.path(PLOT_DIR, "predicted_probability_grid.png"),
         p_prob, width = 10, height = 7, dpi = 150)

  if (verbose) cat("Saved evaluation and probability plots.\n")

  list(cm_train = cm_train, cm_test = cm_test,
       met_train = met_train, met_test = met_test,
       acc_train = acc_train, acc_test = acc_test,
       ll_train = ll_train, ll_test = ll_test,
       W_post_mean = W_post_mean, W_post_sd = W_post_sd,
       train_probs = train_probs, test_probs = test_probs)
}

print_summary <- function(results, mcmc, data_splits, sample_idx) {
  cat(sprintf("\nTrain Accuracy : %.4f\n", results$acc_train))
  cat(sprintf("Test Accuracy : %.4f\n", results$acc_test))
  cat(sprintf("Train Log-Loss : %.4f\n", results$ll_train))
  cat(sprintf("Test Log-Loss : %.4f\n", results$ll_test))

  cat("\nTest set metrics\n")
  cat(sprintf("%-30s %9s %9s %9s %9s\n",
              "Class", "Precision", "Recall", "F1", "Support"))
  for (i in seq_len(class_count)) {
    cat(sprintf("%-30s %9.4f %9.4f %9.4f %9d\n",
                results$met_test$Class[i], results$met_test$Precision[i],
                results$met_test$Recall[i], results$met_test$F1[i],
                results$met_test$Support[i]))
  }
  cat(sprintf("%-30s %9.4f %9.4f %9.4f\n",
              "Macro Avg",
              mean(results$met_test$Precision),
              mean(results$met_test$Recall),
              mean(results$met_test$F1)))

  cat("\nPosterior Weight Summary\n")
  post_chain <- mcmc$post_chain
  feature_count <- dim(post_chain)[2]
  for (class_idx in seq_len(class_count - 1)) {
    cat(sprintf("\nClass: %-30s (vs. Difficult)\n", CLASS_LEVELS[class_idx]))
    for (feature_idx in seq_len(feature_count)) {
      weight_samples <- post_chain[, feature_idx, class_idx]
      cat(sprintf("%-20s Mean: %.4f Standard Deviation: %.4f 95 CI: [%.4f, %.4f]\n",
                  FEAT_NAMES[feature_idx], mean(weight_samples), sd(weight_samples),
                  quantile(weight_samples, 0.025), quantile(weight_samples, 0.975)))
    }
  }
}

save_outputs <- function(results, mcmc, data_splits, sample_idx) {
  w_mean_df <- data.frame(
    feature = FEAT_NAMES,
    setNames(as.data.frame(results$W_post_mean),
             paste0("class_", CLASS_LEVELS[seq_len(class_count - 1)])))
  w_sd_df <- data.frame(
    feature = FEAT_NAMES,
    setNames(as.data.frame(results$W_post_sd),
             paste0("class_", CLASS_LEVELS[seq_len(class_count - 1)])))

  # saves a ton of the outputs to corresponding CSV files
  write.csv(w_mean_df,
            file.path(MODEL_OUT_DIR, "posterior_mean_weights.csv"), row.names = FALSE)
  write.csv(w_sd_df,
            file.path(MODEL_OUT_DIR, "posterior_sd_weights.csv"), row.names = FALSE)
  write.csv(
    data.frame(split = c("train", "test"),
               accuracy = c(results$acc_train, results$acc_test),
               log_loss = c(results$ll_train, results$ll_test),
               macro_precision = c(mean(results$met_train$Precision),
                                   mean(results$met_test$Precision)),
               macro_recall = c(mean(results$met_train$Recall),
                                   mean(results$met_test$Recall)),
               macro_f1 = c(mean(results$met_train$F1),
                                   mean(results$met_test$F1))),
    file.path(MODEL_OUT_DIR, "evaluation_metrics.csv"), row.names = FALSE)
  write.csv(results$met_test,
            file.path(MODEL_OUT_DIR, "per_class_metrics_test.csv"), row.names = FALSE)
  write.csv(results$met_train,
            file.path(MODEL_OUT_DIR, "per_class_metrics_train.csv"), row.names = FALSE)
  write.csv(
    data.frame(
      parameter = c("n_iter", "burnin", "burnin_ratio", "thin",
                    "proposal_sd", "sigma_prior", "n_samples_kept", "acceptance_rate"),
      value = c(N_ITER, BURNIN, BURNIN_RATIO, THIN,
                    PROPOSAL_SD, SIGMA_PRIOR, sample_idx, mcmc$acceptance_rate)),
    file.path(MODEL_OUT_DIR, "mcmc_config.csv"), row.names = FALSE)

  cat("\nSaved to", MODEL_OUT_DIR, "\n")
  cat("posterior_mean_weights.csv posterior_sd_weights.csv\n")
  cat("per_class_metrics_test.csv per_class_metrics_train.csv\n")
  cat("mcmc_config.csv\n")
  cat(sprintf("Plots: %s\n", PLOT_DIR))
}

main <- function() {
  dir.create(MODEL_OUT_DIR, recursive = TRUE, showWarnings = FALSE)
  dir.create(PLOT_DIR, recursive = TRUE, showWarnings = FALSE)

  # load & preprocess
  trail_df <- load_and_preprocess()

  # balance classes
  df_bal <- balance_classes(trail_df)

  # train/test split & standardize
  data_splits <- split_and_standardize(df_bal)

  # run MCMC
  mcmc <- run_mcmc(data_splits$X_train, data_splits$y_train, data_splits$feature_count, file.path(MODEL_OUT_DIR, "posterior_mean_weights.csv"))

  # MCMC diagnostic plots (chain + ACF, with thinning)
  plot_mcmc(mcmc$post_chain, mcmc$sample_idx)

  # posterior distribution plots
  plot_posteriors(mcmc$post_chain, mcmc$sample_idx)

  # evaluate and plot predictions
  results <- evaluate_and_plot(
    mcmc$post_chain,
    data_splits$X_train, data_splits$X_test,
    data_splits$y_train, data_splits$y_test,
    mcmc$sample_idx
  )

  print_summary(results, mcmc, data_splits, mcmc$sample_idx)
  save_outputs(results, mcmc, data_splits, mcmc$sample_idx)
  mcmc
}

# so that it gets saved to the R studio environment:
mcmc <- main()