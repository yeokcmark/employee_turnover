rm(list=ls())
pacman::p_load(tidyverse, lubridate,
               tidymodels, glmnet, discrim, naivebayes, LiblineaR, kernlab,
               skimr, doParallel, finetune,
               ranger, usemodels)
#import data
df <-read_csv("HRDataset_v14.csv")
skim(df)

# check for NAs ----
sapply(df, FUN = function(x) sum(is.na(x)))
# 8 missing manager IDs Webster Butler ID 39
NA_managerID <-df %>% 
  filter(if_any(ManagerID, is.na))
glimpse(NA_managerID)

df %>% 
  filter(ManagerName == "Webster Butler") %>% 
  glimpse()

# clean the data, get into correct class
data<-
  df %>% 
  mutate(Termd = as.factor(Termd),
         Termd = fct_relevel(Termd,
                             "1"),
         Zip = as.factor(Zip),
         EmpSatisfaction = as.factor(EmpSatisfaction),
         DOB = as.Date(DOB, "%m/%d/%y"),
         # typo error with DOB eg: year 2068, replace with NA
         DOB = case_when(DOB >= as.Date("1993/01/01") ~ NA,
                         .default = DOB),
         DateofHire = as.Date(DateofHire, "%m/%d/%y"),
         DateofTermination = as.Date(DateofTermination, "%m/%d/%y"),
         ManagerID = replace_na(ManagerID, 39),
         LastPerformanceReview_Date = as.Date(LastPerformanceReview_Date, "%m/%d/%y"),
         EmpSatisfaction = as.factor(EmpSatisfaction)
         ) %>% 
  mutate_at(vars(contains("ID")), as.factor) %>% 
  mutate_if(is.character, as.factor)
skim(data)

# Lets create additional features
# assume today is 1 Jan 2023
today_date <- as.Date("2023/01/01")

data<-
  data %>% 
  mutate(days_employed = as.numeric(today_date - DateofHire),
         cat_years_employed = cut(days_employed,
                                  breaks = c(0, 183, 365, 730, 1095, Inf),
                                  labels = c("less than half year",
                                             "half to one year",
                                             "one to two years",
                                             "two to 3 years",
                                             "more than 3 years")),
         age = as.numeric(today_date - DOB)/365) %>% 
  relocate(days_employed, age, cat_years_employed) %>% 
  dplyr::select(-DOB, -DateofHire, -DateofTermination, -LastPerformanceReview_Date,
                -Employee_Name, -Position, -Zip, -MaritalDesc, -TermReason, -ManagerName
                )
# ML question predict terminated 1/0
# split the data
data_split <-
  data %>% 
  initial_split(strata = Termd)
data_train <-
  data_split %>% 
  training()
data_test <-
  data_split %>% 
  testing()
data_fold <-
  data_train %>% 
  vfold_cv(v=10)

# recipe
rec_base <-
  recipe(formula = Termd ~.,
         data = data_train) %>%
  step_impute_bag(age,
                  trees = 50) %>% 
  update_role(EmpID, new_role = "id") %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_other(PositionID, State, ManagerID, threshold = tune("step_other")) %>% 
  step_dummy(all_nominal_predictors())

rec_base %>% prep() %>% juice()
  
rec_pca <-
  rec_base %>% 
  step_pca(all_predictors(), id = "pca", threshold = tune("steo_pca"))
# models

###LASSO
spec_Lasso <-
  logistic_reg(penalty = tune(), mixture = 1
  ) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

spec_Ridge <-
  logistic_reg(penalty = tune(), mixture = 0
  ) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

spec_svm <-
  svm_linear(cost = tune(), margin = tune()) %>%
  set_engine("kernlab") %>% 
  set_mode("classification")

spec_nb <-
  naive_Bayes() %>% 
  set_engine("naivebayes") %>% 
  set_mode("classification")

# random forest
spec_rf <-
  rand_forest() %>% 
  set_engine("ranger",
             importance = "impurity") %>% 
  set_mode("classification")

# %>% 
#   set_args(trees = 1000L,
#            mtry = tune("rf_mtry"),
#            min_n = tune("rf_min_n"))

# xgboost
spec_xgb <-
  boost_tree() %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")
# %>% 
#   set_args(trees = 1000L,
#            tree_depth = tune("xgb_tree_depth"),
#            min_n = tune("xgb_min_n"),
#            loss_reduction = tune("xgb_loss_red"),
#            sample_size = tune("xgb_sample"),
#            mtry = tune("xgb_mtry"),
#            learn_rate = tune("xgb_learn"),
#            stop_iter = 20)

## nnet
spec_nnet <- 
  mlp() %>% 
  set_engine("nnet", MaxNWts = 2600) %>% 
  set_mode("classification")
# %>% 
#   set_args(hidden_units = tune("nn_hidden"),
#            penalty = tune("nn_penalty"),
#            epochs = tune("nn_epochs")
#   )

# Logistic Regression Model
spec_logistic <- 
  logistic_reg() %>%
  set_engine(engine = 'glm') %>%
  set_mode('classification') 

# Logistic Regression Model using glmnet

spec_logistic_glmnet <-
  logistic_reg() %>%
  set_engine(engine = 'glmnet') %>%
  set_mode('classification')

# %>% 
#   set_args(penalty = tune("log_glmnet_penalty")
#   )

#null
spec_null <-
  null_model() %>% 
  set_mode("classification") %>% 
  set_engine("parsnip")


# define metrics
metric_model <-
  metric_set(roc_auc,
             accuracy,
             precision,
             recall,
             f_meas)

#### Using Workflowsets instead
base_set <-
  workflowsets::workflow_set(
    preproc = list(base = rec_base,
                   pca = rec_pca),
    models = list(Lasso = spec_Lasso,
                  Ridge=spec_Ridge,
                  NB = spec_nb,
                  SVM =spec_svm,
                  RF=spec_rf,
                  XGB=spec_xgb,
                  NN=spec_nnet,
                  Logistic=spec_logistic,
                  glmnet=spec_logistic_glmnet,
                  null=spec_null),
    cross = TRUE
  )

cl <- (detectCores()/2) - 1
cores <- cl*2
doParallel::registerDoParallel(cl, cores)

first_tune <-
  workflow_map(base_set,
               fn = "tune_grid",
               verbose = TRUE,
               seed = 2024043001,
               grid = 15,
               resamples = data_fold,
               metrics = metric_model,
               control = control_grid(verbose = TRUE,
                                      allow_par = TRUE,
                                      parallel_over = "everything"))


successful_tune <-
  first_tune[
    map_lgl(first_tune$result, 
            ~pluck(., ".metrics", 1) %>% inherits("tbl_df"), 
            "tune_result"),]

names_wflow_id <-
  successful_tune$wflow_id %>% 
  as_tibble() %>% 
  rename(wflow_id = value) %>% 
  mutate(recipe_name = paste0("rec_", str_extract(wflow_id, "\\w+(?=_)")),
         model_name = paste0("spec_", str_extract(wflow_id, "(?<=_)\\w+")))
str(names_wflow_id)


autoplot(successful_tune, select_best = T) + 
  theme_bw() + 
  theme(legend.position = "bottom")

autoplot(
  successful_tune,
  rank_metric = "f_meas",  # <- how to order models
  metric = "f_meas",       # <- which metric to visualize
  select_best = T     # <- one point per workflow
) +
  geom_text(aes(y = mean - 0.001, label = wflow_id), angle = 90, hjust = 1, check_overlap = T, size = 4) +
  theme(legend.position = "none")

successful_tune %>% 
  workflowsets::rank_results(rank_metric = "roc_auc", select_best = T) %>% 
  filter(.metric == "roc_auc") %>% 
  dplyr::select(wflow_id, mean, std_err, rank) %>% 
  datatable() %>% 
  formatRound(columns = c("mean", "std_err"),
              digits = 5)

successful_tune %>% 
  extract_workflow_set_result(id = "swunigrams_SVM") %>% 
  select_best(metric = "f_meas")

### further tune best recipe model combination


rec_swug_tune <-
  recipe(formula = target ~ text,
         data = data_train) %>% 
  step_tokenize(text,
                token = "words")%>% 
  step_tokenfilter(text,
                   min_times = 3L,
                   max_tokens = 4500L) %>% 
  step_tf(text) %>% 
  step_normalize(all_predictors())

rec_swug_tune %>% prep() %>% juice()

grid_tune <-
  grid_regular(parameters(cost(), min_times(range(c(1L,10L))), max_tokens(range(c(500L,4000L)))),
               levels = 100)

swug_svm_wflow <-
  workflow() %>% 
  add_recipe(rec_swug_tune) %>% 
  add_model(spec_svm)

svm_tune <-
  tune_grid(
    swug_svm_wflow,
    resamples = data_fold,
    grid = 20,
    metrics = metric_model,
    control = control_grid(verbose = TRUE,
                           allow_par = TRUE,
                           parallel_over = "everything")
  )

svm_tune[
  map_lgl(svm_tune$result, 
          ~pluck(., ".metrics", 1) %>% inherits("tbl_df"), 
          "tune_result"),]

save.image("employment_turnover.RData")
