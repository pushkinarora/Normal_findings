numeric_features = all_data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = all_data.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers = [
    ("num", numerical_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

model_KRR = Pipeline(steps = [
    ("preprocessor", preprocessor),
    ("regressor", KernelRidge(alpha=0.2, kernel='polynomial', degree=2, coef0=2.5))
])

model_LightGBM = Pipeline(steps =[
    ("preprocessor",preprocessor),
    ("regressor",LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=1020,
                              max_bin = 685, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11))
])

model_XGBoost = Pipeline(steps = [
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(objective='reg:squarederror', n_estimators=1090, learning_rate=0.02, max_depth=3, subsample=0.8, colsample_bytree=0.8))
])

train_data = all_data.xs("train")
test_data = all_data.xs("test")

model_XGBoost.fit(train_data, y)
model_KRR.fit(train_data, y)
model_LightGBM.fit(train_data, y)

rmse_KRR = np.sqrt(-cross_val_score(model_KRR, X = train_data, y = y, cv = 5, scoring='neg_mean_squared_error'))
rmse_LGBM = np.sqrt(-cross_val_score(model_LightGBM, X = train_data, y = y, cv = 5, scoring='neg_mean_squared_error'))
rmse_XGB = np.sqrt(-cross_val_score(model_XGBoost, X = train_data, y = y, cv = 5, scoring='neg_mean_squared_error'))

logged_preds_KRR = model_KRR.predict(all_data.xs("test"))
logged_preds_XGB = model_XGBoost.predict(all_data.xs("test"))
logged_preds_LGB = model_LightGBM.predict(all_data.xs("test"))

KRR_pred = np.expm1(logged_preds_KRR)
XGB_pred = np.expm1(logged_preds_XGB)
LGB_pred = np.expm1(logged_preds_LGB)

final_preds = 0.02*XGB_pred+0.35*LGB_pred+0.63*KRR_pred
