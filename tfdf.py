# tfdf.keras.get_all_models()
# [tensorflow_decision_forests.keras.RandomForestModel,
#  tensorflow_decision_forests.keras.GradientBoostedTreesModel,
#  tensorflow_decision_forests.keras.CartModel,
#  tensorflow_decision_forests.keras.DistributedGradientBoostedTreesModel]

# rf = tfdf.keras.RandomForestModel(hyperparameter_template="benchmark_rank1", task=tfdf.keras.Task.REGRESSION)
# ----------------------------------------------------------
import tensorflow as tf
import tensorflow_decision_forests as tfdf

label = 'SalePrice'
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task = tfdf.keras.Task.REGRESSION)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label, task = tfdf.keras.Task.REGRESSION)



rf = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION)
rf.compile(metrics=["mse"]) # Optional, you can use this to include a list of eval metrics
rf.fit(x=train_ds)


inspector = rf.make_inspector()
inspector.evaluation()


tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=0, max_depth=3)



print(f"Available variable importances:")
for importance in inspector.variable_importances().keys():
    print("\t", importance)
  
inspector.variable_importances()["NUM_AS_ROOT"]

# # Available variable importances:
# 	 SUM_SCORE
# 	 NUM_NODES
# 	 NUM_AS_ROOT
# 	 INV_MEAN_MIN_DEPTH

plt.figure(figsize=(12, 4))

import matplotlib.pyplot as plt
logs = rf.make_inspector().training_logs()
plt.plot([log.num_trees for log in logs], [log.evaluation.rmse for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("RMSE (out-of-bag)")
plt.show()

evaluation = rf.evaluate(x=valid_ds,return_dict=True)

for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")

# Mean decrease in AUC of the class 1 vs the others.
variable_importance_metric = "NUM_AS_ROOT"
variable_importances = inspector.variable_importances()[variable_importance_metric]

# Extract the feature name and importance values.
#
# `variable_importances` is a list of <feature, importance> tuples.
feature_names = [vi[0].name for vi in variable_importances]
feature_importances = [vi[1] for vi in variable_importances]
# The feature are ordered in decreasing importance value.
feature_ranks = range(len(feature_names))

bar = plt.barh(feature_ranks, feature_importances, label=[str(x) for x in feature_ranks])
plt.yticks(feature_ranks, feature_names)
plt.gca().invert_yaxis()

# TODO: Replace with "plt.bar_label()" when available.
# Label each bar with values
for importance, patch in zip(feature_importances, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{importance:.4f}", va="top")

plt.xlabel(variable_importance_metric)
plt.title("NUM AS ROOT of the class 1 vs the others")
plt.tight_layout()
plt.show()


