import pandas as pd

df = pd.read_csv('combined_all_search_results.csv')

df = df.drop(columns=['history','source_file',"config_id","training_time_sec","mean_val_r2","std_val_r2","min_val_r2","max_val_r2","fold_results",
                      "total_kfold_time","final_val_r2","final_train_r2","final_overfitting_gap","final_best_epoch"
                      ,"final_total_epochs","use_attention","attention_heads","attention_dropout"])

df = df.sort_values(by="best_val_r2", ascending=False)

df.to_csv('final.csv')