This projects made several modifications to the Source Codes, so you need to correspondingly change some files. Please follow the steps below:
1. find and download the original and complete documents of the source projects in these 2 links:
   https://www.synapse.org/Synapse:syn61941777/wiki/629245
   https://www.synapse.org/Synapse:syn61914254/wiki/629194
   
2. add files from the source project that are not in my project, do not overwrite.

3. now you can run all the codes. In case there is any important changed files I did not incorporate in my project, check the details about my modifications.
   
   In the model1-belfaction/src/tree_embeddings/experiments.py, I added experiment_21_oof(), experiment_5_oof(), experiment_1_oof(), experiment_23_oof_feat(), experiment_23_leaderboard_feat() to generate oofs and feats,
   and changed the apply_model() to avoid data leakage.
   
   The model2-D2Smell/augmentation/gt_with_dataset_Original.csv and relevant output gt_with_dataset_V2_augmented_dataset.csv were changed to correct ones. Do not overwrite.

   model2-D2Smell/augmentation/mixture_regressor_test.py is a new file added.
   
4. To run the codes in this project, you need to first generate from(at least):
   model1-belfaction/src/tree_embeddings/experiments.py (experiment_23_oof_feat())
   model2-D2Smell/augmentation/mixture_regressor_test.py

5. then you can run feature_process.py to process and concat features

9. run stacking.py to get results from Quadratic Linear Stacking and Residual Stacking using LGBM
