# python evaluate_gp_model.py \
# -d data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r01_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r03_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r05_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r10_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r15_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r30_unscaled.csv \
# -e experiments/r01_ahc_selected \
#    experiments/r03_ahc_selected \
#    experiments/r05_ahc_selected \
#    experiments/r10_ahc_selected \
#    experiments/r15_ahc_selected \
#    experiments/r30_ahc_selected \
# -s outputs/roc_test_ensemble_svgp_ahc_selected_1_30
python evaluate_gp_model.py \
-d data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r01_unscaled.csv \
   data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r03_unscaled.csv \
   data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r05_unscaled.csv \
   data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r10_unscaled.csv \
   data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r15_unscaled.csv \
   data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r30_unscaled.csv \
   data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r60_unscaled.csv \
   data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r150_unscaled.csv \
   data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r300_unscaled.csv \
   data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r600_unscaled.csv \
   data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r900_unscaled.csv \
   data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r1800_unscaled.csv \
   data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r2700_unscaled.csv \
   data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r3600_unscaled.csv \
   data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r5400_unscaled.csv \
-e experiments/r01_ahc_selected \
   experiments/r03_ahc_selected \
   experiments/r05_ahc_selected \
   experiments/r10_ahc_selected \
   experiments/r15_ahc_selected \
   experiments/r30_ahc_selected \
   experiments/r60_ahc_selected \
   experiments/r150_ahc_selected \
   experiments/r300_ahc_selected \
   experiments/r600_ahc_selected \
   experiments/r900_ahc_selected \
   experiments/r1800_ahc_selected \
   experiments/r2700_ahc_selected \
   experiments/r3600_ahc_selected \
   experiments/r5400_ahc_selected \
-s outputs/roc_test_ensemble_svgp_ahc_selected_1_5400
# python evaluate_gp_model.py \
# -d data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r60_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r150_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r300_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r600_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r900_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r1800_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r2700_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r3600_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r5400_unscaled.csv \
# -e experiments/r60_ahc_selected \
#    experiments/r150_ahc_selected \
#    experiments/r300_ahc_selected \
#    experiments/r600_ahc_selected \
#    experiments/r900_ahc_selected \
#    experiments/r1800_ahc_selected \
#    experiments/r2700_ahc_selected \
#    experiments/r3600_ahc_selected \
#    experiments/r5400_ahc_selected \
# -s outputs/roc_test_ensemble_svgp_ahc_selected_60_5400
# python evaluate_gp_model.py \
# -d data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r01_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r03_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r05_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r15_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r30_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r60_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r150_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r300_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r600_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r900_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r1800_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r2700_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r3600_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r5400_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r7200_unscaled.csv \
# -e experiments/r01_ahc_selected \
#    experiments/r03_ahc_selected \
#    experiments/r05_ahc_selected \
#    experiments/r15_ahc_selected \
#    experiments/r30_ahc_selected \
#    experiments/r60_ahc_selected \
#    experiments/r150_ahc_selected \
#    experiments/r300_ahc_selected \
#    experiments/r600_ahc_selected \
#    experiments/r900_ahc_selected \
#    experiments/r1800_ahc_selected \
#    experiments/r2700_ahc_selected \
#    experiments/r5400_ahc_selected \
#    experiments/r7200_ahc_selected \
# -s outputs/roc_test_ensemble_svgp_ahc_selected_long

# python evaluate_gp_model.py \
# -d data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r01_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r03_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r05_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r15_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r30_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r60_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r150_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r300_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r600_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r900_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r1800_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r2700_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r3600_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r5400_unscaled.csv \
#    data/narco_features/avg_kw21_long_ahc/avg_kw21_long_r7200_unscaled.csv \
# -e experiments/r01_vgp_ahc \
#    experiments/r03_vgp_ahc \
#    experiments/r05_vgp_ahc \
#    experiments/r15_vgp_ahc \
#    experiments/r30_vgp_ahc \
#    experiments/r60_vgp_ahc \
#    experiments/r150_vgp_ahc \
#    experiments/r300_vgp_ahc \
#    experiments/r600_vgp_ahc \
#    experiments/r900_vgp_ahc \
#    experiments/r1800_vgp_ahc \
#    experiments/r2700_vgp_ahc \
#    experiments/r5400_vgp_ahc \
#    experiments/r7200_vgp_ahc \

# -d  data/narco_features/avg_kw21_long_r01_test_unscaled.csv \
#     data/narco_features/avg_kw21_long_r03_test_unscaled.csv \
#     data/narco_features/avg_kw21_long_r05_test_unscaled.csv \
#     data/narco_features/avg_kw21_long_r10_test_unscaled.csv \
#     data/narco_features/avg_kw21_long_r15_test_unscaled.csv \
#     data/narco_features/avg_kw21_long_r30_test_unscaled.csv \
#     data/narco_features/avg_kw21_long_r60_test_unscaled.csv \
#     data/narco_features/avg_kw21_long_r150_test_unscaled.csv \
#     data/narco_features/avg_kw21_long_r300_test_unscaled.csv \
#     data/narco_features/avg_kw21_long_r600_test_unscaled.csv \
#     data/narco_features/avg_kw21_long_r900_test_unscaled.csv \
#     data/narco_features/avg_kw21_long_r1800_test_unscaled.csv \
#     data/narco_features/avg_kw21_long_r2700_test_unscaled.csv \
#     data/narco_features/avg_kw21_long_r5400_test_unscaled.csv \
#     data/narco_features/avg_kw21_long_r7200_test_unscaled.csv \
# -e  experiments/r01_vgp \
#     experiments/r03_vgp \
#     experiments/r05_vgp \
#     experiments/r10_vgp \
#     experiments/r15_vgp \
#     experiments/r30_vgp \
#     experiments/r60_vgp \
#     experiments/r150_vgp \
#     experiments/r300_vgp \
#     experiments/r600_vgp \
#     experiments/r900_vgp \
#     experiments/r1800_vgp \
#     experiments/r2700_vgp \
#     experiments/r5400_vgp \
#     experiments/r7200_vgp \
