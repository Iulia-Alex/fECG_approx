# Plot inference on Test_DB

Run v15/v16/v17 inference on all 11 Test_DB signals and generate comparison plots.

## Steps

1. Run the inference script using the correct Python environment:
```bash
cd /shared_storage/iulia.orvas/paper/fECG_approx/src_2026 && /home/iulia.orvas/miniconda3/envs/ecg/bin/python3 infer_testdb_v15v16v17.py 2>&1
```

2. After the script finishes, list the generated plot files:
```bash
ls -1 /shared_storage/iulia.orvas/paper/fECG_approx/plots_2026/testdb/*_v15_v16_v17.png
```

3. Report:
   - Which model checkpoints were loaded (epoch numbers printed by the script)
   - The list of generated .png files, one per line
   - Any signals that were skipped (SKIP: lines in output)

Keep the response concise: model versions + epoch numbers on one line, then the file list.
