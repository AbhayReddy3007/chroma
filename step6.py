(cognito) C:\Users\P90022569\Downloads\indexing\indexing>python step6.py --input C:\Users\P90022569\Downloads\indexing\indexing\chroma_main\patent_exports\Axitinib_20260811.csv --output_dir ./step6_output 
[Step 6] Reading: C:\Users\P90022569\Downloads\indexing\indexing\chroma_main\patent_exports\Axitinib_20260811.csv
[Step 6] 37 rows loaded, 47 columns
[Step 6] Jurisdictions in scope: ['EP', 'US']
[Step 6] TARGET_ENTRY_YEAR: AUTO
[Step 6] INCLUDE_FORECASTED: False

══════════════════════════════════════════════════════════════════════
  DRUG: __global__
══════════════════════════════════════════════════════════════════════
  ⚠  NO_BLOCKING_IN_SCOPE - confirm with counsel
  → Charting queue : step6_output\__global___charting_queue.json
Traceback (most recent call last):
  File "C:\Users\P90022569\Downloads\indexing\indexing\step6.py", line 435, in <module>
    main()
  File "C:\Users\P90022569\Downloads\indexing\indexing\step6.py", line 429, in main
    _write_outputs(drug_name, result, output_dir)
  File "C:\Users\P90022569\Downloads\indexing\indexing\step6.py", line 372, in _write_outputs
    if result["null_date_set"]:
       ~~~~~~^^^^^^^^^^^^^^^^^
KeyError: 'null_date_set'
