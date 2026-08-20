Traceback (most recent call last):
  File "C:\Users\P90022569\Downloads\indexing\indexing\chroma_main\step8b.py", line 852, in <module>
    main()
  File "C:\Users\P90022569\Downloads\indexing\indexing\chroma_main\step8b.py", line 833, in main
    results = asyncio.run(run_for_drug(
              ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\P90022569\.conda\envs\cognito\Lib\asyncio\runners.py", line 190, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "C:\Users\P90022569\.conda\envs\cognito\Lib\asyncio\runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\P90022569\.conda\envs\cognito\Lib\asyncio\base_events.py", line 654, in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
  File "C:\Users\P90022569\Downloads\indexing\indexing\chroma_main\step8b.py", line 797, in run_for_drug
    _write_output(drug_name, result, output_dir)
  File "C:\Users\P90022569\Downloads\indexing\indexing\chroma_main\step8b.py", line 743, in _write_output
    flags = f" [{', '.join(gap['flags'])}]" if gap.get("flags") else ""
                 ^^^^^^^^^^^^^^^^^^^^^^^
TypeError: sequence item 0: expected str instance, dict found
