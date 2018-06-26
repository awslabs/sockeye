# CPU process per core translation
On multi-core processor computer, translation per core separately can speedup translation performance, due to some operation can't be handled parallel in one process.
Using this method, translation on each core can be parallel.

One python script example is given and you can run it as follows:
```bash
> python cpu_process_per_core_translation.py -m model -i input_file_name -o output_file_name -bs batch_size -t true
```

-t true: each core translate the whole input file.

-t false: each core translate (input file line/core number) lines , then merge the translated file into one complete output file.

