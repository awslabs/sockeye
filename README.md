
# NAACL 2022 publication: The Devil is in the Details: On the Pitfalls of Vocabulary Selection in Neural Machine Translation

This branch contains the code for our publication:
* Tobias Domhan, Eva Hasler, Ke Tran, Sony Trenous, Bill Byrne and Felix Hieber. "The Devil is in the Details: On the Pitfalls of Vocabulary Selection in Neural Machine Translation". Proceedings of NAACL-HLT (2022)


```
@inproceedings{domhan-etal-2022,
    title = "The Devil is in the Details:On the Pitfalls of Vocabulary Selection in Neural Machine Translation",
    author = "Domhan, Tobias  and
      Hasler, Eva  and
      Tran, Ke and
      Trenous, Sony and
       Byrne, Bill and
       Hieber, Felix",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, USA",
    publisher = "Association for Computational Linguistics",
}
```


To use NVS simply specify `--neural-vocab-selection --bow-task-pos-weight 100000` to `sockeye-train`.
This will train a model with Neural Vocabulary Selection that is automatically used by `sockeye-translate`.
If you want look at translations without vocabulary selection specify `--skip-nvs` as an argument to `sockeye-translate`.
For the NVS model see `sockeye/nvs.py`.

## Wikiquote test set
See folder `naacl2022/wikiquote` for the wikiquote test set.
