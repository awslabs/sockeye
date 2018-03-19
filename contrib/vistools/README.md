# VisTools

This package generates javascript-based graphs of the beam search performed by sockeye. The graphs show  which nodes were expended at each step, at which step each hypothesis in the beam finished, the tokens chosen at each step, and the normalised and unnormalised scores at each state in the graph. This project is a modified form of [VisTools](https://github.com/OpenNMT/VisTools).

## Getting started

First, install the dependencies required for the visualizations:

```sh
pip install -r contrib/vistools/requirements.txt
```

### Store the beam histories

By default, sockeye will not store the whole beam search history. In order to obtain it, inference needs to be run with `--output-type beam_store`. E.g.:

```
python3 -m sockeye.translate --models model \
                             --input test.txt \
                             --output beams.json \
                             --output-type beam_store \
                             --beam-size 5
```

### Generate the graphs

After inference, the graphs can be generated with:

```
python3 contrib/vistools/generate_graphs.py -d beams.json -o generated_graphs
```

The `generated_graphs/` folder will contain one `html` file per sentence. Opening it in your browser will show the interactive graph.

## Running tests

To run the tests, run `pytest` from the `vistools` folder. A blank `pytest.ini` has been added so sockeye's main `pytest.ini` doesn't interfere with these tests.
