# PyTorch Transformer

**Heavily-commented "Annotated Transformer" (cf. A.M. Rush's tutorial)**

## Objective

* Clarify the nice [Transformer tutorial by A.M. Rush](https://nlp.seas.harvard.edu/2018/04/03/attention.html), which leaves out details I believe could be helpful to a newbie as I was (e.g. tensor shapes, self-documentation variable naming, etc.)
* Complete the tutorial code with data loading/formatting facitilities, and demo in a paraphrasing example with toy data ("real data" can be downloaded from, e.g. [Prakash/16's paraphrasing datasets](https://github.com/iamaaditya/neural-paraphrase-generation/tree/add-license-1/data).

## Notes

* To see usage, check out `example_train.py` and `example_evaluate.py`.
* Run `example_train.py` to save model and config, which are prerequisites for `example_evaluate.py`. 

## Version Info

* torch=1.1.0
* cuda=9.2
