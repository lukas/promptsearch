# promptsearch

Inspired by [Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution](https://arxiv.org/abs/2309.16797)

Searches for better prompts given a model, dataset and evaluation.

Need to set up weave notion of model, dataset and evaluation - see https://github.com/wandb/weave

```python
ps = PromptSearch(model=model, dataset=dataset, evaluation=evaluation)
ps.steps(10)
```

Try example on hellaswag dataset
```bash
PYTHONPATH=. python examples/hellaswag/hellaswag.py
```
