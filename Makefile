# EmpowerLens Month-1 pipeline.
# Recipes assume a POSIX shell (Linux / Kaggle / Git-Bash on Windows).
# Override the interpreter to use the project venv, e.g.:
#   make splits PYTHON=venv/Scripts/python.exe      (Windows)
#   make splits PYTHON=venv/bin/python              (Linux/Mac)

PYTHON ?= python
SEEDS  ?= 42 1337 2024
ARGS   ?=

.PHONY: help splits baseline smoke binary multiclass multilabel evaluate aggregate test clean

help:
	@echo "Targets:"
	@echo "  splits      generate the frozen train/val/test splits (once; --force to redo)"
	@echo "  baseline    TF-IDF + LogReg floor over \$$SEEDS (val)"
	@echo "  smoke       CPU plumbing check: all 3 tasks, 100 rows, 1 epoch"
	@echo "  binary      fine-tune transformer, binary task, over \$$SEEDS"
	@echo "  multiclass  fine-tune transformer, 11-class task, over \$$SEEDS"
	@echo "  multilabel  fine-tune transformer, 10-label task, over \$$SEEDS"
	@echo "  evaluate    score a checkpoint on val+test (pass CKPT=checkpoints/...)"
	@echo "  aggregate   roll results/eval_*.json into the Month-1 summary tables"
	@echo "  test        run pytest"
	@echo "  clean       remove checkpoints/ and __pycache__/"

splits:
	$(PYTHON) -m src.make_splits $(ARGS)

baseline:
	$(PYTHON) -m src.baseline_classical --seeds $(shell echo $(SEEDS) | tr ' ' ',') $(ARGS)

smoke:
	$(PYTHON) -m src.train_transformer --task binary     --smoke --max-length 128 $(ARGS)
	$(PYTHON) -m src.train_transformer --task multiclass  --smoke --max-length 128 $(ARGS)
	$(PYTHON) -m src.train_transformer --task multilabel  --smoke --max-length 128 --truncation head_tail $(ARGS)

binary:
	@for s in $(SEEDS); do $(PYTHON) -m src.train_transformer --task binary     --seed $$s $(ARGS); done

multiclass:
	@for s in $(SEEDS); do $(PYTHON) -m src.train_transformer --task multiclass  --seed $$s $(ARGS); done

multilabel:
	@for s in $(SEEDS); do $(PYTHON) -m src.train_transformer --task multilabel  --seed $$s $(ARGS); done

evaluate:
	$(PYTHON) -m src.evaluate --checkpoint $(CKPT) --reference $(ARGS)

aggregate:
	$(PYTHON) -m src.aggregate $(ARGS)

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf checkpoints/ $(shell find . -name __pycache__ -type d)
