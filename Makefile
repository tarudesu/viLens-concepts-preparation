.DEFAULT_GOAL := all

PYTHON ?= uv run --active --no-sync python
CONFIG ?= configs/data.yaml
BUILD_DIR := $(shell $(PYTHON) -B -c 'from data.build.common import load_config; import sys; print(load_config(sys.argv[1])["paths"]["build"])' "$(CONFIG)")
ifeq ($(strip $(BUILD_DIR)),)
$(error Unable to read paths.build from $(CONFIG))
endif
STEP_SCRIPTS := $(sort $(wildcard $(BUILD_DIR)/[0-9][0-9]_*.py $(BUILD_DIR)/[0-9][0-9][a-z]_*.py))
DISCOVERED_STEP_NUMBERS := $(foreach script,$(STEP_SCRIPTS),$(firstword $(subst _, ,$(notdir $(script)))))
# This is dependency order, not lexical order: 09b enriches step-11 outputs.
STEP_ORDER := 00 01 02 03 04 05 06 07 08 09 10 10b 11 11b 09b 12 13 13b 14
STEP_NUMBERS := $(filter $(DISCOVERED_STEP_NUMBERS),$(STEP_ORDER))
STEP_TARGETS := $(addprefix step-,$(STEP_NUMBERS))

ifneq ($(words $(STEP_NUMBERS)),$(words $(sort $(STEP_NUMBERS))))
$(error Expected exactly one script per step in $(BUILD_DIR))
endif

.PHONY: all download pool attest split filter backtranslate etymology covariates tokens diacritics m1-extension prompts directions-matched concreteness fertility directions-flores release release-package test $(STEP_TARGETS)
# Steps consume previous steps' outputs, including when called with make -j.
.NOTPARALLEL:

all: $(STEP_TARGETS)

# Concrete dependency edges: 09b consumes the completed main/extension step-11
# tables; prompts consume the resolved values; release follows both prompt paths.
step-09b: step-11b
step-12: step-09b
step-14: step-12 step-13b

$(STEP_TARGETS): step-%:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) -B $(filter $(BUILD_DIR)/$*_%.py,$(STEP_SCRIPTS)) --config "$(CONFIG)"

download: step-01

pool: step-03

attest: step-04

split: step-05

filter: step-06

backtranslate: step-07

etymology: step-08

covariates: step-09

tokens: step-10

diacritics: step-11

m1-extension:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) -B $(BUILD_DIR)/11b_m1_extension.py --config "$(CONFIG)"

prompts: step-12

directions-matched: step-13

concreteness:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) -B $(BUILD_DIR)/09b_concreteness.py --config "$(CONFIG)"

fertility:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) -B $(BUILD_DIR)/10b_fertility.py --config "$(CONFIG)"

directions-flores:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) -B $(BUILD_DIR)/13b_direction_flores.py --config "$(CONFIG)"

release: step-14

release-package:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) -B $(BUILD_DIR)/15_release_package.py --config "$(CONFIG)"

test:
	$(PYTHON) -B -m pytest
