.PHONY: all env_model env_engine validate benchmark clean help

PIP_INSTALL := pip install --user --force-reinstall --no-deps

all: env_model env_engine

env_model:
	cd env_model && maturin build --release
	$(PIP_INSTALL) env_model/target/wheels/orbit_wars_model-*.whl

env_engine:
	cd env_engine && maturin build --release
	$(PIP_INSTALL) env_engine/target/wheels/orbit_wars_engine-*.whl

validate:
	cd env_model && python validate.py
	cd env_engine && python validate.py

benchmark:
	cd env_model && python benchmark.py
	cd env_engine && python benchmark.py

clean:
	rm -rf env_model/target env_engine/target

help:
	@echo "Targets:"
	@echo "  all         build + install both rust modules (default)"
	@echo "  env_model   build + install orbit_wars_model"
	@echo "  env_engine  build + install orbit_wars_engine"
	@echo "  validate    run both validators against kaggle"
	@echo "  benchmark   run both benchmarks"
	@echo "  clean       remove target/ dirs"
