.PHONY: run-api seed-index test

run-api:
	uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

seed-index:
	python backend/workers/seed_index.py --recreate

test:
	pytest
