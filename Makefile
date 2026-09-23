.PHONY: install data pipeline demo test api app docker

install:
	pip install -r requirements.txt && pip install -e .

# Télécharge Favorita (nécessite ~/.kaggle/kaggle.json et d'avoir accepté les règles de la compétition)
data:
	kaggle competitions download -c store-sales-time-series-forecasting -p data/raw
	cd data/raw && unzip -o store-sales-time-series-forecasting.zip

pipeline:            # backtest + rapport + modèle final sur Favorita
	python -m demandforecast.pipeline --data-dir data/raw --horizon 14 --n-folds 6

demo:                # même chose sur données synthétiques (pas de téléchargement)
	python -m demandforecast.pipeline --synthetic

test:
	pytest -q

api:
	uvicorn api.main:app --reload

app:
	streamlit run app/streamlit_app.py

docker:
	docker build -t demand-forecasting-api . && docker run -p 8000:8000 demand-forecasting-api
