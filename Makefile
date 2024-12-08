install:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

run:
	. venv/bin/activate && python app.py

clean:
	rm -rf venv