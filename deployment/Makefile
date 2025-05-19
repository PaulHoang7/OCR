.PHONY: init deploy deploy_ocr client clean streamlit

IS_WINDOWS := $(findstring Windows_NT,$(OS))

clean:
	ray stop || true
ifeq ($(IS_WINDOWS),Windows_NT)
	@if exist %TEMP%\ray_project rmdir /s /q %TEMP%\ray_project
else
	rm -rf /tmp/ray_project/*
endif

init: clean
ifeq ($(IS_WINDOWS),Windows_NT)
	mkdir %TEMP%\ray_project
	ray start --head --disable-usage-stats --temp-dir=%TEMP%\ray_project --dashboard-host=0.0.0.0
else
	mkdir -p /tmp/ray_project
	ray start --head --disable-usage-stats --temp-dir=/tmp/ray_project --dashboard-host=0.0.0.0
endif

deploy:
	@echo "Starting object detection server..."
	serve run object_detection:entrypoint --address="auto"

deploy_ocr:
	@echo "Starting OCR server..."
	serve run ocr:entrypoint --address="auto"

streamlit:
	@echo "Starting Streamlit app..."
	streamlit run app.py
