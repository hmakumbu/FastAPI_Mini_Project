FROM python:3.12-slim

WORKDIR /app
COPY requirement.txt /app/
COPY . /app/
RUN pip install --no-cache-dir -r requirement.txt

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
