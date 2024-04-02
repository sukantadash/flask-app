FROM registry.access.redhat.com/ubi8/python-39:1-97

USER 0

RUN mkdir /app
RUN mkdir /app/templates
RUN mkdir -p /app/cache

ENV TRANSFORMERS_CACHE=/app/cache/

COPY requirements.txt /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py
COPY templates/form.html /app/templates/form.html
COPY future_trend.onnx /app/

RUN chown -R 1001:0 /app\
&&  chmod -R og+rwx /app \
&&  chmod -R +x /app

WORKDIR /app

EXPOSE 5000

ENV PYTHONPATH=/app

USER 1001

CMD ["python", "-m" ,"flask", "run", "--host=0.0.0.0" ]