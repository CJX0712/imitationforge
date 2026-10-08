# ImitationForge · 构建期硬门禁（lint + test 全绿方可完成镜像）。
# 作者：晨星
FROM python:3.13-slim

WORKDIR /app

COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY . .

# 构建期质量门禁：任何一步失败镜像即构建失败。
RUN python -m ruff check . \
 && python -m ruff format --check . \
 && PYTHONPATH=. python -m pytest -q

CMD ["sh", "-c", "PYTHONPATH=. python examples/run_demo.py"]
