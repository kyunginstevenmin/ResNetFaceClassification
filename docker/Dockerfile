FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu118-ubuntu20.04-sagemaker

COPY docker/requirements.txt /opt/ml/code/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt

# Copy both model.py and train.py — train.py imports from model.py
COPY src/ /opt/ml/code/

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM train.py

# Use exec form — ensures SIGTERM reaches Python, not a shell wrapper
ENTRYPOINT ["python", "/opt/ml/code/train.py"]
