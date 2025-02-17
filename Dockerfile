FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency files first to leverage caching
COPY requirements.txt dev-requirements.txt ./

# Upgrade pip and install dependencies in one step
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r dev-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose Jupyter's default port
EXPOSE 8888

# Run Jupyter Lab with recommended settings
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
