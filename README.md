# Anatomical-Context-Loss

## How to Run

### Method 1: Using Docker Compose

1. Make sure you have Docker and Docker Compose installed on your system.

2. Clone the repository:

```
git clone https://github.com/AhmedNasr7/Anatomical-Context-Loss
```

3. Navigate to the project directory:

```
cd Anatomical-Context-Loss
```

4. Run the application using Docker Compose:

```
docker-compose up
```


5. Run the needed files, you can run loss.py to test the loss function or run train.py to run a simple pytorch training pipeline to test UNet and the loss function

```
python3 loss.py
python3 modes.py
python3 train.py
```


### Method 2: Using a Virtual Environment

1. Clone the repository:


```
git clone https://github.com/AhmedNasr7/Anatomical-Context-Loss
```


2. Navigate to the project directory:

```
cd Anatomical-Context-Loss
```


3. Create a virtual environment (optional but recommended):

```
python3 -m venv env
```


4. Activate the virtual environment:

- On Windows:
  ```
  .\env\Scripts\activate
  ```
- On macOS and Linux:
  ```
  source env/bin/activate
  ```

5. Install the required dependencies:

```
python3 -m pip install -r requirements.txt
```


6. Run the needed files, you can run loss.py to test the loss function or run train.py to run a simple pytorch training pipeline to test UNet and the loss function

```
python3 loss.py
python3 modes.py
python3 train.py
```






