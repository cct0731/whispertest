import sys
from tensorflow.keras.models import load_model
import ktrain


predictor = ktrain.load_predictor("test_model")

while True:


    rcv_data = sys.stdin.readline().strip()

    if not rcv_data:
        break

    prediction = predictor.predict(rcv_data)
    print(prediction)
    sys.stdout.flush()