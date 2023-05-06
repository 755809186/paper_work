from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score
model =load_model('model_cnn_text.h5')
X_test = open("./data3/ase.txt","r")

print(X_test)
y_prob = model.predict(X_test)[:, 0]
y_pred = np.round(y_prob)
accuracy_test=accuracy_score(X_test,y_pred)
print(accuracy_test)