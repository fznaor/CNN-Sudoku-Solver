import pandas as pd
import numpy as np
import keras

# load test data and model
data = pd.read_csv('sudoku_test.csv')
model = keras.models.load_model('model')

# reshape test data
X_test = np.array([np.reshape([int(d) for d in flatten_grid], (9, 9, 1)) for flatten_grid in data.quizzes])
y_test = np.array([np.reshape([int(d) for d in flatten_grid], (81, 1)) for flatten_grid in data.solutions])

# convert test data into interval [0,8]
y_test = y_test - 1

correct = 0

for i in range(len(X_test)):

    sudoku = X_test[i].reshape(1,9,9,1)
    
    # repeat while there are zeros in the sudoku
    while True:
        if not 0 in sudoku:
            break
        
        # get results predicted by the model and find the highest probability for a field that isn't already filled
        result = model.predict(sudoku)
        result = result.reshape(81,9)
        args = result.argmax(axis=1)
        sudoku = np.ravel(sudoku)
        result = result.max(axis=1)
        result = np.where(sudoku==0, result, 0)
        sudoku[np.argmax(result)] = args[np.argmax(result)]+1
        sudoku = sudoku.reshape(1,9,9,1)

    # check whether the solution is correct
    if(np.array_equiv(sudoku,(y_test[i]+1).reshape(9,9,1))):
        correct += 1

print(str(correct) + '/' + str(i+1))