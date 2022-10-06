import pygame 
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from random import randrange

with tf.device('/cpu:0'):
    # load sudoku solving model
    model = keras.models.load_model('model')
    
    # set up screen
    pygame.font.init() 
    screen = pygame.display.set_mode((700, 900)) 
    screen.fill((255, 255, 255))
      
    dif = 700 / 9 # width and height of a sudoku cell

    # coordinates of selected cell
    x = 0
    y = 0

    # coordinates of last updated cell (via model)
    lastX = -1
    lastY = -1

    # probability of most recently filled cell
    maxProb = 0

    value = 0 # number to be added to the sudoku
    sudoku = np.zeros((9,9), dtype=np.int8)
    
    sudokuFile = pd.read_csv('sudoku_test.csv')
    defaultSudokus = np.array([np.reshape([int(d) for d in flatten_grid], (9, 9)) for flatten_grid in sudokuFile.quizzes])
    
    font1 = pygame.font.SysFont("comicsans", 60) 
    font2 = pygame.font.SysFont("arial", 15) 
    font3 = pygame.font.SysFont("comicsans", 40) 
    run = True # whether the game is running
    flagCell = False # checks whether the user has selected a cell
    flagFill = False # checks whether the most recently filled cell should be highlighted
    sudokuValid = True
    
    # return the xy coordinates of selected box
    def getCoords(pos): 
        x = pos[0]//dif 
        y = pos[1]//dif
        return x,y
    
    # draw a red square around the currently selected cell
    def drawBox(): 
        for i in range(2): 
            pygame.draw.line(screen, (255, 0, 0), (x * dif-3, (y + i)*dif), (x * dif + dif + 3, (y + i)*dif), 7) 
            pygame.draw.line(screen, (255, 0, 0), ( (x + i)* dif, y * dif ), ((x + i) * dif, y * dif + dif), 7)
            
    # display title 
    pygame.display.set_caption("Neural network sudoku solver") 
    
    def draw(sudoku):
        # highlight the last cell that the model filled
        pygame.draw.rect(screen, (127, 240, 115), (lastX * dif, lastY * dif, dif + 1, dif + 1))

        # draw the current state of the sudoku
        for i in range (9): 
            for j in range (9): 
                if sudoku[i][j]!= 0: 
                    text1 = font1.render(str(sudoku[i][j]), 1, (0, 0, 0)) 
                    screen.blit(text1, (j * dif + 30, i * dif + 23)) 

        # draw the grid           
        for i in range(10): 
            if i % 3 == 0 : 
                thick = 7
            else: 
                thick = 1
            pygame.draw.line(screen, (0, 0, 0), (0, i * dif), (700, i * dif), thick) 
            pygame.draw.line(screen, (0, 0, 0), (i * dif, 0), (i * dif, 700), thick)     
            
    # check if the cell coordinates lie within the sudoku
    def areCoordsValid(x,y):
        return x>=0 and x<=8 and y>=0 and y<=8
    
    # check if all rows are valid (no repeating values except for zeros)
    def validRow(row, grid):
      temp = grid[row]
      temp = list(filter(lambda a: a != 0, temp))
      if len(temp) != len(set(temp)):
        return False
      else:
        return True
    
    # check if all columns are valid (no repeating values except for zeros)
    def validCol(col, grid):
      temp = [row[col] for row in grid]
      temp = list(filter(lambda a: a != 0, temp))
      if len(temp) != len(set(temp)):
        return False
      else:
        return True
    
    # check if all nine 3x3 subsquares are valid (no repeating values except for zeros)
    def validSubsquares(grid):
      for row in range(0, 9, 3):
          for col in range(0,9,3):
             temp = []
             for r in range(row,row+3):
                for c in range(col, col+3):
                  if grid[r][c] != 0:
                    temp.append(grid[r][c])
             if len(temp) != len(set(temp)):
                 return False
      return True
    
    # check if the sudoku is valid
    def validSudoku(grid):
      for i in range(9):
          res1 = validRow(i, grid)
          res2 = validCol(i, grid)
          if (not res1 or not res2):
              return False
      res3 = validSubsquares(grid)
      return res3
    
    # display the message about the validity of the sudoku
    def validMessage(): 
        if(sudokuValid):
            text1 = font1.render("The sudoku is valid", 1, (59, 198, 16))
            screen.blit(text1, (155, 720)) 
        else:
            text1 = font1.render("The sudoku is not valid", 1, (255, 0, 0))        
            screen.blit(text1, (125, 720))  
        
    # display the instructions
    def instruction(): 
        text1 = font2.render("PRESS S TO SOLVE THE ENTIRE SUDOKU", 1, (0, 0, 0)) 
        text2 = font2.render("PRESS Q TO FILL A SINGLE CELL", 1, (0, 0, 0)) 
        text3 = font2.render("PRESS R TO CLEAR THE ENTIRE GRID", 1, (0, 0, 0)) 
        text4 = font2.render("PRESS D TO LOAD A RANDOM SUDOKU", 1, (0, 0, 0)) 
        text5 = font2.render("USE THE ARROW AND BACKSPACE KEYS TO ENTER/DELETE NUMBERS", 1, (0, 0, 0))
        screen.blit(text1, (20, 780))         
        screen.blit(text2, (20, 800))
        screen.blit(text3, (20, 820))
        screen.blit(text4, (20, 840)) 
        screen.blit(text5, (20, 860))

    # display certainty of the last nn-calculated move
    def displayCertainty():
        text1 = font3.render("Certainty:", 1, (0, 0, 0))        
        screen.blit(text1, (480, 790)) 
        text1 = font3.render(str(round(maxProb, 3)), 1, (0, 0, 0))        
        screen.blit(text1, (515, 825)) 
    
    # use the neural network to solve the sudoku
    # the stepByStep parameter controls whether the model will solve the entire
    # board at once or just fill in a single value
    def nnSolve(sudoku,stepByStep):
        # reshape the sudoku into the input shape the network expects
        sudoku = sudoku.reshape(1,9,9,1)
    
        while True:
            # stop when there are no zeros left in the sudoku
            if not 0 in sudoku:
                break
            
            result = model.predict(sudoku) # get predicted results (81*9 values)
            result = result.reshape(81,9)
            args = result.argmax(axis=1) # get predicted value for each cell
            sudoku = np.ravel(sudoku)
            result = result.max(axis=1) # get highest probability for each cell
            result = np.where(sudoku==0, result, 0) # set the probabilities of already filled cells to zero
            sudoku[np.argmax(result)] = args[np.argmax(result)]+1 # fill the sudoku with the value the network is most confident in
            sudoku = sudoku.reshape(1,9,9,1) 
            
            # update max probability (certainty)
            global maxProb
            maxProb = np.max(result)

            # update lastX and lastY in order to display the filled cell
            if stepByStep:
                global lastX, lastY
                lastX = np.argmax(result) % 9
                lastY = np.argmax(result) // 9
                break
        
        if not stepByStep:
            lastX = -1
            lastY = -1

        global sudokuValid
        sudokuValid = validSudoku(sudoku.reshape(9,9))
        validMessage()

        pygame.display.update() 
    
    while run:
        screen.fill((255, 255, 255))
        for event in pygame.event.get(): 
            # end game
            if event.type == pygame.QUIT: 
                run = False  

            # on mouse click select the cell that was clicked    
            if event.type == pygame.MOUSEBUTTONDOWN: 
                flagCell = True
                pos = pygame.mouse.get_pos() 
                x,y = getCoords(pos)     

            if event.type == pygame.KEYDOWN: 
                # move the selected cell in the selected direction, check for overflow
                if event.key == pygame.K_LEFT: 
                    if(x > 0):
                        x-= 1
                        flagCell = True
                if event.key == pygame.K_RIGHT: 
                    if(x < 8):
                        x+= 1
                        flagCell = True
                if event.key == pygame.K_UP:
                    if(y > 0):
                        y-= 1
                        flagCell = True
                if event.key == pygame.K_DOWN: 
                    if(y < 8):
                        y+= 1
                        flagCell = True

                # delete the most recently added value
                if event.key == pygame.K_BACKSPACE: 
                    if areCoordsValid(x,y):
                        sudoku[int(y)][int(x)] = 0
                        sudokuValid = validSudoku(sudoku)
                        flagFill = True
                        maxProb = 0
                
                # update the selected value
                if event.key == pygame.K_1: 
                    value = 1
                if event.key == pygame.K_2: 
                    value = 2    
                if event.key == pygame.K_3: 
                    value = 3
                if event.key == pygame.K_4: 
                    value = 4
                if event.key == pygame.K_5: 
                    value = 5
                if event.key == pygame.K_6: 
                    value = 6 
                if event.key == pygame.K_7: 
                    value = 7
                if event.key == pygame.K_8: 
                    value = 8
                if event.key == pygame.K_9: 
                    value = 9  

                # solve the whole sudoku at once
                if event.key == pygame.K_s: 
                    nnSolve(sudoku, stepByStep=False) 

                # solve just one cell of the sudoku
                if event.key == pygame.K_q: 
                    nnSolve(sudoku, stepByStep=True)
                
                # reset the entire sudoku
                if event.key == pygame.K_r: 
                    sudoku = np.zeros((9,9), dtype=np.int8)
                    flagFill = True
                    maxProb = 0

                # load the default sudoku
                if event.key == pygame.K_d: 
                    sudoku = np.copy(defaultSudokus[randrange(len(defaultSudokus))])
                    flagFill = True
                    maxProb = 0
        
        # fill the selected cell with the inputted value
        if value != 0 and areCoordsValid(x,y):
            sudoku[int(y)][int(x)] = value
            sudokuValid = validSudoku(sudoku)
            value = 0
            flagFill = True
            maxProb = 0

        draw(sudoku)

        # draw a boundary around the selected cell
        if flagCell and areCoordsValid(x,y): 
            drawBox()   

        # remove background color from the last filled in cell if necessary
        if flagFill:
            lastX = -1
            lastY = -1
            flagFill = False

        # update window 
        validMessage()
        instruction()
        if(maxProb != 0):
            displayCertainty()
        pygame.display.update() 
        
    pygame.quit()