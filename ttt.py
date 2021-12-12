import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
from random import randint
import time


class TTT(object):
    def __init__(self):
        self.player = 1
        self.board = [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]]
        self.isFull = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.isWon = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.lastMove = []
        self.moves = 0
        self.states = []
        self.wins1 = 0
        self.wins2 = 0
        self.gain_weights = [0] * 10
        self.loss_weights = [0] * 8
        self.model = Sequential()
        self.model.add(Dense(1, input_dim=81, activation='relu'))
        self.model.add(Dense(1, activation='linear', ))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        self.vals = {}
        '''self.state_pairs = {}
        self.prev_states = {}'''


    def validField(self, i, j):
        last_i = self.lastMove[0]
        last_j = self.lastMove[1]
        if self.isFull[i][j] or self.isWon[i][j]:
            return False
        if self.isWon[last_i][last_j] or self.isFull[last_i][last_j]:
            return True
        if self.lastMove[0] == i and self.lastMove[1] == j:
            return True

    def validMove(self, i, j, x, y):
        return self.validField(i, j) and not self.board[i][j][x][y]

    def move(self, i, j, x, y):
        i, j, x, y = int(i), int(j), int(x), int(y)
        self.board[i][j][x][y] = self.player
        won = self.checkWon(self.board[i][j])
        if won:
            if self.player == 1:
                self.wins1 = self.wins1 + 1
            if self.player == 2:
                self.wins2 = self.wins2 + 1
        if not self.isWon[i][j]:
            self.isWon[i][j] = self.checkWon(self.board[i][j])
        if not self.isFull[i][j]:
            self.isFull[i][j] = self.checkFull(self.board[i][j])
        self.lastMove[0], self.lastMove[1] = x, y
        self.player = 1 if self.player == 2 else 2

    def checkWon(self, field):
        for i in range(3):
            if field[i][0] != 0 and field[i][0] == field[i][1] and field[i][0] == field[i][2]:
                return field[i][0]
            if field[0][i] != 0 and field[0][i] == field[1][i] and field[0][i] == field[2][i]:
                return field[0][i]
        if field[0][0] != 0 and field[0][0] == field[1][1] and field[0][0] == field[2][2]:
            return field[0][0]
        if field[0][2] != 0 and field[0][2] == field[1][1] and field[0][2] == field[2][0]:
            return field[0][2]
        return 0

    def checkFull(self, field):
        for i in range(3):
            if 0 in field[i]:
                return 0
        return 1

    def randomMove(self):  # TODO: implement using listMoves
        validFields = []
        for i in range(3):
            for j in range(3):
                if self.validField(i, j):
                    validFields.append([i, j])
        if len(validFields) == 0:
            self.moves = 81
            return
        elif len(validFields) == 1:
            rand = 0
        else:
            rand = randint(0, len(validFields) - 1)

        ij = validFields[rand]

        validSpaces = []
        for i in range(3):
            for j in range(3):
                if not self.board[ij[0]][ij[1]][i][j]:
                    validSpaces.append([i, j])
        if len(validSpaces) == 0:
            print("moves: %s" % self.moves)
            print(self.board[ij[0]][ij[1]])
            return
        elif len(validSpaces) == 1:
            rand = 0
        else:
            rand = randint(0, len(validSpaces) - 1)

        xy = validSpaces[rand]

        self.move(ij[0], ij[1], xy[0], xy[1])

    def playRand(self):
        self.isWon = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.isFull = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.board = [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]]
        self.player = 1
        self.moves = 0
        self.lastMove = [0, 0] #start in the corner field
        status = 0
        states = []
        while self.moves < 81:
            status = self.checkWon(self.isWon)
            if status:
                return status
            self.randomMove()
            self.moves = self.moves + 1
        return self.checkWon(self.isWon)

    def listMoves(self):
        validSpaces = []
        for i in range(3):
            for j in range(3):
                if self.validField(i, j):
                    for x in range(3):
                        for y in range(3):
                            if not self.board[i][j][x][y]:
                                validSpaces.append([i, j, x, y])

        return validSpaces

    def getState(self):
        state = ""
        for i_b in range(3):
            for j_b in range(3):
                for i_f in range(3):
                    for j_f in range(3):
                        state = state + str(self.board[i_b][j_b][i_f][j_f])
        return state


    def simpleRank(self, i, j, x, y):
        gains = []
        canWinLocal = canWin(self.board[i][j], self.player, x, y)
        firstBoard1 = first_board_won(self.player)
        firstBoard2 = first_board_won(2 if self.player == 1 else 1)
        gains.append(canWinLocal and self.canWin(self.isWon, self.player, i, j)) #can win global game
        gains.append(canWinLocal) #can win local game
        gains.append(self.moves == 0 and self.isCorner(i, j)) #opening move in a corner field

        self.gain_weights = [10000, 1, 1]

        losses = []
        canLoseLocal = self.canWin(self.board[i][j], 2 if self.player == 1 else 1, x, y)
        canLoseGlobal = self.canWin(self.isWon, 2 if self.player == 1 else 1, x, y)

        self.loss_weights = [100, .5]

        rank = 0

        for i in range(len(gains)):
            if gains[i]:
                rank = rank + gain_weights[i]
        for i in range(len(losses)):
            if losses[i]:
                rank = rank + loss_weights[i]
        return rank

    def rank2(self, i, j, x, y):
        return self.gains2(i, j, x, y) + self.losses2(i, j, x, y)

    def gains2(self, i, j, x, y):
        gains = [0] * 10
        canWinLocal = self.canWin(self.board[i][j], self.player, x, y)
        if canWinLocal and self.canWin(self.isWon, self.player, i, j):
            gains[0] = 1 #can win global
        if canWinLocal:
            gains[1] = 1 #can win local
            if self.makes2_3(self.isWon, self.player, i, j):
                gains[2] = 1 #makes a 2/3
            if self.obstructs2_3(self.isWon, self.player, i, j):
                gains[3] = 1 #obstructs an opponents 2/3
            if self.wins1 == 0 and self.wins2 == 0:
                if self.isCorner(i, j):
                    gains[4] = 1 #if first move is a corner
            if self.wins1 == 1 and self.wins1 >= self.wins2:
                if self.isCorner(i, j) and self.isWon[2 if i == 0 else 0][2 if j == 0 else 0] == self.player:
                    gains[5] = 1 #if own a corner and move is a diagonal corner
            if self.wins1 == 0 and self.wins2 == 1:
                if self.isWon[0][0] or self.isWon[0][2] or self.isWon[2][0] or self.isWon[2][2]:
                    if i == 1 and j == 1:
                        gains[6] = 1 #if opp owns first corner and 2nd move is center
            if self.wins1 == 2 and self.wins2 == 0:
                if self.makes2_3(self.isWon, self.player, i, j) and self.isCorner(i, j):
                    gains[7] = 1 #if 3rd move is a corner and makes a 2/3
        if self.makes2_3(self.board[i][j], self.player, x, y):
            gains[8] = 1 #makes a 2/3 locally
        if self.obstructs2_3(self.board[i][j], self.player, x, y):
            gains[9] = 1 #obstructs a 2/3 locally

        rank = 0
        self.gain_weights = [1000000,75,75,75,75,75,75,65,100,90]

        for i in range(len(gains)):
            if gains[i]:
                rank = rank + self.gain_weights[i]

        return rank

    def losses2(self, i, j, x, y):
        opponent = 1 if self.player == 2 else 2
        losses = [0] * 8
        canLoseNext = self.canLose(self.board[x][y], opponent)
        if canLoseNext and self.canWin(self.isWon, opponent, x, y):
            losses[0] = 1 #can lose global
        if canLoseNext:
            losses[1] = 1 #can lose local
            if self.makes2_3(self.isWon, opponent, x, y):
                losses[2] = 1 #local loss makes 2/3
            if self.obstructs2_3(self.isWon, opponent, x, y):
                losses[3] = 1 #local loss obstructs player 1's 2/3
            if self.wins1 == 0:
                if self.isCorner(x, y):
                    losses[4] = 1 #local loss is first move and a corner
            if self.wins2 == 1 and self.wins2 >= self.wins1:
                if self.isCorner(x, y) and self.isWon[2 if x == 0 else 0][2 if y == 0 else 0] == opponent:
                    losses[5] = 1 #if opp owns a corner and loss is a diagonal corner
            if self.wins1 == 1 and self.wins2 == 0:
                if self.isWon[0][0] or self.isWon[0][2] or self.isWon[2][0] or self.isWon[2][2]:
                    if x == 1 and y == 1:
                        losses[6] = 1 #if own a corner and loss is the center
        if self.isWon[x][y] or self.isFull[x][y]:
            losses[7] = 1


        rank = 0

        self.loss_weights = [2,3,15,30,5,25,1, 25]

        for i in range(len(losses)):
            if losses[i]:
                rank = rank + self.loss_weights[i]

        return rank

    def makes2_3(self, field, play, i, j):
        if (field[i][0] == play and field[i][1] == 0 and field[i][2] == 0) or (field[i][0] == 0 and field[i][1] == play and field[i][2] == 0) or (field[i][0] == 0 and field[i][1] == 0 and field[i][2] == play):
            return True
        if (field[0][j] == play and field[1][j] == 0 and field[2][j] == 0) or (field[0][j] == 0 and field[1][j] == play and field[2][j] == 0) or (field[0][j] == 0 and field[1][j] == 0 and field[2][j] == play):
            return True
        if (i == j) and ((field[0][0] == play and field[1][1] == 0 and field[2][2] == 0) or (field[0][0] == 0 and field[1][1] == play and field[2][2] == 0) or (field[0][0] == 0 and field[1][1] == 0 and field[2][2] == play)):
            return True
        if (i == 0 and j == 2) or (i == 1 and j == 1) or (i == 2 and j == 0):
            if (field[0][2] == play and field[1][1] == 0 and field[2][0] == 0) or (field[0][2] == 0 and field[1][1] == play and field[2][0] == 0) or (field[0][2] == 0 and field[1][1] == 0 and field[2][0] == play):
                return True
        return False

    def obstructs2_3(self, field, play, i, j):
        play = 1 if play == 2 else 2
        if (field[i][0] == play and field[i][1] == play) or (field[i][0] == play and field[i][2] == play) or (field[i][1] == play and field[i][2] == play):
            return True
        if (field[0][j] == play and field[1][j] == play) or (field[0][j] == play and field[2][j] == play) or (field[1][j] == play and field[2][j] == play):
            return True
        if (i == j) and ((field[0][0] == play and field[1][1] == play) or (field[0][0] == play and field[2][2] == play) or (field[1][1] == play and field[2][2] == play)):
            return True
        if (i == 0 and j == 2) or (i == 1 and j == 1) or (i == 2 and j == 0):
            if (field[0][2] == play and field[1][1] == play) or (field[0][2] == play and field[2][0] == play) or (field[1][1] == play and field[2][0] == play):
                return True
        return False

    def canWin(self, field, play, i, j):
        new_field = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for x in range(3):
            for y in range(3):
                new_field[x][y] = field[x][y]
        new_field[i][j] = play
        return(self.checkWon(new_field))

    def canLose(self, field, play):
        for i in range(3):
            for j in range(3):
                if self.canWin(field, play, i, j):
                    return True
        return False

    def isCorner(self, i, j):
        return i != 1 and j != 1

    def first_board_won(self, play): ## TODO: implement using wins1 and wins2
        for row in self.isWon:
            for col in self.isWon:
                if play in col:
                    return True
        return False

    def chooseMoveSimple(self):
        ranks = self.possible_moves()
        if len(ranks) == 0:
            return
        max_rank = -1000 #smaller than smallest possible rank
        max_key = []
        for key in iter(ranks.keys()):
            cur_rank = ranks[key]
            if cur_rank >= max_rank:
                max_key = [int(key[0]), int(key[1]), int(key[2]), int(key[3])]
                max_rank = cur_rank
        self.move(max_key[0], max_key[1], max_key[2], max_key[3])

    def chooseMove2(self):
        ranks = self.possible_moves2()
        if len(ranks) == 0:
            return
        max_rank = -1000 #smaller than smallest possible rank
        max_key = []
        for key in iter(ranks.keys()):
            cur_rank = ranks[key]
            if cur_rank >= max_rank:
                max_key = [int(key[0]), int(key[1]), int(key[2]), int(key[3])]
                max_rank = cur_rank
        self.move(max_key[0], max_key[1], max_key[2], max_key[3])

    def possible_moves(self): ## TODO: binary search tree
        validSpaces = {}
        for i in range(3):
            for j in range(3):
                if self.validField(i, j):
                    for x in range(3):
                        for y in range(3):
                            if not self.board[i][j][x][y]:
                                validSpaces['%s%s%s%s' % (i, j, x, y)] = self.rank(i, j, x, y)
        return validSpaces

    def possible_moves2(self): ## TODO: binary search tree
        validSpaces = {}
        for i in range(3):
            for j in range(3):
                if self.validField(i, j):
                    for x in range(3):
                        for y in range(3):
                            if not self.board[i][j][x][y]:
                                validSpaces['%s%s%s%s' % (i, j, x, y)] = self.rank2(i, j, x, y)

        return validSpaces

    def playWC(self):
        self.isWon = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.isFull = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.board = [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]]
        self.player = 1
        self.moves = 0
        self.lastMove = [0, 0] #start in a corner field
        status = 0
        while self.moves < 81:
            status = self.checkWon(self.isWon)
            if status:
                return status
            if self.player == 1:
                self.chooseMove()
            else:
                self.randomMove()
            self.moves = self.moves + 1
        return self.checkWon(self.isWon)

    def playWC2(self):
        self.isWon = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.isFull = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.board = [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]]
        self.player = 1
        self.moves = 0
        self.lastMove = [0, 0] #start in a corner field
        status = 0
        while self.moves < 81:
            status = self.checkWon(self.isWon)
            if status:
                return status
            if self.player == 1:
                self.chooseMove2()
            else:
                self.randomMove()
            self.moves = self.moves + 1
        return self.checkWon(self.isWon)

    def activeBoardsAfterMove(self, move):
        field = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        status = 9
        for i in range(3):
            for j in range(3):
                field[i][j] = self.board[move[0]][move[1]][i][j]
                if self.isWon[i][j] or self.isFull[i][j]:
                    status = status -1

        field[move[2]][move[3]] = self.player
        if checkWon(field) or checkFull(field):
            status = status - 1
        return status

    def isOver(self):
        if self.moves == 81:
            return True
        for i in range(3):
            for j in range(3):
                if not(self.isWon[i][j] or self.isFull[i][j]):
                    return False
        return True

    def train(self, state, pY):
        '''last_state = states[-1]
        while len(states) < 41:
            states.append(last_state)
        last_pY = pYs[-1]
        while len(pYs) < 41:
            pYs.append(last_pY)'''
        np_state = np.array([self.convertState(state)])
        #np_states = np.reshape(np_states, 81)
        pY = [float(pY)]
        np_pY = np.array(pY)
        '''print(np_pY)
        print(np.shape(np_state))
        print(np.shape(np_pY))'''
        self.model.fit(np_state, np_pY, verbose=0)

    def predict(self, state):
        return self.model.predict(np.array([self.convertState(state)]))[0]

    def loadPreviousStates(self): ## TODO: implement (search tree?)
        ps = open('processeddata.csv', 'r')
        for line in ps:
            data = line.split(',')
            self.prev_states[data[2][:-1]] = [data[0], data[1]]

    def convertState(self, state):
        convertedStates = []
        for elem in state:
            if elem == '1':
                convertedStates.append(1)
            if elem == '2':
                convertedStates.append(-1)
            if elem == '0':
                convertedStates.append(0)
        return convertedStates

    def getValue(self, state):
        pY = self.predict(state)
        '''if self.canWin(self.isWon, self.player, move[0], move[1]) and self.canWin(self.board[move[0]][move[1]], self.player, move[2], move[3]):
            pY = 1.0
            self.vals[state] = pY
        elif moves == 80 or self.activeBoardsAfterMove() == 0:
            pY = .5
            self.vals[state] = pY'''
        if self.checkWon(self.isWon) == 1:
            pY = 1.0
            self.vals[state] = pY
        elif self.checkWon(self.isWon) == 2:
            pY = 0.0
            self.vals[state] = pY
        elif self.isOver():
            pY = 0.5
            self.vals[state] = pY
        return pY


    def playMCTS(self):
        self.isWon = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.isFull = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.board = [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]]
        self.vals = {}
        self.moves = 0
        self.player = 1
        self.lastMove = [0, 0] #start in a corner field
        status = 0
        data_file = open('gamedata2.csv', 'a')
        while self.moves < 81:
            status = self.checkWon(self.isWon)
            '''if status:
                return status'''
            if self.player == 1:
                self.moveMCTS(data_file)
            else:
                self.randomMove()
            self.moves = self.moves + 1
        data_file.close()
        return self.checkWon(self.isWon)

    def moveMCTS(self, file):
        possible_moves = self.listMoves()
        if len(possible_moves) == 0:
            return
        move_states = {}
        current_state = self.getState()
        for coord in possible_moves:
            index = 27 * coord[0] + 9 * coord[1] + 3 * coord[2] + coord[3]
            next_state = current_state[0:index] + str(self.player) + current_state[index + 1:]
            move_states['%s%s%s%s' % (coord[0], coord[1], coord[2], coord[3])] = self.getValue(next_state)
        best_move = max(move_states, key=move_states.get)
        self.move(best_move[0], best_move[1], best_move[2], best_move[3])
        file.write(current_state + ',' + self.getState() + ',%s\n' %move_states[best_move])
        self.learn(current_state)

    def learn(self, prev_state):
        current_state = self.getState()
        current_value = self.getValue(current_state)
        prev_value = self.predict(prev_state)[0]
        self.vals[prev_state] = prev_value + 0.2 * (current_value - prev_value) #why .2 --> tune this constant?

    def save(self):
        self.model.save('learning')

    def load(self):
        self.model = load_model('learning')

    def finishGame(self):
        '''states = []
        pYs = []
        for key in self.vals.keys():
            states.append(key)
            pYs.append(self.vals[key])
        self.train(state, pY)'''

        for state in self.vals.keys():
            self.train(state, self.vals[state])

    def nMCTS(self, n):
        self.load()
        pct1 = 0
        pct2 = 0
        pctT = 0
        total_moves = 0
        time1 = time.perf_counter()
        for i in range(n):
            status = self.playMCTS()
            total_moves = total_moves + self.moves
            if status == 1:
                pct1 = pct1 + 1
            elif status == 2:
                pct2 = pct2 + 1
            elif status == 0:
                pctT = pctT + 1
            self.finishGame()
        time2 = time.perf_counter()
        dTime = time2 - time1
        pct1 = pct1 / n
        pct2 = pct2 / n
        pctT = pctT / n
        summary = open('summary.txt', 'a')
        summary.write('%s GAMES--> x pct: %s  o pct: %s  tie pct: %s' % (n, pct1, pct2, pctT))
        summary.write('time: %s  total moves:  %s' % (dTime, total_moves))
        summary.close()
        self.save()

    def nWC2(self, n):
        pct1 = 0
        pct2 = 0
        pctT = 0
        raw_data = open('rawWC2.csv', 'a')
        total_moves = 0
        time1 = time.perf_counter()
        for i in range(n):
            status = self.playWC2()
            state = self.getState()
            total_moves = total_moves + self.moves
            #raw_data.write('%s,%s\n' % (status, state))
            if status == 1:
                pct1 = pct1 + 1
            elif status == 2:
                pct2 = pct2 + 1
            elif status == 0:
                pctT = pctT + 1
        time2 = time.perf_counter()
        dTime = time2 - time1
        raw_data.close()
        file = open('summaryWC2.txt', 'a')
        file.write('%s GAMES--> x wins: %s  o wins: %s  ties: %s' % (n, pct1, pct2, pctT))
        pct1 = pct1 / n
        pct2 = pct2 / n
        pctT = pctT / n
        file.write(' x pct: %s  o pct: %s  tie pct: %s  weights: ' % (pct1, pct2, pctT))
        for w in self.gain_weights:
            file.write('%s, ' % w)
        for w in self.loss_weights:
            file.write('%s, ' % w)
        file.write('time: %s  total moves:  %s' % (dTime, total_moves))
        file.write('\n')
        file.close()

    def nRand(self, n):
        pct1 = 0
        pct2 = 0
        pctT = 0
        raw_data = open('rawRand.csv', 'a')
        for i in range(n):
            status = self.playRand()
            state = self.getState()
            raw_data.write('%s,%s\n' % (status, state))
            if status == 1:
                pct1 = pct1 + 1
            elif status == 2:
                pct2 = pct2 + 1
            elif status == 0:
                pctT = pctT + 1
        file = open('summaryRand.txt', 'a')
        file.write('%s GAMES--> x wins: %s  o wins: %s  ties: %s' % (n, pct1, pct2, pctT))
        pct1 = pct1 / n
        pct2 = pct2 / n
        pctT = pctT / n
        file.write(' x pct: %s  o pct: %s  tie pct: %s\n ' % (pct1, pct2, pctT))
        file.close()

game = TTT()

#game.nRand(50000)
#game.nWC2(50000)
game.nMCTS(25)
