# Creado por Lucy
# 3-22-2023

# Librerias
import chess
import chess.svg
import torch
import torch.nn as nn
import torch.optim as optim
import random
from IPython.display import SVG, display

# Arquitectura de la red neuronal
class ChessAI(nn.Module):
    def __init__(self):
        super(ChessAI, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(64 * 8 * 8, 4096)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4096, 4672)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return x

# Funciones para convertir el el tablero de ajedrez a tensor y visceversa
def board_to_tensor(board):
    pieces = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    tensor = torch.zeros(12, 8, 8)
    for i, piece in enumerate(pieces):
        for pos in board.pieces(chess.Piece.from_symbol(piece).piece_type, chess.WHITE if piece.isupper() else chess.BLACK):
            tensor[i, pos // 8, pos % 8] = 1
    return tensor

def tensor_to_move(tensor):
    move_index = tensor.argmax().item()
    from_square = move_index // 64
    to_square = move_index % 64
    return chess.Move(from_square, to_square)

# Función apra msotrar el tablero de ajedrez
def display_board(board):
    display(SVG(chess.svg.board(board=board)))


# FUncion apra entrenar el modelo
def train(model, optimizer, criterion, board):
    model.train()

    legal_moves = list(board.legal_moves)
    random.shuffle(legal_moves)
    best_move = legal_moves[0]

    input_tensor = board_to_tensor(board).unsqueeze(0)
    output_tensor = model(input_tensor)

    target_tensor = torch.zeros(1, 4672)
    target_tensor[0, best_move.from_square * 64 + best_move.to_square] = 1

    loss = criterion(output_tensor, target_tensor)
    loss.backward()
    optimizer.step()

    return loss.item()


# Funcion para que la red neuronal haga un movimiento
def play_move(model, board):
    model.eval()
    input_tensor = board_to_tensor(board).unsqueeze(0)
    output_tensor = model(input_tensor)

    legal_moves = list(board.legal_moves)
    move_probabilities = torch.softmax(output_tensor, dim=1).squeeze()

    best_move = None
    best_prob = -1
    for move in legal_moves:
        move_index = move.from_square * 64 + move.to_square
        move_prob = move_probabilities[move_index].item()
        if move_prob > best_prob:
            best_move = move
            best_prob = move_prob

    return best_move

# Inicializacion del modelo, el optimizadro y el criterio de perdida
model = ChessAI()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Entrenar y jugar una aprtida de ajedrez
board = chess.Board()

for i in range(50):  # Limitamos el juego a 50 movimientos
    if board.is_game_over():
        break

    if board.turn == chess.WHITE:
        move = play_move(model, board)
    else:
        move = random.choice(list(board.legal_moves))  # El oponente juega al azar

    board.push(move)

display_board(board)
