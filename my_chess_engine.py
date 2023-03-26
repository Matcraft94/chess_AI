##

# Importar bibliotecas necesarias
import os
import math
import chess
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np



# Configuracion
PATH_MODEL = 'C:/Users/mated/Documents/GitHub/chess_AI/saved_models/model_win_rate_0.00'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Se está utilizando el dispositivo',device)



# Funciones y clases
def board_to_tensor(board):
    piece_map = board.piece_map()
    board_tensor = np.zeros((12, 8, 8), dtype=np.float32)

    for pos, piece in piece_map.items():
        rank, file = chess.square_rank(pos), chess.square_file(pos)
        piece_idx = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
        board_tensor[piece_idx, rank, file] = 1

    return board_tensor


# Definir el entorno de ajedrez
class ChessEnvironment:
    def __init__(self):
        self.board = chess.Board()
        
    def reset(self):
        self.board.reset()
        return self.board_to_array(self.board)
    
    def board_to_array(self, board):
        # Convierte el tablero en una matriz 8x8x12 que representa las piezas y sus posiciones.
        piece_symbols = "PRNBQKprnbqk"
        piece_indices = {symbol: i for i, symbol in enumerate(piece_symbols)}
        board_matrix = np.zeros((12, 8, 8))

        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                piece_index = piece_indices[piece.symbol()]
                row, col = divmod(i, 8)
                board_matrix[piece_index, row, col] = 1

        return board_matrix

    def step(self, move):
        # Aplicar el movimiento al tablero y devolver el nuevo estado, recompensa y si el juego ha terminado.
        game_over = False
        reward = 0

        if move in self.legal_moves():
            self.board.push(move)
            game_over = self.board.is_game_over()
            if game_over:
                reward = self.get_reward()
        else:
            raise ValueError("Illegal move")

        next_state = self.board_to_array(self.board)
        return next_state, reward, game_over

    def legal_moves(self):
        return list(self.board.legal_moves)

    # def is_game_over(self):
    #     return self.board.is_game_over()
    def is_game_over(self):  # Modifica la función para que no requiera argumentos adicionales
        return self.board.is_game_over()

    def get_reward(self):
        result = self.board.result()
        if result == "1-0":  # White wins
            return 1
        elif result == "0-1":  # Black wins
            return -1
        else:  # Draw
            return 0
    
    def get_state(self):
        return board_to_tensor(self.board)


# Crear la red neuronal

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
    
class ChessNetwork(nn.Module):
    def __init__(self, num_residual_blocks=1, num_channels=64):
        super(ChessNetwork, self).__init__()

        self.conv_input = nn.Conv2d(12, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)

        self.residual_blocks = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_residual_blocks)])

        self.conv_policy = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * 8 * 8, 4096)  # 4096 es el número máximo de movimientos legales en ajedrez

        self.conv_value = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(8 * 8, 256)
        self.fc_value2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.residual_blocks(x)

        policy = F.relu(self.bn_policy(self.conv_policy(x)))
        policy = policy.view(-1, 2 * 8 * 8)
        policy = F.softmax(self.fc_policy(policy), dim=-1)

        value = F.relu(self.bn_value(self.conv_value(x)))
        value = value.view(-1, 8 * 8)
        value = F.relu(self.fc_value1(value))
        value = torch.tanh(self.fc_value2(value))

        return policy, value

class MCTSNode:
    def __init__(self, parent, prior, action):
        self.parent = parent
        self.action = action
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.prior = prior

    def expand(self, env, action_probs):
        for move, prob in action_probs.items():
            if move not in self.children:
                self.children[move] = MCTSNode(self, prob, move)

    def is_expanded(self):
        return len(self.children) > 0

    def select_child(self):
        C = 1.0  # Parámetro de exploración
        best_score = None
        best_action = None
        best_child = None

        for action, child in self.children.items():
            score = child.get_ucb_score(C)
            if best_score is None or score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def get_ucb_score(self, C):
        Q = self.value()  # Valor medio
        U = C * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)  # Potencial de mejora
        return Q + U

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def backpropagate(self, value):
        if self.parent is not None:
            self.parent.backpropagate(value)
        self.visit_count += 1
        self.value_sum += value


def move_to_index(move):
    from_square = move.from_square
    to_square = move.to_square
    return from_square * 64 + to_square

def index_to_move(index):
    from_square = index // 64
    to_square = index % 64
    return chess.Move(from_square, to_square)

def run_mcts(model, env, num_simulations, temperature):
    root = MCTSNode(None, 1.0, None)

    for _ in range(num_simulations):
        node = root
        board_copy = env.board.copy()

        # Selección y expansión
        while node.is_expanded():
            action, node = node.select_child()
            board_copy.push(action)

        # Simulación
        if not env.is_game_over():
            legal_moves = list(board_copy.legal_moves)
            state = env.board_to_array(board_copy)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            policy, value = model(state_tensor)
            policy = policy.cpu().detach().numpy().flatten()

            if len(legal_moves) > 0:  # Verifica si hay al menos un movimiento legal
                action_probs = {move: policy[move_to_index(move)] for move in legal_moves}
                node.expand(env, action_probs)
            else:
                action_probs = {}  # Inicializa un diccionario vacío si no hay movimientos legales

            value = value.item()
        else:
            value = env.get_reward(board_copy)

        # Retroceso
        node.backpropagate(value)

    # Calcula la política final a partir del número de visitas de las acciones.
    legal_moves = list(env.board.legal_moves)  # Añade esta línea para obtener una lista de movimientos legales
    visit_counts = np.array([root.children.get(action, 0).visit_count for action in legal_moves])  # Itera sobre los movimientos legales en lugar de usar num_legal_moves

    if temperature == 0:
        action_idx = np.argmax(visit_counts)
        policy = np.zeros_like(visit_counts)
        policy[action_idx] = 1
    else:
        visit_counts = visit_counts ** (1 / temperature)
        policy = visit_counts / visit_counts.sum()

    return policy

    # max_policy_length = max(len(action) for action in root.children)
    # padded_policy = np.zeros(max_policy_length)
    # for action in root.children:
    #     padded_policy[action] = root.children[action].visit_count ** (1 / temperature)

    # padded_policy /= padded_policy.sum()

    # return padded_policy, root.value()

# Implementar el algoritmo de aprendizaje por refuerzo
def play_game(model, env, num_mcts_simulations, temperature, return_result=False):
    states = []
    policy_targets = []
    value_targets = []

    while not env.board.is_game_over():
        # Calcular la política objetivo utilizando MCTS.
        policy = run_mcts(model, env, num_mcts_simulations, temperature)

        legal_moves = list(env.legal_moves())  # Obtiene los movimientos legales antes de verificar su tamaño
        if len(legal_moves) > 0:  # Verifica si hay al menos un movimiento legal
            # print(f"Policy: {policy}")  # Agrega esta línea para verificar el contenido de la política
            action = np.random.choice(len(policy), p=policy)
            move = legal_moves[action]
            env.board.push(move)

            states.append(env.get_state().copy())

            # Asegúrate de que todas las políticas objetivo tengan la misma longitud que el número máximo de movimientos legales
            padded_policy = np.zeros(4096)
            padded_policy[:len(policy)] = policy
            policy_targets.append(padded_policy)

    result = env.board.result()

    # Calcular las recompensas basadas en el resultado.
    if result == "1-0":
        value_targets = [1] * len(states)
    elif result == "0-1":
        value_targets = [-1] * len(states)
    else:
        value_targets = [0] * len(states)

    if return_result:
        return result
    else:
        return states, policy_targets, value_targets


def play_games(model, num_games=10, num_mcts_simulations=50, temperature=1.0):
    states, policy_targets, value_targets = [], [], []

    for _ in range(num_games):
        env = ChessEnvironment()
        game_states, game_policies, game_values = play_game(model, env, num_mcts_simulations, temperature)
        states.extend(game_states)
        policy_targets.extend(game_policies)
        value_targets.extend(game_values)

    return states, policy_targets, value_targets


def compute_loss(model, states, policy_targets, value_targets, device):
    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    
    policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32).to(device)
    value_targets = torch.tensor(np.array(value_targets), dtype=torch.float32).to(device)

    policies, values = model(states)

    policy_loss = F.binary_cross_entropy(policies, policy_targets)
    value_loss = F.mse_loss(values.view(-1), value_targets)

    total_loss = policy_loss + value_loss

    return total_loss



def train(model, optimizer, epochs, evaluation_interval):
    for epoch in range(epochs):
        # Generar partidas de autojugabilidad.
        states, actions, rewards = play_games(model)
        
        # Calcular la pérdida y actualizar los pesos de la red neuronal.
        loss = compute_loss(model, states, actions, rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluar y guardar el modelo, si es necesario.
        if epoch % evaluation_interval == 0:
            evaluate_and_save_model(model)



def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()


# Entrenar el modelo
# model = ChessNetwork()
# optimizer = optim.Adam(model.parameters())
# epochs = 100
# evaluation_interval = 10

# train(model, optimizer, epochs, evaluation_interval)

model = ChessNetwork()
load_model(model, PATH_MODEL)



# Exportar el modelo entrenado a un formato compatible con UCI
class MyChessEngine:
    def __init__(self, model):
        self.model = model
        self.environment = ChessEnvironment()
        self.position = None

    def uci(self):
        print("id name MyChessEngine")
        print("id author Lucy")
        print("uciok")

    def isready(self):
        print("readyok")

    def position(self, position):
        self.position = position

    def go(self):
        move = self.calculate_next_move()
        print(f"bestmove {move}")

    def calculate_next_move(self):
        # Calcular el siguiente movimiento utilizando tu modelo PyTorch.
        pass

    def quit(self):
        pass


# Crear el bucle principal para leer comandos UCI y responder
def main():
    engine = MyChessEngine(model)

    while True:
        command = input().strip()
        if command == "uci":
            engine.uci()
        elif command == "isready":
            engine.isready()
        elif command.startswith("position"):
            position = command.split(" ")[1:]
            engine.position(position)
        elif command.startswith("go"):
            engine.go()
        elif command == "quit":
            engine.quit()
            break
        else:
            print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()

