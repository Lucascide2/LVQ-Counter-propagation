import numpy as np
from CounterPropagationNetwork import CounterPropagationNetwork

def main():
    print("Hello from lvq-counter-propagation!")


if __name__ == "__main__":
    np.random.seed(42)
 
    # --- Dados de exemplo: classificação binária (2 classes bem separadas) ---
    # Classe 0: região inferior-esquerda
    # Classe 1: região superior-direita
    X_train = np.array([
        [1.0, 2.0],   # classe 0
        [1.5, 1.5],   # classe 0
        [2.0, 1.0],   # classe 0
        [1.2, 1.8],   # classe 0
        [5.0, 6.0],   # classe 1
        [6.0, 5.5],   # classe 1
        [5.5, 6.5],   # classe 1
        [6.5, 5.0],   # classe 1
    ])
 
    Y_train = np.array([
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
    ])
 
    # --- Criar rede CPN ---
    # Entrada: 2 | Oculta: 8 neurônios (1 por padrão) | Saída: 1
    cpn = CounterPropagationNetwork(
        n_input    = 2,
        n_hidden   = 8,
        n_output   = 1,
        lr_instar  = 0.3,
        lr_outstar = 0.3,
    )
 
    # --- Fase 1: Treinar Instar ---
    cpn.train_instar(X_train, epochs=50)
 
    # --- Fase 2: Treinar Outstar ---
    cpn.train_outstar(X_train, Y_train, epochs=50)
 
    # --- Predições ---
    print("=== Predições ===")
    Y_pred = cpn.predict_batch(X_train)
    for i, (x, y_real, y_pred) in enumerate(zip(X_train, Y_train, Y_pred)):
        print(f"  Entrada: {x} | Esperado: {y_real[0]:.2f} | Predito: {y_pred[0]:.4f}")
 
    # --- Visualizações ---
    cpn.plot_errors()
    cpn.plot_clusters(X_train)

