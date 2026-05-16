import numpy as np
import matplotlib.pyplot as plt


class CounterPropagationNetwork:
    """
    Rede de Contrapropagação (CPN)
    
    Arquitetura:
        - Camada de Entrada: normaliza o vetor de entrada
        - Camada Oculta (Instar/Kohonen): aprendizado competitivo não supervisionado
        - Camada de Saída (Outstar/Grossberg): aprendizado supervisionado
    """
 
    def __init__(self, n_input, n_hidden, n_output, lr_instar=0.1, lr_outstar=0.1):
        """
        Parâmetros:
            n_input   : tamanho do vetor de entrada (N)
            n_hidden  : número de neurônios na camada oculta
            n_output  : tamanho do vetor de saída (M)
            lr_instar : taxa de aprendizado α (camada Instar)
            lr_outstar: taxa de aprendizado β (camada Outstar)
        """
        self.n_input    = n_input
        self.n_hidden   = n_hidden
        self.n_output   = n_output
        self.lr_instar  = lr_instar
        self.lr_outstar = lr_outstar
 
        # Pesos da camada Instar: W [n_hidden x n_input]
        # Inicializados aleatoriamente e normalizados
        self.W = np.random.rand(n_hidden, n_input)
        self.W = self._normalize_rows(self.W)
 
        # Pesos da camada Outstar: V [n_output x n_hidden]
        self.V = np.random.rand(n_output, n_hidden)
 
        # Histórico de erros
        self.instar_errors  = []
        self.outstar_errors = []
 
    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------
 
    def _normalize(self, x):
        """Normaliza um vetor para norma unitária ||x|| = 1"""
        norm = np.linalg.norm(x)
        return x / norm if norm > 0 else x
 
    def _normalize_rows(self, M):
        """Normaliza cada linha de uma matriz"""
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return M / norms
 
    def _winner(self, x):
        """
        Encontra o neurônio vencedor (winner-takes-all).
        Retorna o índice do neurônio com menor distância euclidiana a x.
        """
        distances = np.linalg.norm(self.W - x, axis=1)
        return np.argmin(distances)
 
    # ------------------------------------------------------------------
    # Treinamento — Fase 1: Instar (Kohonen)
    # ------------------------------------------------------------------
 
    def train_instar(self, X, epochs=100):
        """
        Treina a camada oculta de forma não supervisionada.
        
        Parâmetros:
            X      : matriz de entradas [n_samples x n_input]
            epochs : número de épocas
        """
        print("=== Fase 1: Treinamento Instar (Kohonen) ===")
 
        for epoch in range(epochs):
            total_error = 0
 
            for x_raw in X:
                # 1. Normalizar entrada
                x = self._normalize(x_raw)
 
                # 2. Encontrar neurônio vencedor
                k = self._winner(x)
 
                # 3. Atualizar pesos do vencedor
                delta = self.lr_instar * (x - self.W[k])
                self.W[k] += delta
 
                # 4. Renormalizar pesos do vencedor
                self.W[k] = self._normalize(self.W[k])
 
                total_error += np.linalg.norm(delta)
 
            avg_error = total_error / len(X)
            self.instar_errors.append(avg_error)
 
            if (epoch + 1) % 10 == 0:
                print(f"  Época {epoch+1}/{epochs} | Erro médio: {avg_error:.6f}")
 
        print("  Instar treinado!\n")
 
    # ------------------------------------------------------------------
    # Treinamento — Fase 2: Outstar (Grossberg)
    # ------------------------------------------------------------------
 
    def train_outstar(self, X, Y, epochs=100):
        """
        Treina a camada de saída de forma supervisionada.
        
        Parâmetros:
            X      : matriz de entradas  [n_samples x n_input]
            Y      : matriz de saídas    [n_samples x n_output]
            epochs : número de épocas
        """
        print("=== Fase 2: Treinamento Outstar (Grossberg) ===")
 
        for epoch in range(epochs):
            total_error = 0
 
            for x_raw, y in zip(X, Y):
                # 1. Normalizar entrada
                x = self._normalize(x_raw)
 
                # 2. Encontrar neurônio vencedor (Instar congelado)
                k = self._winner(x)
 
                # 3. Saída atual da Outstar
                y_pred = self.V[:, k]
 
                # 4. Calcular erro
                error = y - y_pred
 
                # 5. Atualizar apenas a coluna k da Outstar
                self.V[:, k] += self.lr_outstar * error
 
                total_error += np.linalg.norm(error)
 
            avg_error = total_error / len(X)
            self.outstar_errors.append(avg_error)
 
            if (epoch + 1) % 10 == 0:
                print(f"  Época {epoch+1}/{epochs} | Erro médio: {avg_error:.6f}")
 
        print("  Outstar treinado!\n")
 
    # ------------------------------------------------------------------
    # Predição
    # ------------------------------------------------------------------
 
    def predict(self, x_raw):
        """
        Realiza a predição para um vetor de entrada.
        Retorna a saída da Outstar.
        """
        x = self._normalize(x_raw)
        k = self._winner(x)
        return self.V[:, k]
 
    def predict_batch(self, X):
        """Predição para múltiplas entradas."""
        return np.array([self.predict(x) for x in X])
 
    # ------------------------------------------------------------------
    # Visualização
    # ------------------------------------------------------------------
 
    def plot_errors(self):
        """Plota a curva de erro das duas fases de treinamento."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Curvas de Erro — Rede CPN", fontsize=14, fontweight="bold")
 
        axes[0].plot(self.instar_errors, color="#e74c3c", linewidth=2)
        axes[0].set_title("Fase 1 — Instar (Kohonen)")
        axes[0].set_xlabel("Época")
        axes[0].set_ylabel("Erro médio")
        axes[0].grid(True, alpha=0.3)
 
        axes[1].plot(self.outstar_errors, color="#2980b9", linewidth=2)
        axes[1].set_title("Fase 2 — Outstar (Grossberg)")
        axes[1].set_xlabel("Época")
        axes[1].set_ylabel("Erro médio")
        axes[1].grid(True, alpha=0.3)
 
        plt.tight_layout()
        plt.savefig("cpn_errors.png", dpi=150)
        plt.show()
        print("Gráfico salvo em cpn_errors.png")
 
    def plot_clusters(self, X, labels=None):
        """
        Visualiza os clusters formados pela Instar (apenas para entrada 2D).
        Plota os pontos originais (sem normalização) coloridos pelo neurônio vencedor.
        """
        if self.n_input != 2:
            print("plot_clusters disponível apenas para entrada 2D.")
            return
 
        # Determinar vencedor usando entrada normalizada (como a rede faz)
        X_norm   = np.array([self._normalize(x) for x in X])
        winners  = np.array([self._winner(x) for x in X_norm])
        n_colors = len(np.unique(winners))
 
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Clusters — Camada Instar", fontsize=13, fontweight="bold")
 
        # --- Subplot 1: coordenadas ORIGINAIS ---
        sc1 = axes[0].scatter(X[:, 0], X[:, 1],
                              c=winners, cmap="tab10", alpha=0.85,
                              s=120, edgecolors="black", linewidths=0.6)
        for i, (px, py) in enumerate(X):
            axes[0].annotate(f"P{i+1}", (px, py),
                             textcoords="offset points", xytext=(6, 4), fontsize=8)
        axes[0].set_title("Pontos originais (escala real)")
        axes[0].set_xlabel("x₁")
        axes[0].set_ylabel("x₂")
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(sc1, ax=axes[0], label="Neurônio vencedor")
 
        # --- Subplot 2: coordenadas NORMALIZADAS (círculo unitário) ---
        theta = np.linspace(0, 2 * np.pi, 200)
        axes[1].plot(np.cos(theta), np.sin(theta),
                     "lightgray", linewidth=1, linestyle="--", label="Círculo unitário")
 
        sc2 = axes[1].scatter(X_norm[:, 0], X_norm[:, 1],
                              c=winners, cmap="tab10", alpha=0.85,
                              s=120, edgecolors="black", linewidths=0.6)
        for i, (px, py) in enumerate(X_norm):
            axes[1].annotate(f"P{i+1}", (px, py),
                             textcoords="offset points", xytext=(6, 4), fontsize=8)
 
        axes[1].scatter(self.W[:, 0], self.W[:, 1],
                        c="black", marker="X", s=200, zorder=5, label="Centróides (W)")
        axes[1].set_title("Pontos normalizados (círculo unitário)")
        axes[1].set_xlabel("x₁ normalizado")
        axes[1].set_ylabel("x₂ normalizado")
        axes[1].set_aspect("equal")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)
        plt.colorbar(sc2, ax=axes[1], label="Neurônio vencedor")
 
        plt.tight_layout()
        plt.savefig("cpn_clusters.png", dpi=150)
        plt.show()
        print("Gráfico salvo em cpn_clusters.png")
