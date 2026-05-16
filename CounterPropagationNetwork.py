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
 
        # Salva cópia dos pesos iniciais para relatório
        self.W_initial = self.W.copy()
 
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
        Retorna (saída one-hot predita, índice do neurônio vencedor).
        """
        x = self._normalize(x_raw)
        k = self._winner(x)
        return self.V[:, k], k
 
    def predict_batch(self, X):
        """Predição para múltiplas entradas. Retorna (Y_pred, winners)."""
        results = [self.predict(x) for x in X]
        Y_pred  = np.array([r[0] for r in results])
        winners = np.array([r[1] for r in results])
        return Y_pred, winners
 
    def print_report(self, X, Y):
        """
        Imprime relatório completo:
          - Pesos iniciais da Instar por cluster
          - Pesos finais  da Instar por cluster
          - Cluster vencedor e resultado para cada entrada
        """
        Y_pred, winners = self.predict_batch(X)
 
        # Quais clusters foram realmente usados
        active = np.unique(winners)
 
        sep = "─" * 60
 
        # ── Pesos iniciais ─────────────────────────────────────────────
        print(f"\n{sep}")
        print("  PESOS INICIAIS — Camada Instar (W_inicial)")
        print(sep)
        for k in active:
            w = self.W_initial[k]
            w_str = "  ".join(f"{v:+.4f}" for v in w)
            print(f"  Neurônio {k+1:>2d}:  [ {w_str} ]")
 
        # ── Pesos finais ───────────────────────────────────────────────
        print(f"\n{sep}")
        print("  PESOS FINAIS — Camada Instar (W_final)")
        print(sep)
        for k in active:
            w = self.W[k]
            w_str = "  ".join(f"{v:+.4f}" for v in w)
            print(f"  Neurônio {k+1:>2d}:  [ {w_str} ]")
 
        # ── Cluster por entrada ────────────────────────────────────────
        print(f"\n{sep}")
        print("  RESULTADOS POR ENTRADA")
        print(sep)
        header = f"  {'Entrada':<26}  {'Cluster':^10}  {'Esperado':<20}  {'Predito':<20}  {'OK?'}"
        print(header)
        print(f"  {'─'*26}  {'─'*10}  {'─'*20}  {'─'*20}  {'─'*4}")
 
        for i, (x, y_real, y_hat, k) in enumerate(zip(X, Y, Y_pred, winners)):
            x_str = f"[{', '.join(f'{v:.1f}' for v in x)}]"
            y_str = f"[{', '.join(f'{v:.0f}' for v in y_real)}]"
            p_str = f"[{', '.join(f'{v:.4f}' for v in y_hat)}]"
            ok    = "✓" if np.argmax(y_hat) == np.argmax(y_real) else "✗"
            print(f"  P{i+1:<2} {x_str:<22}  Neurônio {k+1:>2d}   {y_str:<20}  {p_str:<20}  {ok}")
 
        # ── Acurácia ───────────────────────────────────────────────────
        correct = sum(np.argmax(yh) == np.argmax(yr) for yh, yr in zip(Y_pred, Y))
        print(f"\n  Acurácia: {correct}/{len(X)} ({100*correct/len(X):.1f}%)")
        print(sep + "\n")
 
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
        Visualiza os clusters da Instar em dois subplots:
          - Esquerda : pontos originais coloridos por cluster
          - Direita  : círculo unitário com pontos normalizados e centróides
        Pontos sobrepostos no círculo são separados por jitter radial.
        """
        if self.n_input != 2:
            print("plot_clusters disponível apenas para entrada 2D.")
            return
 
        from matplotlib.lines import Line2D
 
        X_norm  = np.array([self._normalize(x) for x in X])
        winners = np.array([self._winner(x) for x in X_norm])
        active_k = np.unique(winners)
 
        PALETTE   = ["#2980b9", "#e67e22", "#27ae60", "#e74c3c",
                     "#8e44ad", "#16a085", "#d35400", "#2c3e50"]
        MARKERS   = ["o", "s", "^", "D", "P", "X", "v", "h"]
        color_map  = {k: PALETTE[k % len(PALETTE)]  for k in active_k}
        marker_map = {k: MARKERS[i % len(MARKERS)]  for i, k in enumerate(sorted(active_k))}
 
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        fig.patch.set_facecolor("#f8f9fa")
        for ax in axes:
            ax.set_facecolor("#ffffff")
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
                sp.set_color("#cccccc")
 
        # ── Subplot 1: pontos ORIGINAIS ───────────────────────────────────
        ax = axes[0]
        for i, (p, w) in enumerate(zip(X, winners)):
            c = color_map[w]; mk = marker_map[w]
            ax.scatter(p[0], p[1], color=c, marker=mk,
                       s=160, edgecolors="white", linewidths=1.5, zorder=4)
            ax.annotate(f"P{i+1}", xy=(p[0], p[1]),
                        xytext=(8, 5), textcoords="offset points",
                        fontsize=9, fontweight="bold", color=c,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec=c, lw=0.8, alpha=0.9))
 
        ax.set_title("Pontos originais", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("x₁", fontsize=10)
        ax.set_ylabel("x₂", fontsize=10)
        ax.grid(True, alpha=0.2, linestyle=":")
        mg = 1.0
        ax.set_xlim(X[:,0].min()-mg, X[:,0].max()+mg)
        ax.set_ylim(X[:,1].min()-mg, X[:,1].max()+mg)
 
        # ── Subplot 2: CÍRCULO UNITÁRIO ───────────────────────────────────
        ax2 = axes[1]
 
        # Círculo unitário
        tc = np.linspace(0, 2*np.pi, 360)
        ax2.plot(np.cos(tc), np.sin(tc), color="#aaaaaa", linewidth=1.2,
                 linestyle="--", zorder=1)
        ax2.axhline(0, color="#dddddd", lw=0.6, zorder=1)
        ax2.axvline(0, color="#dddddd", lw=0.6, zorder=1)
 
        # Centróides normalizados (losango grande, borda escura)
        W_n = np.array([self._normalize(w) for w in self.W])
        for k in active_k:
            cw = W_n[k]
            ax2.scatter(cw[0], cw[1],
                        color=color_map[k], marker="D",
                        s=220, edgecolors="#333333", linewidths=1.4, zorder=7,
                        alpha=0.85)
            # linha do centro ao centróide
            ax2.annotate("", xy=(cw[0], cw[1]), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-",
                                         color=color_map[k],
                                         lw=0.8, alpha=0.4))
 
        # Pontos normalizados com jitter radial para sobrepostos
        # Agrupa por posição arredondada e distribui em arco
        from collections import defaultdict
        groups = defaultdict(list)
        for i, (xn, yn) in enumerate(zip(X_norm[:, 0], X_norm[:, 1])):
            key = (round(xn, 3), round(yn, 3))
            groups[key].append(i)
 
        JITTER_R = 0.12   # raio do arco de separação
        for (xn, yn), idxs in groups.items():
            n = len(idxs)
            for j, i in enumerate(idxs):
                if n == 1:
                    px, py = xn, yn
                else:
                    # distribui em semicírculo perpendicular à direção do ponto
                    angle_base = np.arctan2(yn, xn) + np.pi/2
                    spread = np.linspace(-np.pi/4, np.pi/4, n)
                    a = angle_base + spread[j]
                    px = xn + JITTER_R * np.cos(a)
                    py = yn + JITTER_R * np.sin(a)
                    # linha conectando ao ponto real no círculo
                    ax2.plot([xn, px], [yn, py],
                             color=color_map[winners[i]], lw=0.8,
                             alpha=0.45, zorder=4, linestyle=":")
 
                c  = color_map[winners[i]]
                mk = marker_map[winners[i]]
                ax2.scatter(px, py, color=c, marker=mk,
                            s=150, edgecolors="white",
                            linewidths=1.3, zorder=6)
                ax2.annotate(f"P{i+1}", xy=(px, py),
                             xytext=(7, 4), textcoords="offset points",
                             fontsize=9, fontweight="bold", color=c,
                             bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                       ec=c, lw=0.8, alpha=0.9))
 
        ax2.set_title("Círculo unitário — pontos normalizados e centróides",
                      fontsize=11, fontweight="bold", pad=10)
        ax2.set_xlabel("x₁ normalizado", fontsize=10)
        ax2.set_ylabel("x₂ normalizado", fontsize=10)
        ax2.set_aspect("equal")
        ax2.set_xlim(-1.35, 1.35)
        ax2.set_ylim(-1.35, 1.35)
        ax2.grid(True, alpha=0.15, linestyle=":")
 
        # ── Legenda compartilhada ─────────────────────────────────────────
        elems = [
            Line2D([0],[0], marker=marker_map[k], color="w",
                   markerfacecolor=color_map[k], markersize=10,
                   markeredgecolor="white", label=f"Neurônio {k+1}")
            for k in sorted(active_k)
        ]
        elems.append(
            Line2D([0],[0], marker="D", color="w", markerfacecolor="gray",
                   markersize=10, markeredgecolor="#333333",
                   label="Centróide (W)")
        )
        fig.legend(handles=elems, loc="lower center",
                   ncol=len(active_k)+1, fontsize=9,
                   framealpha=0.92, edgecolor="#cccccc",
                   bbox_to_anchor=(0.5, -0.03))
 
        fig.suptitle("CPN — Visualização de Clusters (Instar)",
                     fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig("cpn_clusters.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Gráfico salvo em cpn_clusters.png")
