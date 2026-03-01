import numpy as np
import scipy.stats as stats
from Programming_task_Bandits import Bandits

class EpsilonGreedy:
    def __init__(self, bandit, n, eps, q_0):
        """
        :param bandit: Bandit-Instanz mit mehreren Armen.
        :param n: Anzahl der Runden.
        :param eps: Entweder ein einzelner Epsilon-Wert (float) oder eine Liste/Array der Länge n.
        :param q_0: Initiale Erwartungswerte.
        """
        if isinstance(eps, (int, float)):  # Falls eps eine Zahl ist
            eps = np.full(n, eps)  # Erzeuge ein Array mit `n` Kopien von `eps`
        elif len(eps) != n:  # Falls eps eine Liste ist, muss sie die richtige Länge haben
            raise ValueError("eps muss entweder eine Zahl oder eine Liste/Array der Länge n sein.")

        self.bandit = bandit
        self.n = n
        self.eps = np.array(eps)  # Umwandlung in NumPy-Array
        self.k = bandit.num_arms  # Anzahl der Arme
        self.arm_expectations = np.ones(self.k) * q_0  # Erwartungswerte initialisieren
        self.arm_counts = np.zeros(self.k)  # Anzahl der Ziehungen pro Arm

    def select_arm(self, i):
        """ Wählt einen Arm basierend auf dem Epsilon-Greedy-Ansatz für Runde i """
        if np.random.rand() < self.eps[i]:  # Nutze `eps[i]` für die aktuelle Runde
            return np.random.choice(self.k)  # Zufälliger Arm (Exploration)
        else:
            return np.argmax(self.arm_expectations)  # Bester Arm (Exploitation)

    def run(self):
        """ Führt das Epsilon-Greedy-Verfahren aus """
        best_arm_chosen = np.zeros(self.n)
        self.max_Q = np.max(self.bandit.means)  
        self.regret = np.zeros(self.n)
        self.norm_regret = np.zeros(self.n)
        for i in range(self.n):
            arm = self.select_arm(i)
            reward = self.bandit.pull_arm(arm)

            if i == 0: 
                self.regret[i] = max( 0, self.max_Q - reward)
            else:
                self.regret[i] = self.regret[i-1] +  max(0, self.max_Q - reward)

            self.norm_regret[i] = self.regret[i]/ max(- self.max_Q , self.max_Q)

            # Erwartungswerte aktualisieren
            self.arm_counts[arm] += 1
            self.arm_expectations[arm] = self.arm_expectations[arm] + (reward - self.arm_expectations[arm]) / self.arm_counts[arm]
            best_arm_chosen[i] = arm == np.argmax(self.bandit.means)
            probability_to_play_best_arm = np.cumsum(best_arm_chosen) / np.arange(1, self.n + 1)


        return self.arm_expectations, np.argmax(self.arm_expectations), self.regret, self.norm_regret, probability_to_play_best_arm


#Bandit erzeugen 
num_arms = 3
means = [0.3, 0.6, 0.8]  # True mean rewards for each arm
bandit = Bandits(num_arms=num_arms, dist_mode='gaussian', means=means)

#Inputvariabln festlegen 
n = 500
eps = .05
q_0 = 0

etc_algo = EpsilonGreedy(bandit, n, eps, q_0)
arm_expectations, best_arm , regret, norm_regret, _ = etc_algo.run()

print(f"Arm Expectations: {arm_expectations}")
print(f"Best Arm Selected: {best_arm}")
print(f"Regret: {regret[:5]}")
print(f"norm Regret over time: {norm_regret[:5]}")
#print(f"prob to play best arm: {prob_best}")