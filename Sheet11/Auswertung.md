# Sheet 11 – Aufgabe 4: Auswertung der Actor-Critic-Algorithmen

Vergleich verschiedener Policy-Gradient- bzw. Actor-Critic-Verfahren auf den
Gymnasium *Classic-Control*-Umgebungen. Grundlage der Auswertung ist der
Datensatz [`rl_evaluation_results_realrun.csv`](rl_evaluation_results_realrun.csv)
(3300 Messpunkte), erzeugt mit [`algorithms.py`](algorithms.py).

### Datengrundlage & Datenintegrität

Es liegen drei CSV-Dateien vor; ausgewertet wird ausschließlich `realrun`:

| Datei | Inhalt | Verwendung |
|---|---|---|
| `rl_evaluation_results_realrun.csv` | vollständiger Lauf, 3300 Punkte, 100 k Steps, 20 Checkpoints | **Auswertung** |
| `rl_evaluation_results_realbackup.csv` | **byte-für-byte identisch** zu `realrun` | nur Checkpoint-Backup, keine eigene Info |
| `rl_evaluation_results_testrun.csv` | Smoke-Test: 1650 Punkte, nur 10 Mikro-Checkpoints à 1 Step | **nicht** ausgewertet (untrainiert) |

- Das `realbackup` ist die vom Skript nach jedem Seed automatisch geschriebene
  Sicherung desselben Laufs (vgl. `algorithms.py`, Zeile 129 f.). Der Abgleich
  ergibt vollständige Identität → bestätigt, dass der Lauf komplett durchlief
  und keine Messpunkte fehlen.
- Der `testrun` diente nur der Pipeline-Validierung über alle Envs/Algos
  (`total_timesteps=10`, `eval_freq=1`). Die Rewards sind praktisch
  untrainiertes Rauschen (z. B. A2C auf CartPole 10–120 je nach Seed; PPO =
  500 nur, weil CartPole schon initial trivial lösbar ist) und damit für einen
  Algorithmenvergleich ungeeignet.

---

## Versuchsaufbau (Teil a)

**Umgebungen (alle 5 Classic-Control-Envs):**

| Umgebung | Aktionsraum | Reward-Charakteristik | Success-Schwelle |
|---|---|---|---|
| CartPole-v1 | diskret | dicht, +1 pro Schritt | ≥ 500 (Maximum) |
| Acrobot-v1 | diskret | −1 pro Schritt bis Ziel | ≥ −499 (vor Timeout am Ziel) |
| MountainCar-v0 | diskret | −1 pro Schritt, sparse | ≥ −199 (Flagge vor Timeout) |
| Pendulum-v1 | kontinuierlich | dicht, negativ | ≥ −200 |
| MountainCarContinuous-v0 | kontinuierlich | sparse, +Bonus am Ziel | ≥ 90 |

**Algorithmen (8 vorimplementierte + eigene REINFORCE-Implementierung):**
A2C, DDPG, PPO, SAC, TD3 (Stable-Baselines3), ARS, TQC, TRPO (sb3-contrib)
sowie die eigene **REINFORCE**-Implementierung aus Sheet 10.
*Mini-Batch-REINFORCE wurde – wie abgesprochen – nicht ausgewertet.*

DDPG, SAC, TD3 und TQC sind reine Continuous-Control-Verfahren und wurden nur
auf den beiden kontinuierlichen Umgebungen ausgeführt.

**Protokoll:**
- 5 Seeds pro Kombination: `7, 42, 1337, 12345, 666`
- 100 000 Trainings-Steps, Evaluation alle 5 000 Steps (20 Checkpoints)
- pro Checkpoint 5 deterministische Evaluations-Episoden
- erfasste Metriken: mittlerer Reward, Std über Episoden, Success-Rate,
  kumulierte Trainingszeit, mittlere Episodenlänge

---

## Ergebnisse (Teil a)

Alle Werte = Mittelwert über die 5 Seeds **am letzten Checkpoint (100 000 Steps)**.
`Reward` = mittlerer Episoden-Reward, `σ(Seeds)` = Streuung zwischen den Seeds
(Robustheits-/Stabilitätsmaß), `Erfolg` = Success-Rate in %.

### CartPole-v1 (diskret, dicht)

| Algorithmus | Reward | σ(Seeds) | Erfolg | Zeit (s) |
|---|---:|---:|---:|---:|
| **PPO** | **500.0** | 0.0 | 100 % | 56 |
| **TRPO** | **500.0** | 0.0 | 100 % | 38 |
| ARS | 434.8 | 145.9 | 80 % | 19 |
| A2C | 417.4 | 116.4 | 60 % | 54 |
| REINFORCE | 89.4 | 56.8 | 0 % | 57 |

### Acrobot-v1 (diskret, dicht-negativ)

| Algorithmus | Reward | σ(Seeds) | Erfolg | Zeit (s) |
|---|---:|---:|---:|---:|
| **TRPO** | **−77.8** | 4.0 | 100 % | 43 |
| **PPO** | **−82.1** | 3.7 | 100 % | 68 |
| A2C | −221.1 | 170.8 | 72 % | 67 |
| REINFORCE | −260.7 | 218.8 | 60 % | 60 |
| ARS | −337.9 | 222.0 | 40 % | 28 |

### MountainCar-v0 (diskret, sparse)

| Algorithmus | Reward | σ(Seeds) | Erfolg | Zeit (s) |
|---|---:|---:|---:|---:|
| A2C / ARS / PPO / TRPO / REINFORCE | −200.0 | 0.0 | 0 % | 20–65 |

→ **Kein** Algorithmus löst MountainCar-v0 innerhalb von 100 k Steps. Der
sparse Reward (−1 pro Schritt, Bonus nur am Ziel) liefert ohne gerichtete
Exploration kein Lernsignal; alle Läufe laufen ins 200-Schritt-Timeout.

### Pendulum-v1 (kontinuierlich, dicht)

| Algorithmus | Reward | σ(Seeds) | Erfolg | Zeit (s) |
|---|---:|---:|---:|---:|
| **DDPG** | **−151.9** | 32.3 | 64 % | 414 |
| **TD3** | **−152.4** | 32.7 | 64 % | 360 |
| **SAC** | **−153.2** | 34.3 | 64 % | 476 |
| **TQC** | **−154.4** | 28.2 | 64 % | 652 |
| TRPO | −201.1 | 150.3 | 64 % | 34 |
| PPO | −952.8 | 91.3 | 0 % | 55 |
| ARS | −1127.9 | 52.9 | 0 % | 18 |
| A2C | −1335.6 | 299.3 | 0 % | 55 |
| REINFORCE | −1528.9 | 200.8 | 0 % | 66 |

→ Klare Dominanz der **Off-Policy-Verfahren** (DDPG, TD3, SAC, TQC). Die
On-Policy-/Policy-Gradient-Verfahren ohne Replay-Buffer scheitern weitgehend.

### MountainCarContinuous-v0 (kontinuierlich, sparse)

| Algorithmus | Reward | Erfolg |
|---|---:|---:|
| alle (A2C, ARS, DDPG, PPO, SAC, TD3, TQC, TRPO, REINFORCE) | ≈ 0.0 | 0 % |

→ Ebenfalls von **keinem** Algorithmus gelöst. Die Agenten lernen die
triviale „Nichts-tun“-Politik (Reward ≈ 0), um die Aktions-Kostenstrafe zu
vermeiden, finden aber das Ziel (+100) nie. Erneutes klassisches
Hard-Exploration-Problem.

### Sample-Effizienz – Steps bis zum ersten Erreichen der Schwelle

| Umgebung | schnellster | Steps | langsamster (gelöst) | Steps |
|---|---|---:|---|---:|
| CartPole-v1 | A2C | 14 000 | ARS | 52 500 |
| Acrobot-v1 | PPO | 8 000 | A2C | 26 250 |
| Pendulum-v1 | SAC | 5 000 | TRPO | 95 000 |

(REINFORCE erreicht in CartPole die Schwelle nie; auf Pendulum erreicht keiner
der On-Policy-Algorithmen die Schwelle.)

### Rechenkosten – mittlere Trainingszeit für 100 k Steps

| Algorithmus | Zeit (s) | Klasse |
|---|---:|---|
| ARS | 29 | gradientenfrei |
| TRPO | 38 | On-Policy |
| REINFORCE | 59 | On-Policy (eigene Impl.) |
| A2C | 62 | On-Policy |
| PPO | 62 | On-Policy |
| TD3 | 451 | Off-Policy |
| DDPG | 475 | Off-Policy |
| SAC | 589 | Off-Policy |
| TQC | 918 | Off-Policy |

→ Off-Policy-Verfahren sind wegen der Updates pro Step **7–15× teurer** als
On-Policy-Verfahren, dafür auf den kontinuierlichen Aufgaben deutlich besser.

### Gesamtbild

- **Beste Allrounder:** PPO und TRPO – lösen alle diskreten *lösbaren*
  Aufgaben zuverlässig (Erfolg 100 %, σ ≈ 0) und sind billig.
- **Beste auf kontinuierlich-dicht (Pendulum):** Off-Policy (DDPG/TD3/SAC/TQC).
- **Eigene REINFORCE-Implementierung:** funktioniert prinzipiell (löst Acrobot
  in 60 % der Seeds), ist aber durchgehend das schwächste und varianzreichste
  Verfahren – erwartbar, da es die reine Monte-Carlo-Policy-Gradient-Variante
  ohne Critic/Baseline ist.
- **Ungelöst von allen:** die beiden Sparse-Reward-Envs MountainCar-v0 und
  MountainCarContinuous-v0 → reine Explorations-Probleme.

---

## Vergleich mit dem tabellarischen Setting (Teil b)

**Welche Metriken im tabellarischen Setting?**
Dort (z. B. Q-Learning/SARSA auf FrozenLake, Cliff-Walking) misst man typisch:
Konvergenz der Wertfunktion/des Q-Tableaus, Bellman-Fehler, ob die *optimale*
Politik exakt gefunden wurde, sowie Reward-pro-Episode über die Lernkurve. Die
Auswertung ist dort **fast deterministisch** und **exakt** – man kann gegen die
nachweisbar optimale Lösung vergleichen.

### Beobachtete Unterschiede

| Aspekt | Tabellarisch | Hier (Function Approximation) |
|---|---|---|
| Optimum | exakt bekannt/berechenbar | unbekannt, nur relativer Vergleich |
| Konvergenz | garantiert (unter Bedingungen) | keine Garantie, oft Plateaus/Kollaps |
| Streuung über Seeds | klein | teils riesig (σ bis ~300 bei A2C/REINFORCE) |
| Erfolgsmaß | „optimale Politik gefunden?“ | Reward-Schwelle / Success-Rate |
| Hauptkostenfaktor | Episodenzahl | Wall-Clock-Zeit (29 s vs. 918 s) |

### Warum entstehen diese Unterschiede?

1. **Funktionsapproximation statt Tabelle.** Die Politik ist ein neuronales
   Netz; es gibt keine Konvergenzgarantie mehr. Ergebnisse hängen stark von
   Initialisierung und Seed ab – sichtbar an σ(Seeds) von >150 bei A2C/ARS/
   REINFORCE gegenüber 0.0 bei PPO/TRPO.
2. **Kein bekanntes Optimum.** Im tabellarischen Fall kann man „korrekt gelöst“
   definieren. Hier behelfen wir uns mit heuristischen *Success-Schwellen* – die
   sind willkürlich (z. B. Pendulum hat keinen echten „Sieg“) und machen die
   Success-Rate weniger aussagekräftig.
3. **Exploration wird zum Engpass.** Bei dichten Rewards (CartPole, Acrobot,
   Pendulum) lernen die Verfahren gut; bei sparse Rewards (beide MountainCar)
   scheitern **alle**, weil ε-greedy-/Gauss-Rauschen-Exploration das Ziel nie
   findet – im kleinen tabellarischen State-Space ist das durch Besuch aller
   Zustände kein Problem.
4. **On- vs. Off-Policy / Stichprobeneffizienz.** Im tabellarischen Setting ist
   Rechenzeit kaum Thema. Hier trennt sich der Reward-Erfolg (Off-Policy gewinnt
   auf Pendulum) deutlich vom Zeit­budget (Off-Policy 7–15× teurer) – ein
   Trade-off, den es tabellarisch so nicht gibt.

### Vorschläge für einen besseren Algorithmenvergleich

1. **Deutlich mehr Seeds + Konfidenzintervalle.** 5 Seeds sind bei σ ≈ 300 zu
   wenig; Mittelwert ± Bootstrap-CI statt Punktwerten berichten (vgl. *rliable*).
2. **Sample-Effizienz als Erstklassen-Metrik:** „Steps bis zur Schwelle“ und die
   **Fläche unter der Lernkurve (AUC)**, nicht nur Endperformance – ein
   langsames, aber stabil konvergierendes Verfahren wird sonst unterschätzt.
3. **Reward normalisieren.** Reward-Skalen sind pro Env völlig verschieden
   (+500 vs. −1500). Auf einen *normalized/human-normalized score* in [0,1]
   abbilden, um über Envs aggregieren zu können.
4. **Rechenkosten mitberichten** (Wall-Clock + Sample-Budget), da der
   Performance-Vorsprung der Off-Policy-Verfahren teuer erkauft ist.
5. **Stabilität explizit messen:** σ über Seeds, Worst-Case-Seed und
   Trainings-Kollaps-Rate – Robustheit ist hier wichtiger als im garantiert
   konvergierenden tabellarischen Fall.
6. **Längeres Training / bessere Exploration für die Sparse-Envs** (z. B.
   Reward-Shaping, mehr Steps), sonst liefern MountainCar(-Continuous) nur
   uninformative Nullergebnisse.
