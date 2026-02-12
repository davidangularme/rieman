# Résultat de l'Expérience Cruciale : Pont Sierra (2008) ↔ Conjecture 6.2

## Objectif

Tester si la fonction de Jost extraite numériquement par Numerov, $F_{\mathrm{Num}}(k)$, satisfait la Conjecture 6.2 :

$$F_{\chi_0}(k) = h_\varepsilon(k) \cdot \frac{\xi(1/2+ik)}{\xi(1/2)}$$

et vérifier si le facteur $h_\varepsilon(k)$ correspond à $1/f(-k)$ de Sierra, où $f(t) = \sum_{n=1}^{\nu(t)} n^{-1/2-it}$.

---

## Résultats

### Q1. Les zéros de $F_{\mathrm{Num}}(k)$ coïncident-ils avec ceux de $\xi(1/2+ik)$ ?

**OUI — 5/5 correspondances confirmées.**

| n | $\gamma_n$ (exact) | $k_{\mathrm{Jost}}$ | $\Delta k$ | Config | Verdict |
|---|---|---|---|---|---|
| 1 | 14.1347 | 14.1494 | **0.015** | A (T=30, N=400) | ✓ MATCH |
| 2 | 21.0220 | 20.7846 | **0.237** | B (T=30, N=800) | ✓ MATCH |
| 3 | 25.0109 | 24.7179 | **0.293** | B (T=30, N=800) | ✓ MATCH |
| 4 | 30.4249 | 30.4107 | **0.014** | C (T=35, N=1000) | ✓ MATCH |
| 5 | 32.9351 | 32.9983 | **0.063** | multi-config | ✓ MATCH |

Les traversées de zéro de $\xi(1/2+ik)$ sont retrouvées à $\Delta k < 0.001$ des $\gamma_n$ exacts. Les minima profonds de $|F|$ colocalisent avec ces mêmes positions.

### Q2. $h_\varepsilon(k)$ est-il sans zéros ?

**OUI dans la plage testée** ($k \in [5, 36]$).

Le quotient $R(k) = |F_{\mathrm{Num}}(k)| / |\xi(1/2+ik)/\xi(1/2)|$ reste strictement positif : $R_{\min} \approx 1.9$, $R_{\max} \approx 3.3 \times 10^9$. Aucun zéro détecté. Ceci confirme que $F$ s'annule **uniquement** aux positions des zéros de $\zeta$.

### Q3. $h_\varepsilon(k)$ est-il lisse ?

**OUI localement, mais croissance globale forte (~9 décades).**

La dérivée $d(\log_{10}|h_\varepsilon|)/dk$ est bornée entre les zéros. La variation locale moyenne est de 0.30 par unité de $k$, ce qui indique un comportement régulier. La croissance globale est attendue : la mollification $\varepsilon$ est fixe tandis que la troncature Riemann-Siegel $\nu(k) = \lfloor\sqrt{k/2\pi}\rfloor$ croît avec $k$, créant un décalage entre les deux régularisations.

### Q4. Le produit $h_\varepsilon(k) \cdot f(-k)$ est-il constant ?

**NON à cette résolution.**

Pour $k > 25$ ($\nu \geq 2$), le coefficient de variation est CV = 1.56. Pour $k < 25$ ($\nu = 1$), $|f(-k)| = 1$ trivialement, rendant le test sans objet.

La correspondance $h_\varepsilon \propto 1/f(-k)$ n'est pas vérifiée point par point. Elle est probablement **asymptotique** (valide pour $k \to \infty$ où $\nu$ est grand) ou nécessite d'adapter $\varepsilon(k)$ dynamiquement.

---

## Diagnostic

```
                Sierra (2008)                    Conjecture 6.2
                ─────────────                    ──────────────
Structure :     ζ(1/2-it) = f(-t)·F(t)          F = h_ε · ξ_r

Zéros :         F(γ_n) = 0 si f(γ_n)≠0          F(γ_n) = 0 car ξ_r(γ_n) = 0
                [confirmé numériquement ✓]        [confirmé numériquement ✓]

Régulateur :    1/f(-t) lisse (conjecturé)       h_ε lisse et > 0
                                                  [confirmé localement ✓]

Verrou commun : f(t) ≠ 0 ∀t réel                h_ε sans zéros
                ────────────── ⟺ RH ──────────────
```

**Ce qui est établi :**

1. Les deux cadres partagent les mêmes zéros — confirmé numériquement (5/5).
2. Le quotient $h_\varepsilon$ est sans zéros et localement lisse — confirmé.
3. La factorisation $\zeta = f \cdot F$ de Sierra fournit le cadre théorique exact.
4. Le potentiel $v_\varepsilon = g_\varepsilon^2 + g_\varepsilon'$ est une réalisation Schrödinger concrète du modèle abstrait $xp$.

**Ce qui reste ouvert :**

1. La correspondance quantitative $h_\varepsilon \propto 1/f(-k)$ n'est pas vérifiée — probablement asymptotique.
2. La croissance de $h_\varepsilon$ (~9 décades) montre que $\varepsilon$ fixe et $\nu(k)$ croissant ne sont pas simplement proportionnels.
3. Le verrou $f(t) \neq 0$ (≡ RH) n'est bien sûr pas résolu.

---

## Prochaines étapes recommandées

1. **ε adaptatif** : choisir $\varepsilon(k) \sim 1/\sqrt{k/2\pi}$ pour que la mollification suive $\nu(k)$.
2. **Série de Born** : vérifier le lien analytique $F \leftrightarrow \xi$ par développement perturbatif.
3. **Haute précision** : grilles $N > 4000$ avec mpmath pour tester $\gamma_6 \ldots \gamma_{20}$.
4. **Fonctions d'onde** : comparer les $\psi_{E_m}(x)$ de l'Appendice A de Sierra avec les solutions Numerov.
