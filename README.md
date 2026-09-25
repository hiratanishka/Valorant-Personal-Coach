🏆 Agent Recommendation System

At recommendation time, the player provides the current map.

Example:

Map = Haven

The system constructs one candidate profile for every agent the player has
previously used.

Haven + Sage
Haven + Killjoy
Haven + Clove
Haven + Breach
...

For every candidate agent, the system calculates the player's historical
pre-match features.

The selected model then estimates a recommendation score:

Candidate Agent
      ↓
Historical Player Features
      ↓
ML Model
      ↓
Predicted Win Probability

The candidates are sorted from highest to lowest predicted probability.

The final output is:

1. Agent A
2. Agent B
3. Agent C
4. Agent D
5. Agent E
6. Agent F

This produces the player's personalized Top 5–6 agent recommendations for
the selected map.