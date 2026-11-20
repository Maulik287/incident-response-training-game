# Incident Response Training Game

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/gwp2jgcwgr/badge.svg)](https://ask.thili.ai/b/gwp2jgcwgr)

This project develops an Incident Response Training Game for public service trainees. It leverages frequent itemset mining and language model pretraining on historical incident data to simulate real-world scenarios, allowing players to practice predicting incident sequences and making critical resource allocation decisions.

## What you are building

**Goal.** You will build a text-based simulation game where public service trainees predict incident sequences and allocate resources, learning optimal response strategies.

**Done when.** You will have modeled public incident reports as ordered event sequences, discovered frequent patterns using `thilimarketbasket`, and trained a language model to predict next incidents. The project is complete when you implement a game that uses these predictions for resource allocation and evaluates player performance.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Model public incident reports as ordered sequences of events.**  
      Hint: Prepare your incident report data into ordered sequences of events.
- [ ] **2. Discover frequent incident sequences using `thilimarketbasket`.**  
      Hint: Apply the `thilimarketbasket` technique to identify frequent incident sequences.
- [ ] **3. Train an LM to predict the next likely incident given a current situation.**  
      Hint: Use language model continued pretraining to predict the next likely incident in a sequence.
- [ ] **4. Implement a simple text-based game where the player makes resource allocation decisions based on predicted events.**  
      Hint: Implement the game logic, allowing players to make decisions based on the LM's predictions.
- [ ] **5. Evaluate player performance based on efficient resource use and accurate predictions.**  
      Hint: Define and apply metrics to evaluate player performance based on resource use and prediction accuracy.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Enhance the game's interface beyond simple text for a more engaging user experience.
- Compare player resource allocation strategies against an optimal baseline derived from the discovered frequent sequences.
- Integrate a feedback mechanism to allow the language model to learn from player decisions or new simulated incident data.
