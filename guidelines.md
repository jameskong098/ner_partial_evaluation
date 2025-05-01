## Notes on Partial Credit

It doesn't make sense to award full credit for partial matches, because it incentivizes mistakes. We want a model to accurately identify named entities in text, so awarding the same credit to an exact and impartial match is not a good idea. On the other hand, mention F1 penalizes close guesses, which may still be useful.

1. For partial matching metrics, first look at what % of matches are partial. The metric is only useful if there is partial credit to award.
2. What % of partial matches are good?
    - Use guidelines

# If we want to award partial credit, how much?
- At what rate are partial matches "good"?
    - Review guidelines
    - Is a constant rate of partial credit appropriate?
- Can we award partial credit based on closeness to gold annotations?
    - Overlap percentage

## Guidelines: What Makes a Good Partial Match?
* Partial match clearly refers to the same entity
* Partial match does not include extra meaningful context, which changes meaning or contains unncessessary information

# Examples
| Quality | Gold             | Prediction         |
|---------|------------------|--------------------|
| Good    | The New York Times | New York Times     |
| Bad     | Premier League   | Premier League club |
| Good    | Sprint           | Sprint Network     |
| Good    | @eBay            | eBay               |