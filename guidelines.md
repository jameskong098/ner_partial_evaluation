# Guidelines: What Makes a Good Partial Match?
* Partial match clearly refers to the same entity.
* Partial match refers to a more general entity which contains the entity mentioned and is clearly related.
* Partial match does not add or omit context which significantly changes its meaning.
* Partial match captures one of multiple valid entities in a mention.
    * This case was added because some annotations in the BTC group mulitple username mentions into one. In those cases, partial credit is acceptable for identifying one of the entities in the group.
* Partial matches must be assigned to one of two categories: "Good" or "Bad"

## Examples
| Quality | Gold             | Prediction         |
|---------|------------------|--------------------|
| Good    | The New York Times | New York Times     |
| Bad     | Premier League   | Premier League club |
| Bad     | Big Ben   | Ben |
| Good    | Sprint           | Sprint Network     |
| Good    | @eBay            | eBay               |
