# Partial Credit Analysis for NER Predictions

This document analyzes whether the partial credit assigned in `partial_dev_0.5.csv` aligns with the principle of partial matching through overlap.

## Dataset Overview

The dataset contains predictions for named entity recognition (NER) tasks, where partial credit (0.5) is awarded when the prediction partially matches (through simple overlap) the gold standard. Below are the key columns:
- **gold**: The ground truth entity.
- **prediction**: The predicted entity.
- **credit**: The partial credit awarded (fixed at 0.5 for all rows).

### Sample Data

| Gold Entity                                           | Predicted Entity                          | Credit | Analysis                                      |
|-------------------------------------------------------|-------------------------------------------|--------|----------------------------------------------|
| New Hampshire                                         | Hampshire                                 | 0.5    | **Good**: Captures part of the entity        |
| The Health and Safety Executive                      | Health and Safety Executive               | 0.5    | **Good**: Matches the full entity            |
| Universidade Federal do Tocantins                    | Universidade Federal                      | 0.5    | **Good**: Captures the main part but omits context |
| VERIZON WIRELESS                                     | VERIZON                                   | 0.5    | **Good**: Captures the main part but omits context |
| Philips AVENT                                        | Philips                                   | 0.5    | **Bad**: Overly general prediction           |
| Yume & Co                                            | Yume & Co Grimsby                         | 0.5    | **Bad**: Adds extraneous information         |
| Grimsby                                              | Yume & Co Grimsby                         | 0.5    | **Bad**: Does not refer to the same entity   |
| @ MarylandMSoccer                                    | MarylandMSoccer                           | 0.5    | **Good**: Matches the entity, missing only the @ symbol |
| Venezuela #SOSVenezuela #OscarsForVenezuela          | Venezuela                                 | 0.5    | **Bad**: Missing significant context         |
| @ john                                               | john                                      | 0.5    | **Good**: Matches the entity, missing only the @ symbol |
| @ justinbieber @ scooterbraun @ Ludacris             | @ justinbieber                            | 0.5    | **Bad**: Missing significant context         |
| Big Ben                                              | Ben                                       | 0.5    | **Bad**: Missing significant context, especially since "Ben" is overly general (could be a person's name instead of the building) |
| SAM C                                                | SAM                                       | 0.5    | **Good**: Captures most of the entity        |
| Eric Plott Eric Plott Patricia Plott E . G . Plott   | Eric Plott Eric Plott Patricia Plott      | 0.5    | **Good**: Captures most of the entity        |
| Blink Technologies , Inc .                           | Blink Technologies                        | 0.5    | **Good**: Captures the main entity           |
| Tour De Chesapeake                                   | Chesapeake                                | 0.5    | **Bad**: Overly general prediction           |
| Gurudev                                              | Gurudev #yoga                             | 0.5    | **Good**: Includes extraneous information but core entity is present |
| Kim Seung Soo Bertolak                               | Kim Seung                                 | 0.5    | **Bad**: Missing significant context         |
| T . W . Shannon                                      | Shannon                                   | 0.5    | **Bad**: Missing significant context         |
| Austin Mahone #AustinMahone                          | Austin Mahone                             | 0.5    | **Good**: Captures the main entity           |
| Cassese 's                                           | Cassese                                  | 0.5    | **Good**: Matches the entity                 |
| NY )                                                 | NY                                        | 0.5    | **Good**: Matches the entity                 |
| @ followBack_080                                     | followBack_080                            | 0.5    | **Good**: Matches the entity, missing only the @ symbol |
| @ oln_9                                              | oln_9                                     | 0.5    | **Good**: Matches the entity, missing only the @ symbol |
| @ butchey12                                          | butchey12                                 | 0.5    | **Good**: Matches the entity, missing only the @ symbol |
| @ Symposiu                                           | Symposiu                                  | 0.5    | **Good**: Matches the entity, missing only the @ symbol |
| #Pirate n ' Princess Vacations                      | #Pirate                                   | 0.5    | **Bad**: Missing significant context         |



### **Good Matches**: 62.96% 
### **Bad Matches**: 37.04% 

## Analysis

### Observations
1. **Good Partial Matches**:
   - Predictions that clearly refer to the same entity as the gold annotation are considered **Good** matches (e.g., "New Hampshire" → "Hampshire").
   - Minor omissions, such as missing symbols like `@`, do not significantly affect the meaning (e.g., "@ john" → "john").

2. **Bad Partial Matches**:
   - Predictions that omit significant context or add extraneous information are considered **Bad** matches (e.g., "Venezuela #SOSVenezuela #OscarsForVenezuela" → "Venezuela").
   - Overly general predictions that could refer to multiple entities (e.g., "Big Ben" → "Ben") are also **Bad** matches.

3. **Extraneous Additions**:
   - Predictions that include unnecessary information (e.g., "Yume & Co Grimsby" → "Yume & Co") are **Bad** matches if the added information changes the meaning.

4. **Context Omission**:
   - Predictions that omit critical context but still refer to the same entity (e.g., "Universidade Federal do Tocantins" → "Universidade Federal") are acceptable but could benefit from a dynamic scoring system.

## Conclusion

The current partial credit system (fixed at 0.5) is overly simplistic and does not differentiate between **Good** and **Bad** partial matches. A more dynamic scoring system that considers overlap percentage and adherence to guidelines would provide a more accurate evaluation of model performance.
