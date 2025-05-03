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
| New Hampshire                                         | Hampshire                                 | 0.5    | Appropriate: Captures part of the entity     |
| The Health and Safety Executive                      | Health and Safety Executive               | 0.5    | Appropriate: Matches the full entity         |
| Universidade Federal do Tocantins                    | Universidade Federal                      | 0.5    | Appropriate: Captures the main part but omits context |
| VERIZON WIRELESS                                     | VERIZON                                   | 0.5    | Appropriate: Captures the main part but omits context |
| Philips AVENT                                        | Philips                                   | 0.5    | Too Generous: Overly general prediction      |
| Yume & Co                                            | Yume & Co Grimsby                         | 0.5    | Appropriate: Includes extraneous information but core entity is present |
| Grimsby                                              | Yume & Co Grimsby                         | 0.5    | Too Generous: Prediction does not match the gold entity |
| @ MarylandMSoccer                                    | MarylandMSoccer                           | 0.5    | Appropriate: Matches the entity, missing only the @ symbol |
| Venezuela #SOSVenezuela #OscarsForVenezuela          | Venezuela                                 | 0.5    | Too Generous: Missing significant context    |
| @ john                                               | john                                      | 0.5    | Appropriate: Matches the entity, missing only the @ symbol |
| @ justinbieber @ scooterbraun @ Ludacris             | @ justinbieber                            | 0.5    | Too Generous: Missing significant context    |
| Big Ben                                              | Ben                                       | 0.5    | Too Generous: "Ben" is overly general        |
| SAM C                                                | SAM                                       | 0.5    | Appropriate: Captures most of the entity     |
| Eric Plott Eric Plott Patricia Plott E . G . Plott   | Eric Plott Eric Plott Patricia Plott      | 0.5    | Appropriate: Captures most of the entity     |
| Blink Technologies , Inc .                           | Blink Technologies                        | 0.5    | Appropriate: Captures the main entity        |
| Tour De Chesapeake                                   | Chesapeake                                | 0.5    | Too Generous: Overly general prediction      |
| Gurudev                                              | Gurudev #yoga                             | 0.5    | Appropriate: Includes extraneous information but core entity is present |
| Kim Seung Soo Bertolak                               | Kim Seung                                 | 0.5    | Too Generous: Missing significant context    |
| T . W . Shannon                                      | Shannon                                   | 0.5    | Too Generous: Missing significant context    |
| Austin Mahone #AustinMahone                          | Austin Mahone                             | 0.5    | Appropriate: Captures the main entity        |
| Cassese 's                                           | Cassese                                  | 0.5    | Appropriate: Matches the entity              |
| NY )                                                 | NY                                        | 0.5    | Appropriate: Matches the entity              |
| @ followBack_080                                     | followBack_080                            | 0.5    | Appropriate: Matches the entity, missing only the @ symbol |
| @ oln_9                                              | oln_9                                     | 0.5    | Appropriate: Matches the entity, missing only the @ symbol |
| @ butchey12                                          | butchey12                                 | 0.5    | Appropriate: Matches the entity, missing only the @ symbol |
| @ Symposiu                                           | Symposiu                                  | 0.5    | Appropriate: Matches the entity, missing only the @ symbol |
| #Pirate n ' Princess Vacations                      | #Pirate                                   | 0.5    | Too Generous: Missing significant context    |

## Analysis

### Observations
1. **Exact Matches**:
   - Predictions that match the gold entity exactly or with minor differences (e.g., missing symbols like `@`) are appropriately awarded partial credit (e.g., "@ john" → "john").

2. **Partial Matches with Context Omission**:
   - Predictions that capture the main part of the entity but omit critical context (e.g., "Universidade Federal do Tocantins" → "Universidade Federal") are reasonably awarded partial credit, though a dynamic scoring system could better reflect the degree of overlap.

3. **Overly General Predictions**:
   - Predictions that are overly general (e.g., "Ben" for "Big Ben") are awarded partial credit too generously. These predictions fail to capture the specificity of the gold entity. Ben could be referring to a person's name instead of the building.

4. **Extraneous Additions**:
   - Predictions that include extraneous information but still contain the core entity (e.g., "Yume & Co Grimsby" for "Yume & Co") are appropriately awarded partial credit.

5. **Significant Context Loss**:
   - Predictions that lose significant context (e.g., "Venezuela #SOSVenezuela #OscarsForVenezuela" → "Venezuela") are awarded partial credit too generously, as the omitted context is crucial to the entity's meaning.

## Conclusion

The current partial credit system (fixed at 0.5) is a reasonable baseline but lacks nuance. A more dynamic scoring system could better reflect the quality of partial matches and improve evaluation accuracy.
