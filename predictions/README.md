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
| New Hampshire                                         | Hampshire                                 | 0.5    | Good: Refers to the same entity              |
| The Health and Safety Executive                      | Health and Safety Executive               | 0.5    | Good: Refers to the same entity              |
| Universidade Federal do Tocantins                    | Universidade Federal                      | 0.5    | Good: Refers to the same entity but omits context |
| VERIZON WIRELESS                                     | VERIZON                                   | 0.5    | Good: Refers to the same entity but omits context |
| Philips AVENT                                        | Philips                                   | 0.5    | Bad: Overly general, could refer to other entities |
| Yume & Co                                            | Yume & Co Grimsby                         | 0.5    | Bad: Adds extraneous information            |
| Grimsby                                              | Yume & Co Grimsby                         | 0.5    | Bad: Does not refer to the same entity       |
| @ MarylandMSoccer                                    | MarylandMSoccer                           | 0.5    | Good: Refers to the same entity, missing only the @ symbol |
| Venezuela #SOSVenezuela #OscarsForVenezuela          | Venezuela                                 | 0.5    | Good: Refers to the same entity              |               |
| @ john                                               | john                                      | 0.5    | Good: Refers to the same entity, missing only the @ symbol |
| @ justinbieber @ scooterbraun @ Ludacris             | @ justinbieber                            | 0.5    | Good: Refers to the same entity                            |
| Big Ben                                              | Ben                                       | 0.5    | Good: Refers to the same entity              | |
| SAM C                                                | SAM                                       | 0.5    | Good: Refers to the same entity              |
| Eric Plott Eric Plott Patricia Plott E . G . Plott   | Eric Plott Eric Plott Patricia Plott      | 0.5    | Good: Refers to the same entity              |
| Blink Technologies , Inc .                           | Blink Technologies                        | 0.5    | Good: Refers to the same entity              |
| Tour De Chesapeake                                   | Chesapeake                                | 0.5    | Good: Refers to the same entity              | |
| Gurudev                                              | Gurudev #yoga                             | 0.5    | Good: Refers to the same entity              |            |
| Kim Seung Soo Bertolak                               | Kim Seung                                 | 0.5    | Good: Refers to the same entity              |
| T . W . Shannon                                      | Shannon                                   | 0.5    | Good: Refers to the same entity              |
| Austin Mahone #AustinMahone                          | Austin Mahone                             | 0.5    | Good: Refers to the same entity              |
| Cassese 's                                           | Cassese                                  | 0.5    | Good: Refers to the same entity              |
| NY )                                                 | NY                                        | 0.5    | Good: Refers to the same entity              |
| @ followBack_080                                     | followBack_080                            | 0.5    | Good: Refers to the same entity, missing only the @ symbol |
| @ oln_9                                              | oln_9                                     | 0.5    | Good: Refers to the same entity, missing only the @ symbol |
| @ butchey12                                          | butchey12                                 | 0.5    | Good: Refers to the same entity, missing only the @ symbol |
| @ Symposiu                                           | Symposiu                                  | 0.5    | Good: Refers to the same entity, missing only the @ symbol |
| #Pirate n ' Princess Vacations                      | #Pirate                                   | 0.5    | Good: Refers to the same entity              |               |

## Analysis

### Observations
1. **Good Partial Matches**:
   - Predictions that clearly refer to the same entity as the gold annotation are considered good matches (e.g., "New Hampshire" → "Hampshire").
   - Minor omissions, such as missing symbols like `@`, do not significantly affect the meaning (e.g., "@ john" → "john").

2. **Bad Partial Matches**:
   - Predictions that omit significant context or add extraneous information are considered bad matches (e.g., "Venezuela #SOSVenezuela #OscarsForVenezuela" → "Venezuela").

3. **Extraneous Additions**:
   - Predictions that include unnecessary information (e.g., "Yume & Co Grimsby" → "Yume & Co") are bad matches if the added information changes the meaning.

4. **Context Omission**:
   - Predictions that omit critical context but still refer to the same entity (e.g., "Universidade Federal do Tocantins" → "Universidade Federal") are acceptable but could benefit from a dynamic scoring system.
