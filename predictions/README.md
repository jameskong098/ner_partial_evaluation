# Partial Credit Analysis for NER Predictions

This document analyzes the effectiveness of different partial credit strategies for Named Entity Recognition (NER) predictions using sample data where partial credit (0.5) was awarded.

## 1. Overlap Analysis (`partial_dev_overlap.csv`)

Partial credit is awarded if the predicted entity string and the gold entity string have any overlapping characters (case-insensitive).

### Sample Data (Overlap) - 0.5 Credit Examples

| No. | Gold Entity                                           | Predicted Entity                          | Credit | Analysis                                      |
|-----|-------------------------------------------------------|-------------------------------------------|--------|-----------------------------------------------|
| 1   | New Hampshire                                         | Hampshire                                 | 0.5    | **Bad**: Loss of context                      |
| 2   | The Health and Safety Executive                       | Health and Safety Executive               | 0.5    | **Good**: Minor omission                      |
| 3   | Universidade Federal do Tocantins                     | Universidade Federal                      | 0.5    | **Good**: Meaningful prefix                   |
| 4   | VERIZON WIRELESS                                      | VERIZON                                   | 0.5    | **Good**: Core entity                         |
| 5   | Philips AVENT                                         | Philips                                   | 0.5    | **Good**: Core entity                         |
| 6   | Grimsby                                               | Yume & Co Grimsby                         | 0.5    | **Bad**: Different entity and extraneous information                     |
| 7   | @ firefox                                             | firefox                                   | 0.5    | **Good**: Missing symbol                      |
| 8   | @ MarylandMSoccer                                     | MarylandMSoccer                           | 0.5    | **Good**: Missing symbol                      |
| 9   | Venezuela #SOSVenezuela #OscarsForVenezuela           | Venezuela                                 | 0.5    | **Good**: Core entity, missing context        |
| 10  | @ john                                                | john                                      | 0.5    | **Good**: Missing symbol                      |
| 11  | @ justinbieber @ scooterbraun @ Ludacris              | @ justinbieber                            | 0.5    | **Good**: Identified one entity               |
| 12  | SAM C                                                 | SAM                                       | 0.5    | **Good**: Core entity                         |
| 13  | Eric Plott Eric Plott Patricia Plott E . G . Plott    | Eric Plott Eric Plott Patricia Plott    | 0.5    | **Good**: Partial match, captures most        |
| 14  | @ ryanbutcher321 @ ijamesmacdonald @ ReganStevely @ GeorgioMartini | @ ryanbutcher321                          | 0.5    | **Good**: Identified one entity               |
| 15  | "Blink Technologies , Inc ."                          | Blink Technologies                        | 0.5    | **Good**: Minor omission                      |
| 16  | Gurudev                                               | Gurudev #yoga                             | 0.5    | **Good**: Gold is subset, core entity present |
| 17  | Kim Seung Soo Bertolak                                | Kim Seung                                 | 0.5    | **Good**: Meaningful prefix                   |
| 18  | Sprint                                                | Sprint Network                            | 0.5    | **Bad**: Extraneous Info |
| 19  | T . W . Shannon                                       | Shannon                                   | 0.5    | **Good**: Core entity                         |
| 20  | Austin Mahone #AustinMahone                           | Austin Mahone                             | 0.5    | **Good**: Core entity, missing context        |
| 21  | Cassese 's                                            | Cassese                                   | 0.5    | **Good**: Minor omission                      |
| 22  | NY )                                                  | NY                                        | 0.5    | **Good**: Minor omission                      |
| 23  | @ followBack_080                                      | followBack_080                            | 0.5    | **Good**: Missing symbol                      |
| 24  | @ oln_9                                               | oln_9                                     | 0.5    | **Good**: Missing symbol                      |
| 25  | @ butchey12                                           | butchey12                                 | 0.5    | **Good**: Missing symbol                      |
| 26  | @ Symposiu                                            | Symposiu                                  | 0.5    | **Good**: Missing symbol                      |

### Analysis (Overlap - 0.5 Credit Only)

- **Sample Quality**: 23 out of 26 (approx. 88.5%) examples show "Good" partial credit assignment based on semantic relevance among the 0.5 credit cases.
- **Observations**: Simple overlap often awards credit appropriately when partial credit is due (approx. 88.5% of these cases). The few "Bad" cases involve significant loss of context ("New Hampshire" -> "Hampshire") or matching a different entity ("Grimsby" -> "Yume & Co Grimsby"). Most good cases involve missing symbols, minor omissions, identifying core entities, or meaningful prefixes/suffixes.

---

## 2. Left Boundary Matching Analysis (`partial_dev_left_bound.csv`)

Partial credit is awarded if the predicted entity string matches the beginning of the gold entity string (case-insensitive).

### Sample Data (Left Boundary) - 0.5 Credit Examples

| No. | Gold Entity                                           | Predicted Entity                          | Credit | Analysis                                      |
|-----|-------------------------------------------------------|-------------------------------------------|--------|-----------------------------------------------|
| 1   | Universidade Federal do Tocantins                     | Universidade Federal                      | 0.5    | **Good**: Meaningful prefix                   |
| 2   | VERIZON WIRELESS                                      | VERIZON                                   | 0.5    | **Good**: Meaningful prefix                   |
| 3   | Philips AVENT                                         | Philips                                   | 0.5    | **Good**: Meaningful prefix                   |
| 4   | Venezuela #SOSVenezuela #OscarsForVenezuela           | Venezuela                                 | 0.5    | **Good**: Meaningful prefix                   |
| 5   | @ justinbieber @ scooterbraun @ Ludacris              | @ justinbieber                            | 0.5    | **Good**: Meaningful prefix                   |
| 6   | SAM C                                                 | SAM                                       | 0.5    | **Good**: Meaningful prefix                   |
| 7   | Eric Plott Eric Plott Patricia Plott E . G . Plott    | Eric Plott Eric Plott Patricia Plott    | 0.5    | **Good**: Meaningful prefix                   |
| 8   | @ ryanbutcher321 @ ijamesmacdonald @ ReganStevely @ GeorgioMartini | @ ryanbutcher321                          | 0.5    | **Good**: Meaningful prefix                   |
| 9   | "Blink Technologies , Inc ."                          | Blink Technologies                        | 0.5    | **Good**: Meaningful prefix                   |
| 10  | Gurudev                                               | Gurudev #yoga                             | 0.5    | **Good**: Prediction is superset              |
| 11  | Kim Seung Soo Bertolak                                | Kim Seung                                 | 0.5    | **Good**: Meaningful prefix                   |
| 12  | Sprint                                                | Sprint Network                            | 0.5    | **Good**: Prediction is superset              |
| 13  | Austin Mahone #AustinMahone                           | Austin Mahone                             | 0.5    | **Good**: Meaningful prefix                   |
| 14  | Cassese 's                                            | Cassese                                   | 0.5    | **Good**: Meaningful prefix                   |
| 15  | NY )                                                  | NY                                        | 0.5    | **Good**: Meaningful prefix                   |

### Analysis (Left Boundary - 0.5 Credit Only)

- **Sample Quality**: 15 out of 15 (100%) examples show "Good" partial credit assignment among the 0.5 credit cases.
- **Observations**: Left boundary matching appears highly effective for awarding partial credit. All cases represent meaningful prefixes or instances where the prediction is a superset starting with the gold entity. This strategy correctly captures the core entity when the prediction is a shorter, valid prefix.

---

## 3. Right Boundary Matching Analysis (`partial_dev_right_bound.csv`)

Partial credit is awarded if the predicted entity string matches the end of the gold entity string (case-insensitive).

### Sample Data (Right Boundary) - 0.5 Credit Examples

| No. | Gold Entity                                           | Predicted Entity                          | Credit | Analysis                                      |
|-----|-------------------------------------------------------|-------------------------------------------|--------|-----------------------------------------------|
| 1   | New Hampshire                                         | Hampshire                                 | 0.5    | **Good**: Meaningful suffix                   |
| 2   | The Health and Safety Executive                       | Health and Safety Executive               | 0.5    | **Good**: Meaningful suffix                   |
| 3   | @ firefox                                             | firefox                                   | 0.5    | **Good**: Missing symbol                      |
| 4   | Grimsby                                               | Yume & Co Grimsby                         | 0.5    | **Bad**: Different entity, coincidental suffix|
| 5   | @ MarylandMSoccer                                     | MarylandMSoccer                           | 0.5    | **Good**: Missing symbol                      |
| 6   | @ john                                                | john                                      | 0.5    | **Good**: Missing symbol                      |
| 7   | @ justinbieber @ scooterbraun @ Ludacris              | @ Ludacris                                | 0.5    | **Good**: Identified one entity               |
| 8   | @ ryanbutcher321 @ ijamesmacdonald @ ReganStevely @ GeorgioMartini | @ GeorgioMartini                          | 0.5    | **Good**: Identified one entity               |
| 9   | T . W . Shannon                                       | Shannon                                   | 0.5    | **Good**: Meaningful suffix                   |
| 10  | @ followBack_080                                      | followBack_080                            | 0.5    | **Good**: Missing symbol                      |
| 11  | @ oln_9                                               | oln_9                                     | 0.5    | **Good**: Missing symbol                      |
| 12  | @ butchey12                                           | butchey12                                 | 0.5    | **Good**: Missing symbol                      |
| 13  | @ Symposiu                                            | Symposiu                                  | 0.5    | **Good**: Missing symbol                      |

### Analysis (Right Boundary - 0.5 Credit Only)

- **Sample Quality**: 12 out of 13 (approx. 92.3%) examples show "Good" partial credit assignment among the 0.5 credit cases.
- **Observations**: Right boundary matching is also quite effective, similar to overlap. Most good cases involve missing prefix symbols or meaningful suffixes. The single "Bad" case involved a coincidental suffix match with a different entity. This strategy is useful for cases like "T. W. Shannon" -> "Shannon" or identifying the last entity in a list.

This directory contains CSV files detailing partial matches found during evaluation on the development set.

*   `partial_dev_overlap.csv`: Lists gold standard entities and predicted entities that have *any* token overlap (with the same entity type). Credit is 1.0 for exact matches and 0.5 for partial overlaps.
*   `partial_dev_left_bound.csv`: Lists gold standard entities and predicted entities that share the *same starting token* (with the same entity type). Credit is 1.0 for exact matches and 0.5 if only the left boundary matches.
*   `partial_dev_right_bound.csv`: Lists gold standard entities and predicted entities that share the *same ending token* (with the same entity type). Credit is 1.0 for exact matches and 0.5 if only the right boundary matches.

## Visualizations

A summary table and a comparison bar chart of the different evaluation metrics (Exact, Exact Boundary, Left Boundary, Right Boundary, Partial Overlap) are generated and saved in the `../charts/` directory relative to this README.
