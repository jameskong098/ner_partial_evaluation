# Partial Credit Analysis for NER Predictions (Dev Set)

## 1. Overlap Analysis (`partial_dev_overlap.csv`)

Partial credit is awarded if the predicted entity span and the gold entity span have any overlapping tokens.

| No. | Gold Mention                                           | Predicted Mention                          | Credit | Analysis                                      |
|-----|-------------------------------------------------------|-------------------------------------------|--------|-----------------------------------------------|
| 1   | New Hampshire                                         | Hampshire                                 | 0.5    | **Bad** |
| 2   | The Health and Safety Executive                       | Health and Safety Executive               | 0.5    | **Good**|
| 3   | Universidade Federal do Tocantins                     | Universidade Federal                      | 0.5    | **Good**:  |
| 4   | VERIZON WIRELESS                                      | VERIZON                                   | 0.5    | **Good**|
| 5   | Philips AVENT                                         | Philips                                   | 0.5    | **Good**|
| 6   | Grimsby                                               | Yume & Co Grimsby                         | 0.5    | **Bad**|
| 7   | @ firefox                                             | firefox                                   | 0.5    | **Good** |
| 8   | @ MarylandMSoccer                                     | MarylandMSoccer                           | 0.5    | **Good**|
| 9   | Venezuela #SOSVenezuela #OscarsForVenezuela           | Venezuela                                 | 0.5    | **Good**|
| 10  | @ john                                                | john                                      | 0.5    | **Good** |
| 11  | @ justinbieber @ scooterbraun @ Ludacris              | @ justinbieber                            | 0.5    | **Good**|
| 12  | SAM C                                                 | SAM                                       | 0.5    | **Good**|
| 13  | Eric Plott Eric Plott Patricia Plott E . G . Plott    | Eric Plott Eric Plott Patricia Plott    | 0.5    | **Good**  |
| 14  | @ ryanbutcher321 @ ijamesmacdonald @ ReganStevely @ GeorgioMartini | @ ryanbutcher321                          | 0.5    | **Good** |
| 15  | Blink Technologies , Inc .                       | Blink Technologies                        | 0.5    | **Good** |
| 16  | Gurudev                                               | Gurudev #yoga                             | 0.5    | **Good**|
| 17  | Kim Seung Soo Bertolak                                | Kim Seung                                 | 0.5    | **Good**|
| 18  | Sprint                                                | Sprint Network                            | 0.5    | **Good**|
| 19  | T . W . Shannon                                       | Shannon                                   | 0.5    | **Good**|
| 20  | Austin Mahone #AustinMahone                           | Austin Mahone                             | 0.5    | **Good**|
| 21  | Cassese 's                                            | Cassese                                   | 0.5    | **Good**|
| 22  | NY )                                                  | NY                                        | 0.5    | **Good**|
| 23  | @ followBack_080                                      | followBack_080                            | 0.5    | **Good**|
| 24  | @ oln_9                                               | oln_9                                     | 0.5    | **Good**|
| 25  | @ butchey12                                           | butchey12                                 | 0.5    | **Good**|
| 26  | @ Symposiu                                            | Symposiu                                  | 0.5    | **Good**|

* There were zero overlap matches which were not either a left or right boundary match

## 2. Left Boundary Matching Analysis (`partial_dev_left_bound.csv`)

Partial credit is awarded if the left boundary of the predicted mention matches the left boundary of the gold mention.


| No. | Gold Mention                                           | Predicted Mention                          | Credit | Analysis                                      |
|-----|-------------------------------------------------------|-------------------------------------------|--------|-----------------------------------------------|
| 1   | Universidade Federal do Tocantins                     | Universidade Federal                      | 0.5    | **Good**|
| 2   | VERIZON WIRELESS                                      | VERIZON                                   | 0.5    | **Good**|
| 3   | Philips AVENT                                         | Philips                                   | 0.5    | **Good**|
| 4   | Venezuela #SOSVenezuela #OscarsForVenezuela           | Venezuela                                 | 0.5    | **Good**|
| 5   | @ justinbieber @ scooterbraun @ Ludacris              | @ justinbieber                            | 0.5    | **Good**|
| 6   | SAM C                                                 | SAM                                       | 0.5    | **Good**|
| 7   | Eric Plott Eric Plott Patricia Plott E . G . Plott    | Eric Plott Eric Plott Patricia Plott    | 0.5    | **Good**|
| 8   | @ ryanbutcher321 @ ijamesmacdonald @ ReganStevely @ GeorgioMartini | @ ryanbutcher321                          | 0.5    | **Good** |
| 9   | "Blink Technologies , Inc ."                          | Blink Technologies                        | 0.5    | **Good**|
| 10  | Gurudev                                               | Gurudev #yoga                             | 0.5    | **Good**|
| 11  | Kim Seung Soo Bertolak                                | Kim Seung                                 | 0.5    | **Good**|
| 12  | Sprint                                                | Sprint Network                            | 0.5    | **Good**|
| 13  | Austin Mahone #AustinMahone                           | Austin Mahone                             | 0.5    | **Good**|
| 14  | Cassese 's                                            | Cassese                                   | 0.5    | **Good**|
| 15  | NY )                                                  | NY                                        | 0.5    | **Good**|

* We considered all of these appropriate cases for awarding partial credit. 
* Predictions 1 and 3 are OK because we decided in our guidelines that it is acceptable, for partial credit, to find a more general entity than the one in the gold mention, as long as the connection is clear.
* Prediction 8 is OK because it identifies one mention in the group - it would probably be best for those mentions to be annotated separately but we did not make that change to the dataset.
* In other cases, there is missing punctuation or variations on valid names for the same entity, so partial credit is appropriate.

## 3. Right Boundary Matching Analysis (`partial_dev_right_bound.csv`)

Partial credit is awarded if the right boundary of the predicted mention matches the right boundary of the gold mention.

| No. | Gold Mention                                           | Predicted Mention                          | Credit | Analysis                                      |
|-----|-------------------------------------------------------|-------------------------------------------|--------|-----------------------------------------------|
| 1   | New Hampshire                                         | Hampshire                                 | 0.5    | **Bad**|
| 2   | The Health and Safety Executive                       | Health and Safety Executive               | 0.5    | **Good**|
| 3   | @ firefox                                             | firefox                                   | 0.5    | **Good**|
| 4   | Grimsby                                               | Yume & Co Grimsby                         | 0.5    | **Bad**|
| 5   | @ MarylandMSoccer                                     | MarylandMSoccer                           | 0.5    | **Good**|
| 6   | @ john                                                | john                                      | 0.5    | **Good**|
| 7   | @ justinbieber @ scooterbraun @ Ludacris              | @ Ludacris                                | 0.5    | **Good**|
| 8   | @ ryanbutcher321 @ ijamesmacdonald @ ReganStevely @ GeorgioMartini | @ GeorgioMartini                          | 0.5    | **Good**|               |
| 9   | T . W . Shannon                                       | Shannon                                   | 0.5    | **Good**|
| 10  | @ followBack_080                                      | followBack_080                            | 0.5    | **Good**|
| 11  | @ oln_9                                               | oln_9                                     | 0.5    | **Good**|
| 12  | @ butchey12                                           | butchey12                                 | 0.5    | **Good**|
| 13  | @ Symposiu                                            | Symposiu                                  | 0.5    | **Good**|

* Prediction 1 is bad because Hampshire does not clearly refer to the same thing as New Hampshire.
* Prediction 4 is bad because Yume & Co is a separate mention from Grimsby.
* Other predictions involve missing @ symbols.
* We decided to accept predictions 7 and 8 because they predicted one mention in a group - again, it would probably be best for those mentions to be annotated as separate mentions since they refer to different entities.